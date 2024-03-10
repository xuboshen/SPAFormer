import os
import sys
import torch
import numpy as np
import importlib
sys.path.append("../")
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from quaternion import qrot
import distributed as du
from iopath.common.file_io import g_pathmgr

def printout(flog, strout):
    print(strout)
    flog.write(strout + '\n')

def ca_check_one_fun(dist1, dist2, thresh):
    ret = torch.logical_and(dist1 > thresh, dist2 > thresh)
    return ret.float()

def shape_diversity_score(shapes, network, conf, batch_size):
    cdsV1 = torch.zeros([batch_size], device=conf.device)
    cdsV2 = torch.zeros([batch_size], device=conf.device)
    for i in range(len(shapes)):
        for j in range(len(shapes)):
            shape_cd_loss_per_data = network.get_shape_cd_loss_ddp(
                shapes[i][0], shapes[i][1][:,:,3:], shapes[j][1][:,:,3:],
                shapes[i][2], shapes[i][1][:,:,:3], shapes[j][1][:,:,:3], 
                conf.device)
            cdsV1 += shape_cd_loss_per_data * ca_check_one_fun(shapes[i][4], shapes[j][4], 0.5)
            cdsV2 += shape_cd_loss_per_data * shapes[i][4] * shapes[j][4]

    return cdsV1.cpu()/len(shapes)/len(shapes), cdsV2.cpu()/len(shapes)/len(shapes)

def save_checkpoint(models, model_names, dirname, epoch=None, prepend_epoch=False, optimizers=None, optimizer_names=None, cfg=None):
    if not du.is_master_proc(cfg.num_gpus * cfg.num_shards):
        return
    if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
        raise ValueError('Number of models, model names, or optimizers does not match.')

    try:
        os.makedirs(dirname, exist_ok=True)
    except:
        pass

    for model, model_name in zip(models, model_names):
        filename = f'net_{model_name}.pth'
        if prepend_epoch:
            filename = f'{epoch}_' + filename
        if cfg.num_gpus > 1:
            torch.save(model.module.state_dict(), os.path.join(dirname, filename))
        else:
            torch.save(model.state_dict(), os.path.join(dirname, filename))
    
    if optimizers is not None:
        filename = 'checkpt.pth'
        if prepend_epoch:
            filename = f'{epoch}_' + filename
        checkpt = {'epoch': epoch}
        for opt, optimizer_name in zip(optimizers, optimizer_names):
            checkpt[f'opt_{optimizer_name}'] = opt.state_dict()
        torch.save(checkpt, os.path.join(dirname, filename))

def load_checkpoint(models, model_names, dirname, epoch=None, optimizers=None, optimizer_names=None, strict=True):
    if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
        raise ValueError('Number of models, model names, or optimizers does not match.')
    # get last epoch
    d = dirname
    names = g_pathmgr.ls(d) if g_pathmgr.exists(d) else []
    names = [int(f.split('_')[0]) for f in names if "checkpt" in f]
    if len(names) == 0:
        print("No checkpoints found in '{}'.".format(d))
        epoch = None
        return 0
    else:
        # Sort the checkpoints by epoch.
        epoch = str(sorted(names)[-1])

    for model, model_name in zip(models, model_names):
        filename = f'net_{model_name}.pth'
        if epoch is not None:
            filename = f'{epoch}_' + filename
        if hasattr(model, "module"):
            model.module.load_state_dict(torch.load(os.path.join(dirname, filename), map_location='cpu'), strict=strict)
        else:
            model.load_state_dict(torch.load(os.path.join(dirname, filename), map_location='cpu'), strict=strict)

    start_epoch = 0
    if optimizers is not None:
        filename = os.path.join(dirname, f'{epoch}_checkpt.pth')
        if os.path.exists(filename):
            checkpt = torch.load(filename, map_location='cpu')
            start_epoch = checkpt['epoch']
            for opt, optimizer_name in zip(optimizers, optimizer_names):
                opt.load_state_dict(checkpt[f'opt_{optimizer_name}'])
            print(f'resuming from checkpoint {filename}')
        else:
            response = input(f'Checkpoint {filename} not found for resuming, refine saved models instead? (y/n) ')
            if response != 'y':
                sys.exit()

    return start_epoch

def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def get_model_module(model_version):
    importlib.invalidate_caches()
    return importlib.import_module('models.' + model_version)

def pad_sequence(sequences, batch_first=False, padding_value=0.0, square=False):

    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        if square == True:
            out_dims = (len(sequences), max_len, max_len) + max_size[2:]
        else:
            out_dims = (len(sequences), max_len) + trailing_dims

    else:
        raise TypeError('only support batch_first')

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    # print(out_tensor.shape, square)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            if square == True:
                out_tensor[i, :length, :length, ...] = tensor
            else:
                out_tensor[i, :length, ...] = tensor

        else:
            raise TypeError('only support batch_first')
    
    return out_tensor

def collate_variant_input(b):
    # 分别提取两个feature
    return_list = []
    for cnt, feat in enumerate(zip(*b)):
        if cnt == 6:
            feat = pad_sequence(list(feat), batch_first=True, padding_value=0.0, square=True)
        else:
            feat = pad_sequence(list(feat), batch_first=True, padding_value=0.0)
        return_list.append(feat)
    return return_list
    
def collate_feats(b):
    return list(zip(*b))

def collate_feats_with_none(b):
    b = filter (lambda x:x is not None, b)
    return list(zip(*b))

def worker_init_fn(worker_id):
    """ The function is designed for pytorch multi-process dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.
        References:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    #print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)

# pc is N x 3, feat is B x 10-dim
def transform_pc_batch(pc, feat, anchor=False):
    batch_size = feat.size(0)
    num_point = pc.size(0)
    pc = pc.repeat(batch_size, 1, 1)
    center = feat[:, :3].unsqueeze(dim=1).repeat(1, num_point, 1)
    shape = feat[:, 3:6].unsqueeze(dim=1).repeat(1, num_point, 1)
    quat = feat[:, 6:].unsqueeze(dim=1).repeat(1, num_point, 1)
    if not anchor:
        pc = pc * shape
    pc = qrot(quat.view(-1, 4), pc.view(-1, 3)).view(batch_size, num_point, 3)
    if not anchor:
        pc = pc + center
    return pc

def get_surface_reweighting_batch(xyz, cube_num_point):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    assert cube_num_point % 6 == 0, 'ERROR: cube_num_point %d must be dividable by 6!' % cube_num_point
    np = cube_num_point // 6
    out = torch.cat([(x*y).unsqueeze(dim=1).repeat(1, np*2), \
                     (y*z).unsqueeze(dim=1).repeat(1, np*2), \
                     (x*z).unsqueeze(dim=1).repeat(1, np*2)], dim=1)
    out = out / (out.sum(dim=1).unsqueeze(dim=1) + 1e-12)
    return out


import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

def get_pc_center(pc):
    return np.mean(pc, axis=0)

def get_pc_scale(pc):
    return np.sqrt(np.max(np.sum((pc - np.mean(pc, axis=0))**2, axis=1)))

def get_pca_axes(pc):
    axes = PCA(n_components=3).fit(pc).components_
    return axes

def get_chamfer_distance(pc1, pc2):
    dist = cdist(pc1, pc2)
    error = np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))
    scale = get_pc_scale(pc1) + get_pc_scale(pc2)
    return error / scale

