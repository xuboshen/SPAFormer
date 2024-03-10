"""
    Training models
"""

import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.distributed as dist
torch.multiprocessing.set_sharing_strategy('file_system')
from subprocess import call
from data_loader import PartNetPartDataset
import utils
from torch.nn.parallel import DistributedDataParallel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import render_using_blender as render_utils
from quaternion import qrot
import ddp_utils
import distributed as du
import dataset_utils 
import model_utils
import mylogging as logging
import torch.multiprocessing as mp
import psutil
from models.utils import generate_square_subsequent_mask
from utils import shape_diversity_score

logger = logging.get_logger(__name__)
torch.autograd.set_detect_anomaly(True)

def test(conf):
    # create training and validation datasets and data loaders
    data_features = ['part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'match_ids', 'contact_points', 'sym', 'label']
    
    val_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.val_data_fn, data_features, \
                                     max_num_part=20, level=conf.level, pose_sequence=conf.pose_sequence)
    #utils.printout(conf.flog, str(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False,
                                                 pin_memory=True, \
                                                 num_workers=5, drop_last=False,
                                                 collate_fn=utils.collate_variant_input,
                                                 worker_init_fn=utils.worker_init_fn)
    
    # load network model
    model_def = utils.get_model_module(conf.model_version)

    # create models
    network = model_def.Network(conf)
    if hasattr(network, 'module'):
        network.module.load_state_dict(torch.load(conf.model_dir, map_location='cpu'))
    else:
        network.load_state_dict(torch.load(conf.model_dir, map_location='cpu'))

    
    models = [network]
    model_names = ['network']

    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    optimizers = [network_opt]
    optimizer_names = ['network_opt']

    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)
    # network_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(network_opt, T_max=conf.lr_restart)


    # send parameters to device
    for m in models:
        m.to(conf.device)
    for o in optimizers:
        utils.optimizer_to_device(o, conf.device)

    # start training
    start_time = time.time()
    #last_checkpoint_step = None
    last_val_console_log_step = None
    val_num_batch = len(val_dataloader)

    val_batches = enumerate(val_dataloader, 0)
    val_fraction_done = 0.0
    val_batch_ind = -1
    
    sum_part_cd_loss = 0
    sum_shape_cd_loss = 0
    sum_scores = 0
    sum_contact_point_loss = 0
    total_acc_num = 0
    sum_resample_shape_cd_loss = 0
    total_valid_num = 0
    total_max_count = 0
    total_total_num = 0
    sum_cdsV1_sum = 0.
    sum_cdsV2_sum = 0.
    real_val_data_set = 0
    # success rate
    total_success_rate = 0
    total_number = 0
    # validate one batch
    while val_batch_ind + 1 < val_num_batch:
        val_batch_ind, val_batch = next(val_batches)
        val_fraction_done = (val_batch_ind + 1) / val_num_batch
        if len(val_batch)==0:
            continue
        #val_step = (epoch + val_fraction_done) * train_num_batch - 1

        # log_console = not conf.no_console_log and (last_val_console_log_step is None or \
        #                                            val_step - last_val_console_log_step >= conf.console_log_interval)
        # if log_console:
        #     last_val_console_log_step = val_step

        # set models to evaluation mode

        for m in models:
            m.eval()
            
        #ipdb.set_trace()
        with torch.no_grad():
            # forward pass (including logging)
            scores = 0
            part_cd_loss, shape_cd_loss, contact_point_loss, acc_num, valid_num, max_count, total_num, success_rate, real_batch_size, cdsV1_sum, cdsV2_sum = forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
                        batch_ind=val_batch_ind, num_batch=val_num_batch,
                        start_time=start_time, \
                        log_console=1, log_tb=not conf.no_tb_log, tb_writer=None,
                        lr=network_opt.param_groups[0]['lr'])

            sum_part_cd_loss += part_cd_loss
            sum_shape_cd_loss += shape_cd_loss
            sum_contact_point_loss += contact_point_loss
            total_acc_num += acc_num
            sum_scores += scores
            sum_cdsV2_sum += cdsV2_sum
            sum_cdsV1_sum += cdsV1_sum
            total_valid_num += valid_num 
            total_max_count += max_count
            total_total_num += total_num  
            total_number += len(val_batch[data_features.index('part_pcs')])
            total_success_rate += success_rate
            
            real_val_data_set += real_batch_size
            
    total_max_count = float(total_max_count)
    total_total_num = float(total_total_num)
    total_shape_loss = sum_shape_cd_loss / real_val_data_set
    total_sum_scores = sum_scores / real_val_data_set
    total_part_loss = sum_part_cd_loss / real_val_data_set
    total_contact_loss = sum_contact_point_loss / real_val_data_set
    total_acc = total_acc_num / total_valid_num
    total_contact = total_max_count / total_total_num
    total_cdsV1 = sum_cdsV1_sum / real_val_data_set
    total_cdsV2 = sum_cdsV2_sum / real_val_data_set

    utils.printout(conf.flog, f'\nTesting on {conf.exp_suffix}, {conf.model_dir}')
    utils.printout(conf.flog, f'Category: {conf.val_data_fn}, pose_sequence: {conf.pose_sequence}')
    utils.printout(conf.flog, f'total_shape_loss, SCD:{total_shape_loss.item()}')
    utils.printout(conf.flog, f'QDS: {total_cdsV1.item()}')
    utils.printout(conf.flog, f'WQDS: {total_cdsV2.item()}')
    utils.printout(conf.flog, f'total_part_loss:{total_part_loss.item()}')
    utils.printout(conf.flog, f'total_contact_loss:{total_contact_loss.item()}')
    utils.printout(conf.flog, f'total_acc_num:{100 * total_acc_num.item()}')
    utils.printout(conf.flog, f'total_valid_num:{100 * total_valid_num.item()}')
    utils.printout(conf.flog, f'total_acc:{100 * total_acc.item()}')
    utils.printout(conf.flog, f'total_contact:{total_contact}')
    utils.printout(conf.flog, f'total_scores:{total_sum_scores}')
    utils.printout(conf.flog, f'total_success_rate:{total_success_rate}, total_num in validation: {total_number}, success_rate: {total_success_rate / total_number}')
    utils.printout(conf.flog, f'total_max_count:{total_max_count}, total_total_num:{total_total_num}\n')


    print('total_shape_loss:',total_shape_loss.item())
    print('total_part_loss:',total_part_loss.item())
    print('total_contact_loss:', total_contact_loss.item())
    print('total_acc_num:', total_acc_num)
    print('total_valid_num:', total_valid_num)
    print('total_acc:',100 * total_acc.item())
    print('total_contact:', total_contact)
    print('total_success_rate:', total_success_rate)
    print(total_max_count, total_total_num)
    

def forward(batch, data_features, network, conf, \
        is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
        log_console=False, log_tb=False, tb_writer=None, lr=None, pretrained_model=None):
    # prepare input
    input_part_pcs = batch[data_features.index('part_pcs')].cuda(non_blocking=True)           # B x P x N x 3
    input_part_valids = batch[data_features.index('part_valids')].cuda(non_blocking=True)     # B x P
    batch_size = input_part_pcs.shape[0]
    num_part = input_part_pcs.shape[1]
    num_point = input_part_pcs.shape[2]
    part_ids = batch[data_features.index('part_ids')].cuda()      # B x P 
    match_ids = batch[data_features.index('match_ids')]  
    gt_part_poses = batch[data_features.index('part_poses')].cuda()      # B x P x (3 + 4)
    contact_points = batch[data_features.index("contact_points")].to(conf.device)
    sym_info = batch[data_features.index("sym")]  # B x P x 3
    labels = batch[data_features.index('label')].cuda(non_blocking=True, device=torch.cuda.current_device())      # B x P 
    shape_id = batch[data_features.index('shape_id')]
    # # get instance label
    # if conf.use_inst_encoded:
    instance_label = torch.zeros(batch_size, num_part, 20).cuda()
    same_class_list = []
    for i in range(batch_size):
        num_class = [ 0 for i in range(160) ]
        cur_same_class_list = [[] for i in range(160)]
        for j in range(num_part):
            cur_class = int(part_ids[i][j])
            if j < input_part_valids[i].sum():
                cur_same_class_list[cur_class].append(j)
            if cur_class == 0: continue
            cur_instance = int(num_class[cur_class])
            instance_label[i][j][cur_instance] = 1
            num_class[int(part_ids[i][j])] += 1
        for i in range(cur_same_class_list.count([])):
            cur_same_class_list.remove([])
        same_class_list.append(cur_same_class_list)
    # else:
    #     instance_label = None
    # forward through the network

    repeat_times = conf.repeat_times
    array_trans_l2_loss_per_data = []
    array_rot_l2_loss_per_data = []
    array_rot_cd_loss_per_data = []
    array_total_cd_loss_per_data = []
    array_shape_cd_loss_per_data = []
    array_contact_point_loss_per_data = []
    array_acc = []
    array_success_rate = []
    array_pred_part_poses = []
    array_total_scores_per_data = []
    array_sds_cd_per_data = []

    with torch.cuda.amp.autocast(enabled=True):
        for repeat_ind in range(repeat_times):
            if conf.MoN:
                random_noise = np.random.normal(loc=0.0, scale=1, size=[batch_size, num_part, conf.random_noise_size]).astype(
                                np.float32)  # B x P x 16
                random_noise = torch.tensor(random_noise).cuda(torch.cuda.current_device())  # B x P x 16
                position_index = torch.tensor(range(0, num_part), dtype=torch.long).repeat(input_part_pcs.shape[0], 1).cuda(torch.cuda.current_device())
                if conf.use_label:
                    pred_part_poses = network(input_part_pcs, input_part_valids, None, position_index, part_ids, random_noise, instance_label=instance_label, label=labels)
                else:
                    pred_part_poses = network(input_part_pcs, input_part_valids, None, position_index, part_ids, random_noise, instance_label=instance_label)
            else:
                position_index = torch.tensor(range(0, num_part), dtype=torch.long).repeat(input_part_pcs.shape[0], 1).cuda(torch.cuda.current_device())
                if conf.use_label:
                    pred_part_poses = network(input_part_pcs, input_part_valids, None, position_index, part_ids, instance_label=instance_label, label=labels)
                else:
                    pred_part_poses = network(input_part_pcs, input_part_valids, None, position_index, part_ids, instance_label=instance_label)
            array_pred_part_poses.append(pred_part_poses)

            # matching loss
            for ind in range(len(batch[0])):
                cur_match_ids = match_ids[ind]
                for i in range(1,10):
                    need_to_match_part = []
                    for j in range(num_part):
                        if cur_match_ids[j] == i:
                            need_to_match_part.append(j)
                    if len(need_to_match_part) == 0:break
                    cur_input_pts = input_part_pcs[ind,need_to_match_part]
                    cur_pred_poses = pred_part_poses[ind,need_to_match_part]
                    cur_pred_centers = cur_pred_poses[:,:3]
                    cur_pred_quats = cur_pred_poses[:,3:]
                    cur_gt_part_poses = gt_part_poses[ind,need_to_match_part]
                    cur_gt_centers = cur_gt_part_poses[:,:3]
                    cur_gt_quats = cur_gt_part_poses[:,3:]
                    if conf.num_gpus > 1:
                        matched_pred_ids , matched_gt_ids = network.module.linear_assignment_ddp(cur_input_pts, cur_pred_centers, cur_pred_quats, cur_gt_centers, cur_gt_quats)
                    else:
                        matched_pred_ids , matched_gt_ids = network.linear_assignment_ddp(cur_input_pts, cur_pred_centers, cur_pred_quats, cur_gt_centers, cur_gt_quats)
                    match_pred_part_poses = pred_part_poses[ind,need_to_match_part][matched_pred_ids]
                    pred_part_poses[ind,need_to_match_part] = match_pred_part_poses
                    match_gt_part_poses = gt_part_poses[ind,need_to_match_part][matched_gt_ids]
                    gt_part_poses[ind,need_to_match_part] = match_gt_part_poses

            # prepare gt
            input_part_pcs = input_part_pcs[:, :, :1000, :]
            # for each type of loss, compute losses per data
            if conf.num_gpus > 1:
                trans_l2_loss_per_data = network.module.get_trans_l2_loss_ddp(pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], input_part_valids)  # B
                rot_l2_loss_per_data = network.module.get_rot_l2_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids)      # B
                rot_cd_loss_per_data = network.module.get_rot_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids, conf.device)      # B
            else:
                trans_l2_loss_per_data = network.get_trans_l2_loss_ddp(pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], input_part_valids)  # B
                rot_l2_loss_per_data = network.get_rot_l2_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids)      # B
                rot_cd_loss_per_data = network.get_rot_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids, conf.device)      # B
            # prepare gt
            input_part_pcs = input_part_pcs[:, :, :1000, :]
            total_cd_loss_per_data, acc = network.get_total_cd_loss(input_part_pcs, pred_part_poses[:, :, 3:],
                                                                    gt_part_poses[:, :, 3:],
                                                                    input_part_valids, pred_part_poses[:, :, :3],
                                                                    gt_part_poses[:, :, :3], conf.device)  # B)
            shape_cd_loss_per_data = network.get_shape_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:],
                                                            gt_part_poses[:, :, 3:],
                                                            input_part_valids, pred_part_poses[:, :, :3],
                                                            gt_part_poses[:, :, :3], conf.device)
            # contact_point_loss_per_data, count, total_num = network.get_contact_point_loss(pred_part_poses[:, :, :3],
            #                                                             pred_part_poses[:, :, 3:], contact_points, sym_info)
            contact_point_loss_per_data, count, total_num, batch_count, batch_total_num = network.get_contact_point_loss_score_pa(pred_part_poses[:, :, :3],
                                                                        pred_part_poses[:, :, 3:], contact_points, sym_info)
            
            batch_single_ca = batch_count.float() / batch_total_num.float()
            mask_nan = torch.isnan(batch_single_ca)
            batch_single_ca[mask_nan] = 0.0
            

            array_sds_cd_per_data.append([
                input_part_pcs.clone(),
                pred_part_poses[:, :, :].clone(),
                input_part_valids.clone(),
                shape_cd_loss_per_data.clone(),
                batch_single_ca.to(conf.device)
            ])
            array_trans_l2_loss_per_data.append(trans_l2_loss_per_data)
            array_rot_l2_loss_per_data.append(rot_l2_loss_per_data)
            array_rot_cd_loss_per_data.append(rot_cd_loss_per_data)
            array_total_cd_loss_per_data.append(total_cd_loss_per_data)
            array_shape_cd_loss_per_data.append(shape_cd_loss_per_data)
            array_contact_point_loss_per_data.append(contact_point_loss_per_data)
            # B x P -> B
            acc = torch.tensor(acc)
            acc = acc.sum(-1).float()  # B
            valid_number = input_part_valids.sum(-1).float().cpu()  # B
            # success rate
            success_rate = (acc == valid_number).sum()
            array_success_rate.append(success_rate)
            acc_rate = acc / valid_number
            array_acc.append(acc_rate)
            count = torch.tensor(count)
            # import pdb; pdb.set_trace()


            # try pose sequence
            try:
                save_pred_pose_sequence()
            except:
                pass

            if repeat_ind == 0:
                if not conf.use_pretrained_pct == 0:
                    res_scores = scores
                res_total_cd = total_cd_loss_per_data
                res_shape_cd = shape_cd_loss_per_data
                res_contact_point = contact_point_loss_per_data
                res_acc = acc
                res_count = count
            else:
                res_total_cd = res_total_cd.min(total_cd_loss_per_data)
                res_shape_cd = res_shape_cd.min(shape_cd_loss_per_data)
                res_contact_point = res_contact_point.min(contact_point_loss_per_data)
                res_acc = res_acc.max(acc)  # B
                res_count = res_count.max(count)
                if not conf.use_pretrained_pct == 0:
                    res_scores = res_scores.min(scores)
        
    shape_cd_loss = res_shape_cd.sum()
    total_cd_loss = res_total_cd.sum()
    contact_point_loss = res_contact_point.sum()
    acc_num = res_acc.sum()  # how many parts are right in total in a certain batch
    valid_num = input_part_valids.sum()  # how many parts in total in a certain batch
    cdsV1, cdsV2 = shape_diversity_score(array_sds_cd_per_data, network, conf, batch_size)
    cdsV1_sum = cdsV1.sum()
    cdsV2_sum = cdsV2.sum()
    data_split = 'train'
    if is_val:
        data_split = 'val'
    with torch.no_grad():
        # gen visu
        if conf.vis_number > 10000:
            is_val = False
        else:
            is_val = True
            conf.no_visu = False
        if is_val and (not conf.no_visu):
            visu_dir = os.path.join(conf.exp_dir, 'val_visu')
            out_dir = os.path.join(visu_dir, 'test_196')
            input_part_pcs_dir = os.path.join(out_dir, 'input_part_pcs')
            gt_assembly_dir = os.path.join(out_dir, 'gt_assembly')
            pred_assembly_dir = os.path.join(out_dir, 'pred_assembly')
            info_dir = os.path.join(out_dir, 'info')

            if batch_ind == 0:
                # create folders
                try:
                    os.mkdir(out_dir)
                    os.mkdir(input_part_pcs_dir)
                    os.mkdir(gt_assembly_dir)
                    os.mkdir(pred_assembly_dir)
                    os.mkdir(info_dir)
                except:
                    pass
            if batch_ind < conf.num_batch_every_visu:
                #utils.printout(conf.flog, 'Visualizing ...')

                for repeat_ind in range(repeat_times):
                    pred_center = array_pred_part_poses[repeat_ind][:, :, :3]
                    gt_center = gt_part_poses[:, :, :3]

                    # compute pred_pts and gt_pts
                    # import ipdb; ipdb.set_trace()

                    pred_pts = qrot(array_pred_part_poses[repeat_ind][:, :, 3:].unsqueeze(2).repeat(1, 1, num_point, 1),
                                    input_part_pcs) + pred_center.unsqueeze(2).repeat(1, 1, num_point, 1)
                    gt_pts = qrot(gt_part_poses[:, :, 3:].unsqueeze(2).repeat(1, 1, num_point, 1),
                                  input_part_pcs) + gt_center.unsqueeze(2).repeat(1, 1, num_point, 1)

                    for i in range(batch_size):
                        fn = 'data-%03d-%03d.png' % (batch_ind * batch_size + i, repeat_ind)

                        cur_input_part_cnt = input_part_valids[i].sum().item()
                        # print(cur_input_part_cnt)
                        cur_input_part_cnt = int(cur_input_part_cnt)
                        cur_input_part_pcs = input_part_pcs[i, :cur_input_part_cnt]
                        cur_gt_part_poses = gt_part_poses[i, :cur_input_part_cnt]
                        cur_pred_part_poses = array_pred_part_poses[repeat_ind][i, :cur_input_part_cnt]

                        pred_part_pcs = qrot(cur_pred_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1),
                                             cur_input_part_pcs) + \
                                        cur_pred_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)
                        gt_part_pcs = qrot(cur_gt_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1),
                                           cur_input_part_pcs) + \
                                      cur_gt_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)

                        part_pcs_to_visu = cur_input_part_pcs.cpu().detach().numpy()
                        render_utils.render_part_pts(os.path.join(BASE_DIR, input_part_pcs_dir, fn), part_pcs_to_visu,
                                                     blender_fn='object_centered.blend')
                        part_pcs_to_visu = pred_part_pcs.cpu().detach().numpy()
                        np.save(os.path.join(BASE_DIR, pred_assembly_dir, str(int(shape_id.cpu()))), part_pcs_to_visu)
                        print(part_pcs_to_visu.shape)
                        
                        render_utils.render_part_pts(os.path.join(BASE_DIR, pred_assembly_dir, fn), part_pcs_to_visu,
                                                     blender_fn='object_centered.blend')
                        part_pcs_to_visu = gt_part_pcs.cpu().detach().numpy()
                        render_utils.render_part_pts(os.path.join(BASE_DIR, gt_assembly_dir, fn), part_pcs_to_visu,
                                                     blender_fn='object_centered.blend')
                        np.save(os.path.join(BASE_DIR, gt_assembly_dir, str(int(shape_id.cpu()))), part_pcs_to_visu)
                        print(part_pcs_to_visu.shape)
                        
                        with open(os.path.join(info_dir, fn.replace('.png', '.txt')), 'w') as fout:
                            fout.write('shape_id: %s\n' % batch[data_features.index('shape_id')][i])
                            fout.write('num_part: %d\n' % cur_input_part_cnt)
                            fout.write('trans_l2_loss: %f\n' % array_trans_l2_loss_per_data[repeat_ind][i].item())
                            fout.write('rot_l2_loss: %f\n' % array_rot_l2_loss_per_data[repeat_ind][i].item())
                            fout.write('rot_cd_loss: %f\n' % array_rot_cd_loss_per_data[repeat_ind][i].item())
                            fout.write('total_cd_loss: %f\n' % array_total_cd_loss_per_data[repeat_ind][i].item())
                            fout.write('shape_cd_loss: %f\n' % array_shape_cd_loss_per_data[repeat_ind][i].item())
                            fout.write('contact_point_loss: %f\n' % array_contact_point_loss_per_data[repeat_ind][i].item())
                            fout.write('part_accuracy: %f\n' % array_acc[repeat_ind][i].item())
                            # fout.write('success_rate: %f\n' % array_success_rate[repeat_ind][i].item())

            # if batch_ind == conf.num_batch_every_visu - 1:
            #     # visu html
            #     utils.printout(conf.flog, 'Generating html visualization ...')
            #     sublist = 'input_part_pcs,gt_assembly,pred_assembly,info'
            #     cmd = 'cd %s && python %s . 10 htmls %s %s > /dev/null' % (out_dir, os.path.join(BASE_DIR, '../utils/gen_html_hierarchy_local.py'), sublist, sublist)
            #     call(cmd, shell=True)
            #     utils.printout(conf.flog, 'DONE')
    return total_cd_loss, shape_cd_loss, contact_point_loss, acc_num, valid_num, res_count, total_num, max(array_success_rate), batch_size, cdsV1_sum, cdsV2_sum

   
   


if __name__ == '__main__':
    mp.set_start_method('spawn')
    ### get parameters
    parser = ArgumentParser()
    
    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--model_version', type=str, help='model def file')
    parser.add_argument('--category', type=str, help='model def file')
    parser.add_argument('--train_data_fn', type=str, help='training data file that indexs all data tuples')
    parser.add_argument('--val_data_fn', type=str, help='validation data file that indexs all data tuples')

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    #parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
    parser.add_argument('--data_dir', type=str, default='../../prep_data', help='data directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--level',type=str,default='3',help='level of dataset')

    # network settings
    parser.add_argument('--feat_len', type=int, default=256)
    parser.add_argument('--max_num_part', type=int, default=20)

    # training parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)
    parser.add_argument('--iter', default = 5, type=int, help = 'times to iteration')

    # loss weights
    parser.add_argument('--loss_weight_trans_l2', type=float, default=1.0, help='loss weight')
    parser.add_argument('--loss_weight_rot_l2', type=float, default=1.0, help='loss weight')
    parser.add_argument('--loss_weight_rot_cd', type=float, default=10.0, help='loss weight')
    parser.add_argument('--loss_weight_shape_cd', type=float, default=1.0, help='loss weight')


    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10, help='number of optimization steps beween console log prints')
    parser.add_argument('--checkpoint_interval', type=int, default=300, help='number of optimization steps beween checkpoints')

    # visu
    parser.add_argument('--num_batch_every_visu', type=int, default=10, help='num batch every visu')
    parser.add_argument('--num_epoch_every_visu', type=int, default=1, help='num epoch every visu')
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')

    # ddp
    parser.add_argument('--num_gpus', type=int, default=1, help='num of gpu')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='num of gpu')
    # parser.add_argument('--init_method', type=int, default=1, help='num epoch every visu')
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )

    # new added
    parser.add_argument('--auto_resume', action='store_true', default=False, help='auto resume')
    parser.add_argument('--output_dir', default=None, help='auto resume')
    parser.add_argument('--pose_sequence', default=1, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--decoding_type', default='teacher_forcing', type=str, help='use our pose sequence or not, default is True')
    parser.add_argument('--pre_reasoning', default=False, type=bool, help='use our pose sequence or not, default is True')
    parser.add_argument('--use_pretrained_pct', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--rope', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--random_noise_size', default=32, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--repeat_times', default=10, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--use_pretrained_models', default='no', type=str, help='use our pose sequence or not, default is True')
    parser.add_argument('--oracle_scores', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--MoN', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--vis_number', default=10000000, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--use_inst_encoded', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--rand_seq', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--pr', default=1, type=float, help='use our pose sequence or not, default is True')
    parser.add_argument('--multi_cat', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--use_label', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--temperature', default=100, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--without_one_hot', default=0, type=int, help='use our pose sequence or not, default is True')

    # transformer
    parser.add_argument('--num_attention_heads', default=8, type=int, help='auto resume')
    parser.add_argument('--encoder_hidden_dim', default=16, type=int, help='auto resume')
    parser.add_argument('--encoder_dropout', default=0.1, type=float, help='auto resume')
    parser.add_argument('--encoder_activation', default='relu', type=str, help='auto resume')
    parser.add_argument('--encoder_num_layers', default=8, type=int, help='auto resume')
    parser.add_argument('--object_dropout', default=0.0, type=float, help='auto resume')
    parser.add_argument('--theta_loss_divide', default=None, help='auto resume')

    #model path
    parser.add_argument('--model_dir', type=str, help='the path of the model')

    # parse args
    conf = parser.parse_args()
    
    # conf.exp_name = f'exp-{conf.category}-encoder-level3{conf.exp_suffix}'
    try:
        conf.exp_name = f'exp-{conf.category}-{conf.model_version}-level{conf.level}{conf.exp_suffix}'
        conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)

        flog = open(os.path.join(conf.exp_dir, 'test_log.txt'), 'a+')
    except:
        try:
            conf.exp_name = f'exp-{conf.category}-{conf.model_version}-level3{conf.exp_suffix}'
            conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)

            flog = open(os.path.join(conf.exp_dir, 'test_log.txt'), 'a+')
        except:
            raise TypeError("type error")
        # conf.exp_name = f'exp-{conf.category}-{conf.model_version}-{conf.train_data_fn.split(".")[0]}-{conf.exp_suffix}'

    conf.flog = flog
    
    print("conf", conf)

    ### start training
    test(conf)