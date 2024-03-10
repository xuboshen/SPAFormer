"""
    PartNetPartDataset, pad_sequence
"""

import os
import torch
import random
import torch.utils.data as data
import numpy as np
import h5py
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import ipdb
import pickle


class PartNetPartDataset(data.Dataset):

    def __init__(self, category, data_dir, data_fn, data_features, level,\
            max_num_part=20, pose_sequence=1, rand_seq=0, pr=0):
        # store parameters
        self.data_dir = data_dir        # a data directory inside [path/to/codebase]/data/
        self.data_fn = data_fn          # a .npy data indexing file listing all data tuples to load
        self.category = category

        self.max_num_part = max_num_part
        self.max_pairs = max_num_part * (max_num_part-1) / 2
        self.level = level              # 3, the highest level
        self.file = None
        # load data
        self.data = np.load(os.path.join(self.data_dir, data_fn))
        with open('./prep_data/h5py_scripts/shape_ids_2_index.pkl', 'rb') as f:
            self.shape_ids_2_index = pickle.load(f)
        # data features
        self.data_features = data_features
        # load category semantics information
        self.part_sems = []
        self.part_sem2id = dict()

        # new parameters
        self.pose_sequence = pose_sequence
        self.rand_seq = rand_seq
        self.pr = pr
    def get_part_count(self):
        return len(self.part_sems)
        
    def __str__(self):
        strout = '[PartNetPartDataset %s %d] data_dir: %s, data_fn: %s, max_num_part: %d' % \
                (self.category, len(self), self.data_dir, self.data_fn, self.max_num_part)
        return strout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File('./prep_data/h5py_scripts/shapes_all.hdf5', 'r')
        shape_id = self.data[index]
        shape_ind = self.shape_ids_2_index[int(shape_id)]
        cur_num_part = self.file['length'][shape_ind, 0]
        part_label = self.file['label'][shape_ind, 0]
        part_label = 0 if part_label == 4 else 1
        if self.pose_sequence == 4:
            pose_sequence_fn = 'pose_sequence_ascending_diagonal_only'
        elif self.pose_sequence == 5:
            pose_sequence_fn = 'find_pose_sequence_box'
        elif self.pose_sequence == 6:
            # pr_range = [0.45, 0.45+0.35, 0.9, 1]
            pose_sequence_fn = np.random.choice(['pose_sequence_ascending_diagonal_only', 'pose_sequence_box_asending', 'pose_sequence'], p=[0.55, 0.35, 0.1])
        elif self.pose_sequence == 8:
            pose_sequence_fn = "pose_sequence_ascending_topdown"
        elif self.pose_sequence == 9:
            pose_sequence_fn = "pose_sequence_ascending_downtop"
        elif self.pose_sequence == 10:
            pose_sequence_fn = "pose_sequence_ascending_diagonal_reverse"
        elif self.pose_sequence == 7:
            pose_sequence_fn = "pose_sequence_box_asending"
        else:
            pass
        if not self.pose_sequence == 0:
            try:
                cur_pose_sequence_fn = os.path.join(self.data_dir, pose_sequence_fn + '/%s_level' % shape_id + "3" + '.txt')
                cur_pose_sequence = np.array(np.genfromtxt(cur_pose_sequence_fn, delimiter=' ', dtype=np.int)).reshape(-1)
                cur_pose_sequence = cur_pose_sequence[:cur_num_part]
                if self.pose_sequence == 7:
                    cur_pose_sequence = np.flip(cur_pose_sequence)
            except:
                try:
                    cur_pose_sequence_fn = os.path.join(self.data_dir, pose_sequence_fn + '/%s_level' % shape_id + "1" + '.txt')
                    cur_pose_sequence = np.array(np.genfromtxt(cur_pose_sequence_fn, delimiter=' ', dtype=np.int)).reshape(-1)
                    cur_pose_sequence = cur_pose_sequence[:cur_num_part]
                    if self.pose_sequence == 7:
                        cur_pose_sequence = np.flip(cur_pose_sequence)
                except:
                    cur_pose_sequence = np.array(range(len(cur_num_part)))
                    if self.pose_sequence == 7:
                        cur_pose_sequence = np.flip(cur_pose_sequence)

        if self.rand_seq == 1:
            pr = random.random()
            if not self.pose_sequence == 0 and len(cur_pose_sequence) > 2 and pr > self.pr:
                lam = np.random.beta(2, 5)
                max_x = min(len(cur_pose_sequence) - 1, int(np.floor(lam * len(cur_pose_sequence))))
                # max_x = len(cur_pose_sequence) - 1
                x = random.randint(0, max_x)
                y = random.randint(x + 1, len(cur_pose_sequence))
                seq = list(range(x, y))
                np.random.shuffle(seq)
                cur_pose_sequence[x: y] = cur_pose_sequence[seq]
            
        data_feats = ()

        for feat in self.data_features:
        
        
                
            if feat == 'contact_points':
                # cur_contact_data_fn = os.path.join(self.data_dir, 'contact_points/pairs_with_contact_points_%s_level' % shape_id + self.level + '.npy')
                # cur_contacts = np.load(cur_contact_data_fn,allow_pickle=True)
                cur_contacts = self.file['contact_matrix'][shape_ind, :cur_num_part, :cur_num_part, :]
                if not self.pose_sequence == 0:
                    cur_contacts = cur_contacts[:, cur_pose_sequence]
                    cur_contacts = cur_contacts[cur_pose_sequence]
                out = torch.from_numpy(cur_contacts).float()
                data_feats = data_feats + (out,)

            elif feat == 'label':
                out = np.ones((1), dtype=np.int32)
                out[0] = part_label
                out = torch.from_numpy(out).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'sym':
                out = self.file['sym'][shape_ind, :cur_num_part, :]
                if not self.pose_sequence == 0:
                    out = out[cur_pose_sequence]
                out = torch.from_numpy(out).float()    # p x 3
                data_feats = data_feats + (out,)
                
            elif feat == 'semantic_ids':
                cur_part_ids = self.file['part_ids'][shape_ind, :cur_num_part]
                if not self.pose_sequence == 0:
                    cur_part_ids = cur_part_ids[cur_pose_sequence]
                num_parts = len(cur_part_ids)
                out = np.zeros((num_parts), dtype=np.float32)
                out[:num_parts] = cur_part_ids                    
                out = torch.from_numpy(out).float().unsqueeze(0)    # 1 x 20 
                data_feats = data_feats + (out,)
                                
            elif feat == 'part_pcs':
                cur_pts = self.file['part_pcs'][shape_ind, :cur_num_part, :, :]                      # p x N x 3 (p is unknown number of parts for this shape)
                if not self.pose_sequence == 0:
                    cur_pts = cur_pts[cur_pose_sequence, :, :]
                out = torch.from_numpy(cur_pts).float()    # 1 x 20 x N x 3
                data_feats = data_feats + (out,)

            elif feat == 'part_poses':
                cur_pose = self.file['part_poses'][shape_ind, :cur_num_part, :]                   # p x (3 + 4)
                if not self.pose_sequence == 0:
                    cur_pose = cur_pose[cur_pose_sequence, :]
                out = torch.from_numpy(cur_pose).float()    # 1 x 20 x (3 + 4)
                data_feats = data_feats + (out,)

            elif feat == 'part_valids':
                # notice this in collate
                out = np.ones((cur_num_part), dtype=np.float32)
                out = torch.from_numpy(out).float()    # 1 x 20 (return 1 for the first p parts, 0 for the rest)
                data_feats = data_feats + (out,)
            
            elif feat == 'shape_id':
                out = np.ones((1), dtype=np.int32)
                out[0] = shape_id
                out = torch.from_numpy(out).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'part_ids':
                cur_part_ids = np.array(self.file['geo_part_ids'][shape_ind, :cur_num_part], dtype=np.int)
                if not self.pose_sequence == 0:
                    cur_part_ids = cur_part_ids[cur_pose_sequence]
                cur_num_part = cur_num_part
                out = np.zeros((cur_num_part), dtype=np.float32)
                out[:cur_num_part] = cur_part_ids                 
                out = torch.from_numpy(out).float()    # 1 x 20 
                data_feats = data_feats + (out,)

            elif feat == 'match_ids':
                cur_part_ids = np.array(self.file['geo_part_ids'][shape_ind, :cur_num_part], dtype=np.int)
                if not self.pose_sequence == 0:
                    cur_part_ids = cur_part_ids[cur_pose_sequence]
                out = np.zeros((cur_num_part), dtype=np.float32)
                out[:cur_num_part] = cur_part_ids
                index = 1
                for i in range(1,58):
                    idx = np.where(out==i)[0]
                    idx = torch.from_numpy(idx)
                    # print(idx)
                    if len(idx)==0: continue
                    elif len(idx)==1: out[idx]=0
                    else:
                        out[idx] = index
                        index += 1
                # ipdb.set_trace()
                out = torch.from_numpy(out).float()
                data_feats = data_feats + (out,)
            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats


if __name__ == '__main__':
    from argparse import ArgumentParser

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
    parser.add_argument('--num_batch_every_visu', type=int, default=1, help='num batch every visu')
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
        default="tcp://localhost:9998",
        type=str,
    )

    # new added
    parser.add_argument('--auto_resume', action='store_true', default=False, help='auto resume')
    parser.add_argument('--output_dir', default=None, help='auto resume')
    parser.add_argument('--pose_sequence', default=1, type=int, help='use our pose sequence or not, default is True')

    # transformer
    parser.add_argument('--num_attention_heads', default=8, type=int, help='auto resume')
    parser.add_argument('--encoder_hidden_dim', default=16, type=int, help='auto resume')
    parser.add_argument('--encoder_dropout', default=0.1, type=float, help='auto resume')
    parser.add_argument('--encoder_activation', default='relu', type=str, help='auto resume')
    parser.add_argument('--encoder_num_layers', default=8, type=int, help='auto resume')
    parser.add_argument('--object_dropout', default=0.0, type=float, help='auto resume')
    parser.add_argument('--theta_loss_divide', default=None, help='auto resume')


    # parse args
    conf = parser.parse_args()
    from torch.utils.data.distributed import DistributedSampler
    def create_sampler(dataset, shuffle, cfg):
        """
        Create sampler for the given dataset.
        Args:
            dataset (torch.utils.data.Dataset): the given dataset.
            shuffle (bool): set to ``True`` to have the data reshuffled
                at every epoch.
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
        Returns:
            sampler (Sampler): the created sampler.
        """
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=cfg.seed) if cfg.num_gpus > 1 else None

        return sampler
    data_features = ['match_id']
    train_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.train_data_fn, data_features, \
            max_num_part=conf.max_num_part, level=conf.level, pose_sequence=conf.pose_sequence)
    val_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.val_data_fn, data_features, \
            max_num_part=conf.max_num_part, level=conf.level, pose_sequence=conf.pose_sequence)
    test_dataset = PartNetPartDataset(conf.category, conf.data_dir, "Table.test.npy", data_features, \
        max_num_part=conf.max_num_part, level=conf.level, pose_sequence=conf.pose_sequence)
    count = 0
    cnt = 0
    for i, x in enumerate(train_dataset):
        if x == None:
            count += 1
        else:
            cnt += 1
    print(count, cnt)
    count = 0
    cnt = 0
    for i, x in enumerate(val_dataset):
        if x == None:
            count += 1
        else:
            cnt += 1
    print(count, cnt)
    count = 0
    cnt = 0
    for i, x in enumerate(test_dataset):
        if x == None:
            count += 1
        else:
            cnt += 1
    print(count, cnt)
    