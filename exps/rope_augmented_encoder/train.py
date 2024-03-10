"""
    Training models
"""

import os
import time
import sys
import shutil
import random
from torch.nn.utils import clip_grad_norm_
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
logger = logging.get_logger(__name__)
torch.autograd.set_detect_anomaly(True)
os.environ["IPC_LOCK"] = '1'
# modelnet
# category = {"Chair": 8, "Table": 33, "Lamp": 19}
# category = {"Chair": 4, "Table": 23, "Lamp": 14}
# category = {"Chair": 0, "Table": 1, "Lamp": 2}
candidate_part = []
def train(conf):    
    # Set up environment.
    du.init_distributed_training(conf)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    # Set random seed from configs.
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(conf.seed)
        torch.cuda.manual_seed_all(conf.seed)
    random.seed(conf.seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # save config
    torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))
    # file log
    flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'a+')
    conf.flog = flog
    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')
    # backup python files used for this training
    os.system('cp data_loader.py models/%s.py models/utils.py %s %s' % (conf.model_version, __file__, conf.exp_dir))
    # # set training device
    # device = torch.device(conf.device)
    # utils.printout(flog, f'Using device: {conf.device}\n')
    # conf.device = device

    # create training and validation datasets and data loaders
    data_features = ['part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'match_ids', 'contact_points', 'sym']
    
    train_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.train_data_fn, data_features, \
            max_num_part=conf.max_num_part, level=conf.level, pose_sequence=conf.pose_sequence, rand_seq=conf.rand_seq, pr=conf.pr)
    utils.printout(conf.flog, str(train_dataset))
    train_sampler = dataset_utils.create_sampler(train_dataset, shuffle=True, cfg=conf)
    if conf.num_workers > 0:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, pin_memory=False, sampler=train_sampler, \
                num_workers=conf.num_workers, drop_last=True, collate_fn=utils.collate_variant_input, worker_init_fn=utils.worker_init_fn, persistent_workers=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, pin_memory=True, sampler=train_sampler, \
                num_workers=conf.num_workers, drop_last=False, collate_fn=utils.collate_variant_input, worker_init_fn=utils.worker_init_fn, persistent_workers=False)
    
    val_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.val_data_fn, data_features, \
            max_num_part=conf.max_num_part,level=conf.level, pose_sequence=conf.pose_sequence)
    utils.printout(conf.flog, str(val_dataset))
    val_sampler = dataset_utils.create_sampler(val_dataset, shuffle=False, cfg=conf)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, pin_memory=True, sampler=val_sampler, \
            num_workers=0, drop_last=False, collate_fn=utils.collate_variant_input, worker_init_fn=utils.worker_init_fn)

    # load network model
    model_def = utils.get_model_module(conf.model_version)

    # create models
    network = model_def.Network(conf)
    network = model_utils.build_model(conf, network)

    logger.info(conf.flog, '\n' + str(network) + '\n')
    # utils.printout(conf.flog, '\n' + str(network) + '\n')

    models = [network]
    model_names = ['network']
    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    optimizers = [network_opt]
    optimizer_names = ['network_opt']

    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)
    # network_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(network_opt, T_max=conf.lr_restart)


    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR    TransL2Loss    RotL2Loss   RotCDLoss  ShapeCDLoss   TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        from tensorboardX import SummaryWriter
        train_writer = SummaryWriter(os.path.join(conf.exp_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(conf.exp_dir, 'val'))

    # # # send parameters to device
    # for o in optimizers:
    #     utils.optimizer_to_device(o, torch.cuda.current_device())
    # start training
    start_time = time.time()
    last_checkpoint_step = None
    last_train_console_log_step, last_val_console_log_step = None, None
    train_num_batch = len(train_dataloader)
    val_num_batch = len(val_dataloader)
    start_epoch = utils.load_checkpoint(models, model_names, dirname=os.path.join(conf.exp_dir, 'ckpts'), epoch=23, optimizers=optimizers, optimizer_names=model_names)
    if conf.lr not in [1e-4, 1e-5, 5e-5, 1.5e-4]:
        for param_group in optimizers[0].param_groups:
            param_group["lr"] = conf.lr
    # train for every epoch
    min_total_loss_val = None
    for epoch in range(start_epoch, conf.epochs):
        # shuffle dataset
        if conf.num_gpus > 1:
            train_dataloader.sampler.set_epoch(epoch)
        if not conf.no_console_log:
            logger.info(conf.flog, f'training run {conf.exp_name}')
            logger.info(conf.flog, header)
            # utils.printout(conf.flog, f'training run {conf.exp_name}')
            # utils.printout(conf.flog, header)

        train_batches = enumerate(train_dataloader, 0)

        val_batches = enumerate(val_dataloader, 0)
        train_fraction_done = 0.0
        val_fraction_done = 0.0
        val_batch_ind = -1
        val_flag = 0 # to record whether it is the first time


        # train for every batch
        # allowcated_memory_info("Before training")
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind
            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # set models to training mode
            for m in models:
                m.train()

            # forward pass (including logging)
            if len(batch)==0:continue
            # not use pre-trained models
            if conf.contact_loss:
                total_loss, total_trans_l2_loss, total_rot_cd_loss, total_shape_cd_loss, total_contact_loss = forward(batch=batch, data_features=data_features, network=network, conf=conf, is_val=False, \
                    step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time, \
                    log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer, lr=network_opt.param_groups[0]['lr'])
            else:
                total_loss, total_trans_l2_loss, total_rot_cd_loss, total_shape_cd_loss = forward(batch=batch, data_features=data_features, network=network, conf=conf, is_val=False, \
                    step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time, \
                    log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer, lr=network_opt.param_groups[0]['lr'])

            #to sum the training loss of all categories
            if train_batch_ind == 0:
                sum_total_loss = total_loss.item()
                sum_total_trans_l2_loss = total_trans_l2_loss.item()
                sum_total_rot_cd_loss = total_rot_cd_loss.item()
                sum_total_shape_cd_loss = total_shape_cd_loss.item()
                if conf.contact_loss:
                    sum_total_contact_loss = total_contact_loss.item()
            else:
                sum_total_loss += total_loss.item()
                sum_total_trans_l2_loss += total_trans_l2_loss.item()
                sum_total_rot_cd_loss += total_rot_cd_loss.item()
                sum_total_shape_cd_loss += total_shape_cd_loss.item()
                if conf.contact_loss:
                    sum_total_contact_loss = total_contact_loss.item()
            #optimize one step
            network_opt.zero_grad()
            total_loss.backward()
            
            # gradient clip:
            total_norm = clip_grad_norm_(models[0].parameters(), 1.0)
            
            network_opt.step()
            ## detect memory
            # allowcated_memory_info("after updating models in training")

            if epoch % conf.checkpoint_interval == 0:
                # validate one batch
                while val_fraction_done <= train_fraction_done and val_batch_ind+1 < val_num_batch:
                    val_batch_ind, val_batch = next(val_batches)

                    val_fraction_done = (val_batch_ind + 1) / val_num_batch
                    val_step = (epoch + val_fraction_done) * train_num_batch - 1

                    log_console = not conf.no_console_log and (last_val_console_log_step is None or \
                            val_step - last_val_console_log_step >= conf.console_log_interval)
                    if log_console:
                        last_val_console_log_step = val_step

                    # set models to evaluation mode
                    for m in models:
                        m.eval()

                    with torch.no_grad():
                        # forward pass (including logging)
                        if len(val_batch)==0:continue
                        if conf.contact_loss:
                            total_loss, total_trans_l2_loss, total_rot_cd_loss, total_shape_cd_loss, total_contact_loss = forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
                                step=val_step, epoch=epoch, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
                                log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=val_writer, lr=network_opt.param_groups[0]['lr'])
                        else:
                            total_loss, total_trans_l2_loss, total_rot_cd_loss, total_shape_cd_loss = forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
                                step=val_step, epoch=epoch, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
                                log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=val_writer, lr=network_opt.param_groups[0]['lr'])
                        # to sum the validating loss of all categories
                        if val_flag == 0:
                            val_total_trans_l2_loss = total_trans_l2_loss.item()
                            val_total_rot_cd_loss = total_rot_cd_loss.item()
                            val_total_shape_cd_loss = total_shape_cd_loss.item()
                            if conf.contact_loss:
                                val_total_contact_loss = total_contact_loss.item()
                            val_flag = 1
                        else:
                            val_total_trans_l2_loss += total_trans_l2_loss.item()
                            val_total_shape_cd_loss += total_shape_cd_loss.item()
                            val_total_rot_cd_loss += total_rot_cd_loss.item()
                            if conf.contact_loss:
                                val_total_contact_loss += total_contact_loss.item()
                        if min_total_loss_val is None:
                            min_total_loss_val = total_loss.item()
                        else:
                            if total_loss.item() < min_total_loss_val:
                                min_total_loss_val = total_loss.item()
                                logger.info(conf.flog, 'Saving best checkpoint ...... ')
                                # utils.printout(conf.flog, 'Saving final checkpoint ...... ')
                                utils.save_checkpoint(models=models, model_names=model_names, dirname=os.path.join(conf.exp_dir, 'best'), \
                                        epoch=epoch, prepend_epoch=False, optimizers=optimizers, optimizer_names=optimizer_names, cfg=conf)
                                logger.info(conf.flog, "DONE")
        # save checkpoint
        with torch.no_grad():
            if (last_checkpoint_step is None or epoch - last_checkpoint_step >= conf.checkpoint_interval) and int(torch.cuda.current_device()) == 0:
                logger.info(conf.flog, 'Saving checkpoint ...... ')
                # utils.printout(conf.flog, 'Saving checkpoint ...... ')
                utils.save_checkpoint(models=models, model_names=model_names, dirname=os.path.join(conf.exp_dir, 'ckpts'), \
                        epoch=epoch, prepend_epoch=True, optimizers=optimizers, optimizer_names=model_names, cfg=conf)
                utils.printout(conf.flog, 'DONE')
                last_checkpoint_step = epoch
        #using tensorboard to record the losses for each epoch
        with torch.no_grad():
            if not conf.no_tb_log and train_writer is not None and du.is_master_proc():
                train_writer.add_scalar('sum_total_loss', sum_total_loss, epoch)
                train_writer.add_scalar('sum_total_trans_l2_loss', sum_total_trans_l2_loss, epoch)
                train_writer.add_scalar('sum_total_rot_cd_loss', sum_total_rot_cd_loss, epoch)
                train_writer.add_scalar('sum_total_shape_cd_loss', sum_total_shape_cd_loss, epoch)
                if conf.contact_loss:
                    train_writer.add_scalar('sum_contact_loss_loss', sum_total_contact_loss, epoch)
                
                train_writer.add_scalar('lr', network_opt.param_groups[0]['lr'], epoch)
            if not conf.no_tb_log and val_writer is not None and du.is_master_proc():
                try:
                    val_writer.add_scalar('val_total_trans_l2_loss', val_total_trans_l2_loss, epoch)
                    val_writer.add_scalar('val_total_rot_cd_loss', val_total_rot_cd_loss, epoch)
                    val_writer.add_scalar('val_total_shape_cd_loss', val_total_shape_cd_loss, epoch)
                    val_writer.add_scalar('lr', network_opt.param_groups[0]['lr'], epoch)
                    if conf.contact_loss:
                        val_writer.add_scalar('val_total_contact_loss', val_total_contact_loss, epoch)
                        
                except:
                    pass
        network_lr_scheduler.step()

        # allowcated_memory_info("After training")
        # torch.cuda.empty_cache()
    if int(torch.cuda.current_device()) == 0:
        # save the final models
        logger.info(conf.flog, 'Saving final checkpoint ...... ')
        # utils.printout(conf.flog, 'Saving final checkpoint ...... ')
        utils.save_checkpoint(models=models, model_names=model_names, dirname=os.path.join(conf.exp_dir, 'ckpts'), \
                epoch=epoch, prepend_epoch=True, optimizers=optimizers, optimizer_names=optimizer_names, cfg=conf)
        logger.info(conf.flog, "DONE")
    # utils.printout(conf.flog, 'DONE')
    # close file log
    flog.close()


def forward(batch, data_features, network, conf, \
        is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
        log_console=False, log_tb=False, tb_writer=None, lr=None, pretrained_model=None):

    # prepare input
    input_part_pcs = batch[data_features.index('part_pcs')].cuda(non_blocking=True, device=torch.cuda.current_device())           # B x P x N x 3
    input_part_valids = batch[data_features.index('part_valids')].cuda(non_blocking=True, device=torch.cuda.current_device())     # B x P
    batch_size = input_part_pcs.shape[0]
    num_part = input_part_pcs.shape[1]
    num_point = input_part_pcs.shape[2]
    part_ids = batch[data_features.index('part_ids')].cuda(non_blocking=True, device=torch.cuda.current_device())      # B x P 
    # labels = batch[data_features.index('label')].cuda(non_blocking=True, device=torch.cuda.current_device())      # B x P 
    match_ids = batch[data_features.index('match_ids')]
    gt_part_poses = batch[data_features.index('part_poses')].cuda(non_blocking=True, device=torch.cuda.current_device())      # B x P x (3 + 4)
    position_index = torch.tensor(range(0, num_part), dtype=torch.long).repeat(input_part_pcs.shape[0], 1).cuda(non_blocking=True, device=torch.cuda.current_device())
    repeat_times = conf.repeat_times
    if conf.contact_loss:
        contact_points = batch[data_features.index('contact_points')].cuda(non_blocking=True, device=torch.cuda.current_device())      # B x P x (3 + 4)
        sym_info = batch[data_features.index("sym")]  # B x P x 3

    # get instance label
    # if conf.use_inst_encoded in [1, 2]:
    instance_label = torch.zeros(batch_size, num_part, 20).cuda(non_blocking=True, device=torch.cuda.current_device())
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
    num_parts = input_part_pcs.shape[1]
    with torch.cuda.amp.autocast(enabled=True):
        if not conf.MoN == 0: # use MoN
            network.eval()
            random_noise_list = []
            with torch.no_grad():
                for repeat_ind in range(repeat_times):
                    random_noise = np.random.normal(loc=0.0, scale=1, size=[batch_size, num_parts, conf.random_noise_size]).astype(
                        np.float32)  # B x P x 16
                    random_noise_list.append(random_noise)
                    random_noise = torch.tensor(random_noise).cuda(torch.cuda.current_device())  # B x P x 16

                    if conf.use_label:
                        pred_part_poses = network(input_part_pcs, input_part_valids, None, position_index, part_ids, random_noise, instance_label, labels)
                    else:
                        pred_part_poses = network(input_part_pcs, input_part_valids, None, position_index, part_ids, random_noise, instance_label)
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
                        rot_cd_loss_per_data = network.module.get_rot_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids, conf.device)      # B
                        shape_cd_loss_per_data = network.module.get_shape_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:],
                                                    input_part_valids, pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], conf.device)
                        if conf.contact_loss:
                            contact_point_loss_per_data, _, _ = network.module.get_contact_point_loss_train(pred_part_poses[:, :, :3],
                                                                        pred_part_poses[:, :, 3:], contact_points, sym_info, input_part_valids)

                    else:
                        trans_l2_loss_per_data = network.get_trans_l2_loss_ddp(pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], input_part_valids)  # B
                        rot_cd_loss_per_data = network.get_rot_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids, conf.device)      # B
                        shape_cd_loss_per_data = network.get_shape_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:],
                                                    input_part_valids, pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], conf.device)
                        if conf.contact_loss:
                            contact_point_loss_per_data, _, _ = network.get_contact_point_loss_train(pred_part_poses[:, :, :3],
                                                                        pred_part_poses[:, :, 3:], contact_points, sym_info, input_part_valids)

                    # for each type of loss, compute avg loss per batch
                    shape_cd_loss = shape_cd_loss_per_data.mean()
                    trans_l2_loss = trans_l2_loss_per_data.mean()
                    rot_cd_loss = rot_cd_loss_per_data.mean()
                    # compute total loss

                    total_loss = trans_l2_loss * conf.loss_weight_trans_l2 + \
                                    rot_cd_loss * conf.loss_weight_rot_cd + \
                                    shape_cd_loss * conf.loss_weight_shape_cd
                    if conf.contact_loss:
                        contact_point_loss = contact_point_loss_per_data.mean()
                        total_loss += contact_point_loss * conf.loss_weight_contact

                    if repeat_ind == 0:
                        mini_ind = 0
                        res_loss = total_loss
                    else:
                        if total_loss.item() < res_loss.item():
                            mini_ind = repeat_ind
                        res_loss = res_loss.min(total_loss)
                    # torch.cuda.empty_cache()
            network.train()
            random_noise = random_noise_list[mini_ind]
            random_noise = torch.tensor(random_noise).cuda(torch.cuda.current_device())  # B x P x 16
            if conf.use_inst_encoded in [1]:
                pred_part_poses = network(input_part_pcs, input_part_valids, None, position_index, part_ids, random_noise, instance_label)
            elif conf.use_inst_encoded in [2]:
                pred_part_poses_list = network(input_part_pcs, input_part_valids, None, position_index, part_ids, random_noise, instance_label)
                pred_part_poses = torch.cat(pred_part_poses_list, dim=0)
                input_part_pcs = input_part_pcs.repeat(6, 1, 1, 1)
                gt_part_poses = gt_part_poses.repeat(6, 1, 1)
                input_part_valids = input_part_valids.repeat(6, 1)
            else:
                if conf.use_label:
                    pred_part_poses = network(input_part_pcs, input_part_valids, None, position_index, part_ids, random_noise, instance_label=instance_label, label=labels)
                else:
                    pred_part_poses = network(input_part_pcs, input_part_valids, None, position_index, part_ids, random_noise, instance_label=instance_label)            # matching loss
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
                rot_cd_loss_per_data = network.module.get_rot_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids, conf.device)      # B
                shape_cd_loss_per_data = network.module.get_shape_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:],
                                            input_part_valids, pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], conf.device)
                if conf.contact_loss:
                    contact_point_loss_per_data, _, _ = network.module.get_contact_point_loss_train(pred_part_poses[:, :, :3],
                                                                        pred_part_poses[:, :, 3:], contact_points, sym_info, input_part_valids)

            else:
                trans_l2_loss_per_data = network.get_trans_l2_loss_ddp(pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], input_part_valids)  # B
                rot_cd_loss_per_data = network.get_rot_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids, conf.device)      # B
                shape_cd_loss_per_data = network.get_shape_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:],
                                            input_part_valids, pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], conf.device)
                if conf.contact_loss:
                    contact_point_loss_per_data, _, _ = network.get_contact_point_loss_train(pred_part_poses[:, :, :3],
                                                                        pred_part_poses[:, :, 3:], contact_points, sym_info, input_part_valids)

            # for each type of loss, compute avg loss per batch
            shape_cd_loss = shape_cd_loss_per_data.mean()
            trans_l2_loss = trans_l2_loss_per_data.mean()
            rot_cd_loss = rot_cd_loss_per_data.mean()
            # compute total loss

            total_loss = trans_l2_loss * conf.loss_weight_trans_l2 + \
                            rot_cd_loss * conf.loss_weight_rot_cd + \
                            shape_cd_loss * conf.loss_weight_shape_cd
            total_shape_cd_loss = shape_cd_loss.clone().detach()
            total_trans_l2_loss = trans_l2_loss.clone().detach()
            total_rot_cd_loss = rot_cd_loss.clone().detach()
            if conf.contact_loss:
                contact_loss = contact_point_loss_per_data.mean()
                total_loss += contact_loss * conf.loss_weight_contact
                total_contact_loss = contact_loss.clone().detach()

        elif conf.MoN == 0: # not use MoN
            if conf.use_inst_encoded in [1]:
                pred_part_poses = network(input_part_pcs, input_part_valids, None, position_index, part_ids, instance_label=instance_label)
            elif conf.use_inst_encoded in [2]:
                pred_part_poses_list = network(input_part_pcs, input_part_valids, None, position_index, part_ids, instance_label=instance_label)
                if isinstance(pred_part_poses_list, list):
                    pred_part_poses = torch.cat(pred_part_poses_list, dim=0)
                    input_part_pcs = input_part_pcs.repeat(6, 1, 1, 1)
                    gt_part_poses = gt_part_poses.repeat(6, 1, 1)
                    input_part_valids = input_part_valids.repeat(6, 1)
                else:
                    pred_part_poses = pred_part_poses_list
            else:
                if conf.use_label:
                    pred_part_poses = network(input_part_pcs, input_part_valids, None, position_index, part_ids, instance_label=instance_label, label=labels)
                else:
                    pred_part_poses = network(input_part_pcs, input_part_valids, None, position_index, part_ids, instance_label=instance_label)            # matching loss
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
                rot_cd_loss_per_data = network.module.get_rot_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids, conf.device)      # B
                shape_cd_loss_per_data = network.module.get_shape_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:],
                                            input_part_valids, pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], conf.device)
                if conf.contact_loss:
                    contact_point_loss_per_data, _, _ = network.module.get_contact_point_loss_train(pred_part_poses[:, :, :3],
                                                                        pred_part_poses[:, :, 3:], contact_points, sym_info, input_part_valids)

            else:
                trans_l2_loss_per_data = network.get_trans_l2_loss_ddp(pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], input_part_valids)  # B
                rot_cd_loss_per_data = network.get_rot_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids, conf.device)      # B
                shape_cd_loss_per_data = network.get_shape_cd_loss_ddp(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:],
                                            input_part_valids, pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], conf.device)
                if conf.contact_loss:
                    contact_point_loss_per_data, _, _ = network.get_contact_point_loss_train(pred_part_poses[:, :, :3],
                                                                        pred_part_poses[:, :, 3:], contact_points, sym_info, input_part_valids)

            # for each type of loss, compute avg loss per batch
            shape_cd_loss = shape_cd_loss_per_data.mean()
            trans_l2_loss = trans_l2_loss_per_data.mean()
            rot_cd_loss = rot_cd_loss_per_data.mean()
            # compute total loss

            total_loss = trans_l2_loss * conf.loss_weight_trans_l2 + \
                            rot_cd_loss * conf.loss_weight_rot_cd + \
                            shape_cd_loss * conf.loss_weight_shape_cd
            if conf.contact_loss:
                contact_loss = contact_point_loss_per_data.mean()
                total_loss += contact_loss * conf.loss_weight_contact
                total_contact_loss = contact_loss.clone().detach()
            # print(contact_loss * conf.loss_weight_contact, trans_l2_loss * conf.loss_weight_trans_l2)

            total_shape_cd_loss = shape_cd_loss.clone().detach()
            total_trans_l2_loss = trans_l2_loss.clone().detach()
            total_rot_cd_loss = rot_cd_loss.clone().detach()
        else:
            raise ValueError(f"conf MoN:{conf.MoN} not found")
    data_split = 'train'
    if is_val:
        data_split = 'val'

    with torch.no_grad():
        # log to console
        if log_console and du.is_master_proc():
            utils.printout(conf.flog, \
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{data_split:^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{trans_l2_loss.item():>10.5f}   '''
                f'''{rot_cd_loss.item():>10.5f}  '''
                f'''{shape_cd_loss.item():>10.5f}  '''
                f'''{total_loss.item():>10.5f}  '''
                )
            conf.flog.flush()

        # gen visu
        if is_val and (not conf.no_visu) and epoch % conf.num_epoch_every_visu == conf.num_epoch_every_visu - 1 and du.is_master_proc():
            visu_dir = os.path.join(conf.exp_dir, 'val_visu')
            out_dir = os.path.join(visu_dir, 'epoch-%04d' % epoch)
            input_part_pcs_dir = os.path.join(out_dir, 'input_part_pcs')
            gt_assembly_dir = os.path.join(out_dir, 'gt_assembly')
            pred_assembly_dir = os.path.join(out_dir, 'pred_assembly')
            info_dir = os.path.join(out_dir, 'info')

            
            if batch_ind == 0:
                # create folders
                os.makedirs(out_dir, exist_ok=True)
                os.makedirs(input_part_pcs_dir, exist_ok=True)
                os.makedirs(gt_assembly_dir, exist_ok=True)
                os.makedirs(pred_assembly_dir, exist_ok=True)
                os.makedirs(info_dir, exist_ok=True)

            if batch_ind < conf.num_batch_every_visu:
                utils.printout(conf.flog, 'Visualizing ...')
                pred_center = pred_part_poses[:, :, :3]
                gt_center = gt_part_poses[:, :, :3]

                # compute pred_pts and gt_pts
                
                pred_pts = qrot(pred_part_poses[:, :, 3:].unsqueeze(2).repeat(1, 1, num_point, 1), input_part_pcs) + pred_center.unsqueeze(2).repeat(1, 1, num_point, 1)
                gt_pts = qrot(gt_part_poses[:, :, 3:].unsqueeze(2).repeat(1, 1, num_point, 1), input_part_pcs) + gt_center.unsqueeze(2).repeat(1, 1, num_point, 1)

                for i in range(batch_size):
                    fn = 'data-%03d.png' % (batch_ind * batch_size + i)
                    
                    cur_input_part_cnt = input_part_valids[i].sum().item()
                    cur_input_part_cnt = int(cur_input_part_cnt)
                    cur_input_part_pcs = input_part_pcs[i, :cur_input_part_cnt]
                    cur_gt_part_poses = gt_part_poses[i, :cur_input_part_cnt]
                    cur_pred_part_poses = pred_part_poses[i, :cur_input_part_cnt]

                    pred_part_pcs = qrot(cur_pred_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1), cur_input_part_pcs) + \
                            cur_pred_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)
                    gt_part_pcs = qrot(cur_gt_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1), cur_input_part_pcs) + \
                            cur_gt_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)

                    part_pcs_to_visu = cur_input_part_pcs.cpu().detach().numpy()
                    render_utils.render_part_pts(os.path.join(BASE_DIR, input_part_pcs_dir, fn), part_pcs_to_visu, blender_fn='object_centered.blend')
                    part_pcs_to_visu = pred_part_pcs.cpu().detach().numpy()
                    render_utils.render_part_pts(os.path.join(BASE_DIR, pred_assembly_dir, fn), part_pcs_to_visu, blender_fn='object_centered.blend')
                    part_pcs_to_visu = gt_part_pcs.cpu().detach().numpy()
                    render_utils.render_part_pts(os.path.join(BASE_DIR, gt_assembly_dir, fn), part_pcs_to_visu, blender_fn='object_centered.blend')

                    with open(os.path.join(info_dir, fn.replace('.png', '.txt')), 'w') as fout:
                        fout.write('shape_id: %s\n' % batch[data_features.index('shape_id')][i])
                        fout.write('num_part: %d\n' % cur_input_part_cnt)
                        fout.write('trans_l2_loss: %f\n' % trans_l2_loss_per_data[i].item())
                        fout.write('rot_cd_loss: %f\n' % rot_cd_loss_per_data[i].item())
                
            if batch_ind == conf.num_batch_every_visu - 1:
                # visu html
                utils.printout(conf.flog, 'Generating html visualization ...')
                sublist = 'input_part_pcs,gt_assembly,pred_assembly,info'
                cmd = 'cd %s && python %s . 10 htmls %s %s > /dev/null' % (out_dir, os.path.join(BASE_DIR, '../utils/gen_html_hierarchy_local.py'), sublist, sublist)
                call(cmd, shell=True)
                utils.printout(conf.flog, 'DONE')
    if conf.contact_loss:
        return total_loss, total_trans_l2_loss, total_rot_cd_loss, total_shape_cd_loss, total_contact_loss

    return total_loss, total_trans_l2_loss, total_rot_cd_loss, total_shape_cd_loss



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
    parser.add_argument('--decoding_type', default='teacher_forcing', type=str, help='use our pose sequence or not, default is True')
    parser.add_argument('--schedule_sampling_prob', default=0, type=float, help='use our pose sequence or not, default is True')
    parser.add_argument('--pre_reasoning', default=False, type=bool, help='use our pose sequence or not, default is True')
    parser.add_argument('--mask', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--rope', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--random_noise_size', default=32, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--repeat_times', default=5, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--use_pretrained_models', default='no', type=str, help='use our pose sequence or not, default is True')
    parser.add_argument('--cat_number', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--reweight', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--MoN', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--rand_seq', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--pr', default=0, type=float, help='use our pose sequence or not, default is True')
    parser.add_argument('--beta', default=2, type=float, help='use our pose sequence or not, default is True')
    parser.add_argument('--use_inst_encoded', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--multi_cat', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--use_label', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--temperature', default=100, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--without_one_hot', default=0, type=int, help='use our pose sequence or not, default is True')
    parser.add_argument('--loss_weight_contact', default=20.0, type=float, help='use our pose sequence or not, default is True')
    parser.add_argument('--contact_loss', default=0, type=int, help='use our pose sequence or not, default is True')




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
    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    print("conf", conf)


    ### prepare before training
    # make exp_name
    conf.exp_name = f'exp-{conf.category}-{conf.model_version}-level{conf.level}{conf.exp_suffix}'
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    conf.output_dir = os.path.join(conf.exp_dir, 'ckpts')
    # mkdir exp_dir; ask for overwrite if necessary
    if not os.path.exists(conf.log_dir):
       os.makedirs(conf.log_dir, exist_ok=True) 
    if os.path.exists(conf.exp_dir) and conf.overwrite:
        # if not conf.overwrite:
        #     response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
        #     if response != 'y':
        #         exit(1)
        shutil.rmtree(conf.exp_dir)
    
    os.makedirs(conf.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(conf.exp_dir, 'ckpts'), exist_ok=True)
    if not conf.no_visu:
        os.makedirs(os.path.join(conf.exp_dir, 'val_visu'), exist_ok=True)

    # launch job start training
    ddp_utils.launch_job(cfg=conf, init_method=conf.init_method, func=train)