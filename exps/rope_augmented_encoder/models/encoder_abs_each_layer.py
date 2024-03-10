import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_each_layer_pos import TransformerEncoder, TransformerEncoderLayer
from scipy.optimize import linear_sum_assignment
import random
from quaternion import qrot
from cd.chamfer import chamfer_distance
from .utils import MLP, PointNet, DropoutSampler, PosePredictor, index_points
# from pointnet2_ops import pointnet2_utils
from .rope import *

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask



def load_ply(xyz, num_points=10000, y_up=False):
    bs = xyz.shape[0]
    if y_up:
        # swap y and z axis
        xyz[:, [1, 2]] = xyz[:, [2, 1]]
    rgb = torch.ones(bs, 3, num_points).cuda(torch.cuda.current_device()) * 0.4
    features = torch.cat([xyz, rgb], dim=1)
    return xyz, features


class Network(torch.nn.Module):

    def __init__(self, conf):
    # , num_attention_heads=8, encoder_hidden_dim=16, encoder_dropout=0.1, 
    #             encoder_activation="relu", encoder_num_layers=8, object_dropout=0.0, 
    #             theta_loss_divide=None):
        super(Network, self).__init__()
        self.conf = conf
        self.num_attention_heads = conf.num_attention_heads
        encoder_hidden_dim = conf.encoder_hidden_dim
        encoder_dropout = conf.encoder_dropout
        encoder_activation = conf.encoder_activation
        encoder_num_layers = conf.encoder_num_layers
        object_dropout = conf.object_dropout
        theta_loss_divide = conf.theta_loss_divide
        print("Encoder with PointNet")

        # object encode will have dim 256
        # self.object_encoder = PointTransformerEncoderSmall(output_dim=256, input_dim=6, mean_center=True)
        self.object_encoder = PointNet(feat_len=conf.feat_len)

        # 256 = 240 (point cloud) + 8 (position idx) + 8 (token type)
        self.mlp = MLP(256, 216, uses_pt=False)
        self.point_cloud_downscale = torch.nn.Linear(216, 128)
        # self.position_embeddings = torch.nn.Embedding(20, 8)
        if conf.rope == 1:
            half_head_dim = conf.feat_len // self.num_attention_heads
            hw_seq_len = 20
            self.rope = RotaryEmbedding(
                head_dim=half_head_dim,
                use_cache=False,
                base=10000
            )
            print('------------------------using rope!-------------------------')
        else:
            self.rope = None
            print('------------------------not using rope!-------------------------')
        encoder_layer = TransformerEncoderLayer(d_model=256, nhead=self.num_attention_heads, dim_feedforward=encoder_hidden_dim, dropout=encoder_dropout, activation=encoder_activation, rope=self.rope, multi_cat=conf.multi_cat)
        self.transformer = TransformerEncoder(encoder_layer=encoder_layer, num_layers=encoder_num_layers)
        self.use_inst_encoded = conf.use_inst_encoded
        self.obj_inst = 20 if self.use_inst_encoded in [1, 2] else 0
        if conf.MoN == 0:
            self.use_mon = False
            self.obj_dist = PosePredictor(256 + self.obj_inst, 7, dropout_rate=object_dropout)
        else:
            self.use_mon = True
            print(f'------------use mon with {self.obj_inst} -----------------')
            self.obj_dist = PosePredictor(256+conf.random_noise_size + self.obj_inst, 7, dropout_rate=object_dropout)
        self.multi_cat = conf.multi_cat
        if conf.use_label:
            print('-----------------use label', conf.use_label, '------------------')
            self.word_embedding = nn.Embedding(2, 256)

    def encode_pc(self, part_pcs, batch_size, num_parts):
        x = self.object_encoder(part_pcs.view(batch_size * num_parts, -1, 3))

        obj_pc_embed = self.mlp(x)

        return obj_pc_embed.reshape(batch_size, num_parts, -1)

    def print_gpu_mem(self, info=None):
        """打印GPU显存占用"""
        if int(torch.cuda.current_device()) == 0:
            print(f"\n GPU Mem:{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB, Current device:{torch.cuda.current_device()}, {info}") 

    def forward(self, part_pcs, part_valids, src_mask, position_index, group_ids, random_noise=None, instance_label=None, label=None):
        batch_size = part_pcs.shape[0]
        num_parts = part_pcs.shape[1]
        ######### input of encoder ################
        parts_pc_feat = self.encode_pc(part_pcs, batch_size, num_parts)
        ######### positional encoding ############
        # position_embed = self.position_embeddings(position_index)
        individual_position = F.one_hot(position_index, num_classes=20)
        group_position = F.one_hot(torch.tensor(group_ids, dtype=torch.int64), num_classes=20)
        position_embed = group_position + individual_position
        # src_position_embed = position_embed[:, :num_parts, :]
        ######### source inputs and target inputs ############
        src_sequence = torch.cat([parts_pc_feat, group_position, position_embed], dim=-1)
        src_sequence_encode = src_sequence.transpose(0, 1)
        if label is not None:
            src_sequence = torch.cat([self.word_embedding(label).reshape(batch_size, 1, 256), src_sequence], dim=1)
            valids_ones = torch.ones(batch_size).cuda(torch.cuda.current_device())
            part_valids = torch.cat([valids_ones.unsqueeze(1), part_valids], dim=1)
        src_sequence_encode = src_sequence.transpose(0, 1)
        ######### masks ##############
        pad_mask = (part_valids == 0)

        src_pad_mask = pad_mask.clone()
        if src_mask is not None:
            src_mask = torch.repeat_interleave(src_mask, self.num_attention_heads, dim=0)
        # encode: [sequence_length, batch_size, embedding_size]
        encode = self.transformer(src=src_sequence_encode,
                                mask=None,
                                each_layer_instance=2,
                                src_key_padding_mask=src_pad_mask,
                                return_intermedidate=False)
        if self.use_inst_encoded == 2 and self.training:
            obj_encodes = [e.transpose(1, 0)[:, -num_parts:, :] for e in encode]
            pred_poses_list = [self.obj_dist(torch.cat([obj_encode, random_noise, instance_label], dim=-1)) for obj_encode in obj_encodes]
            return pred_poses_list
        encode = encode.transpose(1, 0)
        ########## post-processing ###############
        obj_encodes = encode[:, -num_parts:, :]
        if self.use_mon:
            if self.use_inst_encoded:
                try:
                    obj_encodes = torch.cat([obj_encodes, random_noise, instance_label], dim=-1)
                except:
                    raise ValueError(f"random_noise: {random_noise} incorrect, please check the input of random noise in MoN")

            else:
                try:
                    obj_encodes = torch.cat([obj_encodes, random_noise], dim=-1)
                except:
                    raise ValueError(f"random_noise: {random_noise} incorrect, please check the input of random noise in MoN")
        try:
            pred_poses = self.obj_dist(obj_encodes)
        except:
            raise ValueError(f"obj_encode.shape: {obj_encodes.shape}]")
        return pred_poses


    def criterion(self, predictions, labels):

        loss = 0
        for key in predictions:

            preds = predictions[key]
            gts = labels[key]

            mask = gts == -100
            preds = preds[~mask]
            gts = gts[~mask]

            loss += self.loss(preds, gts)

        return loss

    def convert_logits(self, predictions):

        for key in predictions:
            if key == "rearrange_obj_labels":
                predictions[key] = torch.sigmoid(predictions[key])

        return predictions

    def get_total_cd_loss(self, pts, quat1, quat2, valids, center1, center2, device):
        batch_size = pts.shape[0]
        num_part =  pts.shape[1]
        num_point = pts.shape[2]
        center1 = center1.unsqueeze(2).repeat(1,1,num_point,1)
        center2 = center2.unsqueeze(2).repeat(1,1,num_point,1)
        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

        dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        loss_per_data = loss_per_data.view(batch_size, -1)
        
        thre = 0.01
        loss_per_data = loss_per_data.cuda(torch.cuda.current_device())
        acc = [[0 for i in range(num_part)]for j in range(batch_size)]
        for i in range(batch_size):
            for j in range(num_part):
                if loss_per_data[i,j] < thre and valids[i,j]:
                    acc[i][j] = 1
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data , acc
        
    def get_possible_point_list(self, point, sym):
        sym = torch.tensor([1.0,1.0,1.0]) 
        point_list = []
        #sym = torch.tensor(sym)
        if sym.equal(torch.tensor([0.0, 0.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
        elif sym.equal(torch.tensor([1.0, 0.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
        elif sym.equal(torch.tensor([0.0, 1.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
        elif sym.equal(torch.tensor([0.0, 0.0, 1.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
        elif sym.equal(torch.tensor([1.0, 1.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
            point_list.append(self.get_sym_point(point, 1, 1, 0))
        elif sym.equal(torch.tensor([1.0, 0.0, 1.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
            point_list.append(self.get_sym_point(point, 1, 0, 1))
        elif sym.equal(torch.tensor([0.0, 1.0, 1.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
            point_list.append(self.get_sym_point(point, 0, 1, 1))
        else:
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
            point_list.append(self.get_sym_point(point, 1, 1, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 1))
            point_list.append(self.get_sym_point(point, 0, 1, 1))
            point_list.append(self.get_sym_point(point, 1, 1, 1))

        return point_list

    def get_sym_point(self, point, x, y, z):

        if x:
            point[0] = - point[0]
        if y:
            point[1] = - point[1]
        if z:
            point[2] = - point[2]

        return point.tolist()

    def get_contact_point_loss_score_pa(self, center, quat, contact_points, sym_info):

        batch_size = center.shape[0]
        num_part = center.shape[1]
        contact_point_loss = torch.zeros(batch_size)
        total_num = 0
        batch_total_num = torch.zeros(batch_size, dtype=torch.long)
        count = 0
        batch_count = torch.zeros(batch_size, dtype=torch.long)
        for b in range(batch_size):
            #print("Shape id is", b)
            sum_loss = 0
            for i in range(num_part):
                for j in range(num_part):
                    if contact_points[b, i, j, 0]:
                        contact_point_1 = contact_points[b, i, j, 1:]
                        contact_point_2 = contact_points[b, j, i, 1:]
                        sym1 = sym_info[b, i]
                        sym2 = sym_info[b, j]
                        point_list_1 = self.get_possible_point_list(contact_point_1, sym1)
                        point_list_2 = self.get_possible_point_list(contact_point_2, sym2)
                        dist = self.get_min_l2_dist(point_list_1, point_list_2, center[b, i, :], center[b, j, :], quat[b, i, :], quat[b, j, :])  # 1
                        #print(dist)
                        if dist < 0.01:
                            count += 1
                            batch_count[b] += 1
                        total_num += 1
                        batch_total_num[b] += 1
                        sum_loss += dist
            contact_point_loss[b] = sum_loss


        #print(count, total_num)
        return contact_point_loss, count, total_num, batch_count, batch_total_num
        
    def get_contact_point_loss(self, center, quat, contact_points, sym_info):

        batch_size = center.shape[0]
        num_part = center.shape[1]
        contact_point_loss = torch.zeros(batch_size)
        total_num = 0
        count = 0
        for b in range(batch_size):
            #print("Shape id is", b)
            sum_loss = 0
            for i in range(num_part):
                for j in range(num_part):
                    if contact_points[b, i, j, 0]:
                        contact_point_1 = contact_points[b, i, j, 1:]
                        contact_point_2 = contact_points[b, j, i, 1:]
                        sym1 = sym_info[b, i]
                        sym2 = sym_info[b, j]
                        point_list_1 = self.get_possible_point_list(contact_point_1, sym1)
                        point_list_2 = self.get_possible_point_list(contact_point_2, sym2)
                        dist = self.get_min_l2_dist(point_list_1, point_list_2, center[b, i, :], center[b, j, :], quat[b, i, :], quat[b, j, :])  # 1
                        #print(dist)
                        if dist < 0.01:
                            count += 1
                        total_num += 1
                        sum_loss += dist
            contact_point_loss[b] = sum_loss


        #print(count, total_num)
        return contact_point_loss, count, total_num
        
    def get_min_l2_dist(self, list1, list2, center1, center2, quat1, quat2):

        list1 = torch.tensor(list1) # m x 3
        list2 = torch.tensor(list2) # n x 3
        #print(list1[0])
        #print(list2[0])
        len1 = list1.shape[0]
        len2 = list2.shape[0]
        center1 = center1.unsqueeze(0).repeat(len1, 1)
        center2 = center2.unsqueeze(0).repeat(len2, 1)
        quat1 = quat1.unsqueeze(0).repeat(len1, 1)
        quat2 = quat2.unsqueeze(0).repeat(len2, 1)
        list1 = list1.cuda(torch.cuda.current_device())
        list2 = list2.cuda(torch.cuda.current_device())
        list1 = center1 + qrot(quat1, list1)
        list2 = center2 + qrot(quat2, list2)
        mat1 = list1.unsqueeze(1).repeat(1, len2, 1)
        mat2 = list2.unsqueeze(0).repeat(len1, 1, 1)
        mat = (mat1 - mat2) * (mat1 - mat2)
        #ipdb.set_trace()
        mat = mat.sum(dim=-1)
        return mat.min()

    @staticmethod
    def linear_assignment_ddp(pts, centers1, quats1, centers2, quats2):
        '''
           registration: aligning parts point cloud one-by-one by hungarian matching https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        '''
        pts_to_select = torch.tensor(random.sample([i for i  in range(1000)],100))
        pts = pts[:,pts_to_select] 
        cur_part_cnt = pts.shape[0]
        num_point = pts.shape[1]

        with torch.no_grad():

            cur_quats1 = quats1.unsqueeze(1).repeat(1, num_point, 1)
            cur_centers1 = centers1.unsqueeze(1).repeat(1, num_point, 1)
            cur_pts1 = qrot(cur_quats1, pts) + cur_centers1

            cur_quats2 = quats2.unsqueeze(1).repeat(1, num_point, 1)
            cur_centers2 = centers2.unsqueeze(1).repeat(1, num_point, 1)
            cur_pts2 = qrot(cur_quats2, pts) + cur_centers2

            cur_pts1 = cur_pts1.unsqueeze(1).repeat(1, cur_part_cnt, 1, 1).view(-1, num_point, 3)
            cur_pts2 = cur_pts2.unsqueeze(0).repeat(cur_part_cnt, 1, 1, 1).view(-1, num_point, 3)
            dist1, dist2 = chamfer_distance(cur_pts1, cur_pts2, transpose=False)
            dist_mat = (dist1.mean(1) + dist2.mean(1)).view(cur_part_cnt, cur_part_cnt)
            rind, cind = linear_sum_assignment(dist_mat.cpu().numpy())

        return rind, cind


    """
        Input: B x P x 3, B x P x 3, B x P
        Output: B
    """
    @staticmethod
    def get_trans_l2_loss_ddp(trans1, trans2, valids):
        loss_per_data = (trans1 - trans2).pow(2).sum(dim=-1)

        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data

    """
        Input: B x P x N x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """
    @staticmethod
    def get_rot_l2_loss_ddp(pts, quat1, quat2, valids):
        batch_size = pts.shape[0]
        num_point = pts.shape[2]

        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

        loss_per_data = (pts1 - pts2).pow(2).sum(-1).mean(-1)

        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data

    """
        Input: B x P x N x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """
    @staticmethod
    def get_rot_cd_loss_ddp(pts, quat1, quat2, valids, device):
        batch_size = pts.shape[0]
        num_point = pts.shape[2]

        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

        dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        loss_per_data = loss_per_data.view(batch_size, -1)

        loss_per_data = loss_per_data.cuda(torch.cuda.current_device())
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data  

    @staticmethod
    def get_total_cd_loss_ddp(pts, quat1, quat2, valids, center1, center2, device):
        batch_size = pts.shape[0]
        num_part =  pts.shape[1]
        num_point = pts.shape[2]
        center1 = center1.unsqueeze(2).repeat(1,1,num_point,1)
        center2 = center2.unsqueeze(2).repeat(1,1,num_point,1)
        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

        dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        loss_per_data = loss_per_data.view(batch_size, -1)
        
        thre = 0.01
        loss_per_data = loss_per_data.cuda(torch.cuda.current_device())
        acc = [[0 for i in range(num_part)]for j in range(batch_size)]
        for i in range(batch_size):
            for j in range(num_part):
                if loss_per_data[i,j] < thre and valids[i,j]:
                    acc[i][j] = 1
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data , acc

    @staticmethod
    def get_shape_cd_loss_ddp(pts, quat1, quat2, valids, center1, center2, device):
        batch_size = pts.shape[0]
        num_part = pts.shape[1]
        num_point = pts.shape[2]
        center1 = center1.unsqueeze(2).repeat(1,1,num_point,1)
        center2 = center2.unsqueeze(2).repeat(1,1,num_point,1)
        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

        pts1 = pts1.view(batch_size,num_part*num_point,3)
        pts2 = pts2.view(batch_size,num_part*num_point,3)
        dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
        valids = valids.unsqueeze(2).repeat(1,1,1000).view(batch_size,-1)
        dist1 = dist1 * valids
        dist2 = dist2 * valids
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        
        loss_per_data = loss_per_data.cuda(torch.cuda.current_device())
        return loss_per_data

        """
            output : B
        """
    # @staticmethod
    # def cal_weights(pts, quat1, quat2, valids, center1, center2, category, pct, beta=2):
    #     '''
    #         used in training
    #     '''
    #     batch_size = pts.shape[0]
    #     num_part = pts.shape[1]
    #     num_point = pts.shape[2]
    #     pts1 = qrot(quat1.unsqueeze(2).repeat(1,1,num_point,1), pts) + center1.unsqueeze(2).repeat(1,1,num_point,1)
    #     pts1 = pts1.view(batch_size, -1, 3)
        
    #     fps_idx1 = pointnet2_utils.furthest_point_sample(pts1, 1000).long() # [B, npoint]
    #     new_pts1 = index_points(pts1, fps_idx1).transpose(1, 2)

    #     scores1 = F.softmax(pct(new_pts1), dim=1)[..., category]
    #     return torch.mean((-scores1.add(-1)).pow(beta).reshape(batch_size, -1), dim=1)

    # @staticmethod
    # def cal_scores(pts, quat1, quat2, valids, center1, center2, category, pct, oracle=0):
    #     '''
    #         used in testing
    #     '''
    #     batch_size = pts.shape[0]
    #     num_part = pts.shape[1]
    #     num_point = pts.shape[2]
    #     if oracle == 1:
    #         pts2 = qrot(quat2.unsqueeze(2).repeat(1,1,num_point,1), pts) + center2.unsqueeze(2).repeat(1,1,num_point,1)
    #         pts2 = pts2.view(batch_size, -1, 3)
    #         fps_idx2 = pointnet2_utils.furthest_point_sample(pts2, 1000).long() # [B, npoint]
    #         new_pts2 = index_points(pts2, fps_idx2).transpose(1, 2)

    #         scores2 = F.softmax(pct(new_pts2), dim=1)[..., category]

    #         return torch.mean((scores2).reshape(batch_size, -1), dim=1)
    #     elif oracle == 0:
    #         pts1 = qrot(quat1.unsqueeze(2).repeat(1,1,num_point,1), pts) + center1.unsqueeze(2).repeat(1,1,num_point,1)
    #         pts1 = pts1.view(batch_size, -1, 3)
    #         fps_idx1 = pointnet2_utils.furthest_point_sample(pts1, 1000).long() # [B, npoint]
    #         new_pts1 = index_points(pts1, fps_idx1).transpose(1, 2)
    #         scores1 = F.softmax(pct(new_pts1), dim=1)[..., category]
    #         # print(torch.sum((scores2 - scores1) > 0.1), torch.sum((scores1 - scores2)>0.1))
    #         return torch.mean((scores1).reshape(batch_size, -1), dim=1)
    #     else:
    #         raise TypeError(f"{oracle} not implemented yet.")
    # @staticmethod
    # def cal_weights_openshape(pts, quat1, quat2, valids, center1, center2, pretrained_objs, category, beta=1, temperature=100):
    #     pretrained_3D_encoder = pretrained_objs['pretrained_3D_encoder']
    #     text_embedding = pretrained_objs['text_embedding']
    #     batch_size = pts.shape[0]
    #     num_part = pts.shape[1]
    #     num_point = pts.shape[2]
    #     pts1 = qrot(quat1.unsqueeze(2).repeat(1,1,num_point,1), pts) + center1.unsqueeze(2).repeat(1,1,num_point,1)
    #     pts1 = pts1.view(batch_size, -1, 3)
        
    #     fps_idx1 = pointnet2_utils.furthest_point_sample(pts1, 10000).long() # [B, npoint]
    #     new_pts1 = index_points(pts1, fps_idx1).transpose(1, 2)

    #     xyz, feat = load_ply(new_pts1)

    #     shape_feat = pretrained_3D_encoder(xyz.transpose(1, 2), feat.transpose(1, 2), device='cuda') 
        
    #     similarity = temperature * F.normalize(shape_feat, dim=1) @ F.normalize(text_embedding, dim=1).T
    #     shapescore = F.softmax(similarity, dim=1)[..., category]
    #     return torch.mean((-shapescore.add(-1)).pow(beta).reshape(batch_size, -1), dim=1)


    # @staticmethod
    # def cal_scores_openshape(pts, quat1, quat2, valids, center1, center2, category=None, pretrained_objs=None, oracle=0, temperature=100):
    #     if oracle == 0:
    #         pretrained_3D_encoder = pretrained_objs['pretrained_3D_encoder']
    #         text_embedding = pretrained_objs['text_embedding']
    #         batch_size = pts.shape[0]
    #         num_part = pts.shape[1]
    #         num_point = pts.shape[2]
    #         pts1 = qrot(quat1.unsqueeze(2).repeat(1,1,num_point,1), pts) + center1.unsqueeze(2).repeat(1,1,num_point,1)
    #         pts1 = pts1.view(batch_size, -1, 3)
            
    #         fps_idx1 = pointnet2_utils.furthest_point_sample(pts1, 10000).long() # [B, npoint]
    #         new_pts1 = index_points(pts1, fps_idx1).transpose(1, 2)

    #         xyz, feat = load_ply(new_pts1)

    #         shape_feat = pretrained_3D_encoder(xyz.transpose(1, 2), feat.transpose(1, 2), device='cuda') 
    #         similarity = F.softmax(temperature*F.normalize(shape_feat, dim=1) @ F.normalize(text_embedding, dim=1).T, dim=1)[..., category]
    #         # similarity = 100*(F.normalize(shape_feat, dim=1) @ F.normalize(text_embedding, dim=1).T)[..., category]
    #         return torch.mean(similarity.reshape(batch_size, -1), dim=1)
    #         # return similarity
    #     elif oracle == 1:
    #         pretrained_3D_encoder = pretrained_objs['pretrained_3D_encoder']
    #         text_embedding = pretrained_objs['text_embedding']
    #         batch_size = pts.shape[0]
    #         num_part = pts.shape[1]
    #         num_point = pts.shape[2]
    #         pts2 = qrot(quat2.unsqueeze(2).repeat(1,1,num_point,1), pts) + center2.unsqueeze(2).repeat(1,1,num_point,1)
    #         pts2 = pts2.view(batch_size, -1, 3)
            
    #         fps_idx2 = pointnet2_utils.furthest_point_sample(pts2, 10000).long() # [B, npoint]
    #         new_pts2 = index_points(pts2, fps_idx2).transpose(1, 2)
    #         xyz, feat = load_ply(new_pts2)
    #         shape_feat = pretrained_3D_encoder(xyz.transpose(1, 2), feat.transpose(1, 2), device='cuda') 
    #         similarity = F.softmax(temperature*F.normalize(shape_feat, dim=1) @ F.normalize(text_embedding, dim=1).T, dim=1)[..., category]
    #         # similarity = 100*(F.normalize(shape_feat, dim=1) @ F.normalize(text_embedding, dim=1).T)[..., category]
    #         return torch.mean(similarity.reshape(batch_size, -1), dim=1)
    #         # return similarity
    #     else:
    #         raise TypeError(f'oracle {oracle} not found')

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
    parser.add_argument('--encoder_hidden_dim', default=1024, type=int, help='auto resume')
    parser.add_argument('--encoder_dropout', default=0.1, type=float, help='auto resume')
    parser.add_argument('--encoder_activation', default='relu', type=str, help='auto resume')
    parser.add_argument('--encoder_num_layers', default=6, type=int, help='auto resume')
    parser.add_argument('--object_dropout', default=0.0, type=float, help='auto resume')
    parser.add_argument('--theta_loss_divide', default=None, help='auto resume')


    # parse args
    conf = parser.parse_args()
    model = Network(conf)
    params = model.state_dict() 

    # 计算所有参数元素大小
    param_size = 0
    for param in params.values():
        param_size += param.nelement() * param.element_size()
        
    # 转换为MB    
    param_size_mb = param_size / (1024*1024)

    print("Model Size: {:.2f} MB".format(param_size_mb))    # part_pcs = torch.randn(16, 20, 1000, 3)
    # part_valids = torch.ones(16, 1, 20)
    # gt_pose_inputs = torch.randn(16, 20, 7)
    # position_index = torch.tensor(list(range(20)), dtype=torch.long).unsqueeze(0).repeat(16, 1)
    # tgt_mask = generate_square_subsequent_mask(part_valids.shape[-1])
    # start_token = torch.zeros((part_valids.shape[0], 1), dtype=torch.long)
    # model(part_pcs, part_valids, gt_pose_inputs, start_token, tgt_mask, position_index)
    




