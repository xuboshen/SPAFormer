import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).cuda(torch.cuda.current_device()).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx.cpu()

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint 
    xyz = xyz.contiguous()

    fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, pt_dim=3, uses_pt=True):
        super(MLP, self).__init__()
        self.uses_pt = uses_pt
        self.output = out_dim
        d5 = int(in_dim)
        d6 = int(2 * self.output)
        d7 = self.output
        self.encode_position = nn.Sequential(
                nn.Linear(pt_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.ReLU(),
                )
        d5 = 2 * in_dim if self.uses_pt else in_dim
        self.fc_block = nn.Sequential(
            nn.Linear(int(d5), d6),
            nn.LayerNorm(int(d6)),
            nn.ReLU(),
            nn.Linear(int(d6), d6),
            nn.LayerNorm(int(d6)),
            nn.ReLU(),
            nn.Linear(d6, d7))

    def forward(self, x, pt=None):
        if self.uses_pt:
            if pt is None: raise RuntimeError('did not provide pt')
            y = self.encode_position(pt)
            x = torch.cat([x, y], dim=-1)
        return self.fc_block(x)


class PointNet(nn.Module):
    def __init__(self, feat_len):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.mlp1 = nn.Linear(1024, feat_len)
        self.bn6 = nn.BatchNorm1d(feat_len)

    """
        Input: B x N x 3 (B x P x N x 3)
        Output: B x F (B x P x F)
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = x.max(dim=-1)[0]

        x = F.relu(self.bn6(self.mlp1(x)))
        return x

class DropoutSampler(torch.nn.Module):
    def __init__(self, num_features, num_outputs, dropout_rate = 0.5):
        super(DropoutSampler, self).__init__()
        self.linear = nn.Linear(num_features, num_features)
        self.linear2 = nn.Linear(num_features, num_features)
        self.predict = nn.Linear(num_features, num_outputs)
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = F.relu(self.linear(x))
        if self.dropout_rate > 0:
            x = F.dropout(x, self.dropout_rate)
        x = F.relu(self.linear2(x))
        # x = F.dropout(x, self.dropout_rate)
        return self.predict(x)


class PosePredictor(torch.nn.Module):
    def __init__(self, num_features, num_outputs, dropout_rate = 0.5):
        super(PosePredictor, self).__init__()
        self.linear = nn.Linear(num_features, num_features)
        self.linear2 = nn.Linear(num_features, num_features)
        self.trans = nn.Linear(num_features, 3)
        self.quat = nn.Linear(num_features, 4)
        self.quat.bias.data.zero_()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = F.relu(self.linear(x))
        if self.dropout_rate > 0:
            x = F.dropout(x, self.dropout_rate)
        x = F.relu(self.linear2(x))
        trans = torch.tanh(self.trans(x))
        quat_bias = x.new_tensor([[[1.0, 0.0, 0.0, 0.0]]])
        quat = self.quat(x).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=-1, keepdim=True)).sqrt()
        out = torch.cat([trans, quat], dim=-1)
        return out


def generate_square_subsequent_mask(sz, part_id=None):
    # mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
    # 创建sz x sz的全零矩阵 

    # # 将mask矩阵放到右下角
    # fused_mask[sz:, sz:] = mask

    # # 创建一个sz x sz的-inf矩阵
    # infs = torch.ones(sz, sz) * float('-inf')

    # # 放到右上角
    # fused_mask[:sz, sz:] = infs

    # 放到左上方，grouping
    if part_id is not None:
        matmul_ids = torch.matmul(part_id.unsqueeze(2), part_id.unsqueeze(1))
        square_ids = (part_id * part_id).unsqueeze(2)
        mask = (matmul_ids == square_ids)
        mask = (mask).float().masked_fill((mask)==0, float('-inf')).masked_fill((mask)==1, float(0.0))
    return mask