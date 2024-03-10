import os
import json
import copy
import numpy as np
import ipdb
import torch
import math
import random

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def cal_distance(a,b):  #pts1: N x 3, pts2: N x 3
    num_points = a.shape[0]
    a = torch.tensor(a)
    b = torch.tensor(b)
    A = a.unsqueeze(0).repeat(num_points,1,1)
    B = b.unsqueeze(1).repeat(1,num_points,1)
    C = (A - B)**2
    C = np.array(C).sum(axis=2)
    ind = C.argmin()
    # row
    R_ind = ind//1000
    # column
    C_ind = ind - R_ind*1000
    return C.min(), C_ind

# adjacency matric of parts
def get_pair_list(pts): #pts: p x N x 3
    delta1 = 1e-3
    cnt1 = 0
    num_part = pts.shape[0]
    connect_list = np.zeros((num_part,num_part,4))

    for i in range(0, num_part):
        for j in range(0, num_part):
            if i == j: continue
            dist, point_ind = cal_distance(pts[i], pts[j])
            # point in i
            point = pts[i, point_ind]

            if dist < delta1:
                connect_list[i][j][0] = 1
                connect_list[i][j][1] = point[0]
                connect_list[i][j][2] = point[1]
                connect_list[i][j][3] = point[2]

            else:
                connect_list[i][j][0] = 0
                connect_list[i][j][1] = point[0]
                connect_list[i][j][2] = point[1]
                connect_list[i][j][3] = point[2]
    return connect_list

def find_pts_ind(part_pts,point):
    for i in range(len(part_pts)):
        if part_pts[i,0] == point[0] and part_pts[i,1] == point[1] and part_pts[i,2] == point[2]:
            return i
    return -1


    


if __name__ == "__main__":
    import os
    import sys
    sys.path.append("./")
    cat_name = sys.argv[1]
    root = sys.argv[2]
    root_shape = "../prep_data/shape_data/"
    modes = ['val','train','test']
    levels = [sys.argv[2]]
    for level in levels:
        for mode in modes:
            object_json =json.load(open(root + "stats/train_val_test_split/" + cat_name +"." + mode + ".json"))
            object_list = [int(object_json[i]['anno_id']) for i in range(len(object_json))]
            idx = 0
            for id in object_list:
                idx += 1
                print("level", level, " ", mode, " ", id,"      ",idx,"/",len(object_list))
                #if os.path.isfile(root + "contact_points/" + 'pairs_with_contact_points_%s_level' % id + str(level) + '.npy'):
                if True:
                    cur_data_fn = os.path.join(root_shape, '%s_level' % id + str(level) + '.npy')
                    cur_data = np.load(cur_data_fn, allow_pickle=True).item()  # assume data is stored in seperate .npz filenp.load()
                    cur_pts = cur_data['part_pcs']  # p x N x 3 (p is unknown number of parts for this shape)
                    class_index = cur_data['part_ids']
                    num_point = cur_pts.shape[1]
                    poses = cur_data['part_poses']
                    quat = poses[:,3:]
                    center = poses[:,:3]
                    gt_pts = copy.copy(cur_pts)
                    for i in range(len(cur_pts)):
                        gt_pts[i] = qrot(torch.from_numpy(quat[i]).unsqueeze(0).repeat(num_point,1).unsqueeze(0), torch.from_numpy(cur_pts[i]).unsqueeze(0))
                        gt_pts[i] = gt_pts[i] + center[i]

                    oldfile  = get_pair_list(gt_pts)
                    newfile = oldfile
                    # change the value of point from ground truth at object level to normalized point at part level
                    for i in range(len(oldfile)):
                        for j in range(len(oldfile[0])):
                            if i == j: continue
                            point = oldfile[i,j,1:]
                            ind = find_pts_ind(gt_pts[i], point)
                            if ind == -1:
                                ipdb.set_trace()
                            else:
                                newfile[i,j,1:] = cur_pts[i,ind]
                    np.save("../prep_data/" + "contact_points/" + 'pairs_with_contact_points_%s_level' % id + str(level) + '.npy', newfile)
