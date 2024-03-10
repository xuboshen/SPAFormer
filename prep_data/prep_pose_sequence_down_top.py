import os
import numpy as np
from queue import PriorityQueue
from tqdm import tqdm
import copy
import torch

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
def process_file(file_path, pc_path):
    # Load the data from the .npy file
    cur_data = np.load(pc_path, allow_pickle=True).item()
    cur_pts = cur_data['part_pcs']
    class_index = cur_data['part_ids']
    num_point = cur_pts.shape[1]
    poses = cur_data['part_poses']
    quat = poses[:, 3:]
    center = poses[:, :3]
    gt_pts = copy.copy(cur_pts)
    for i in range(len(cur_pts)):
        gt_pts[i] = qrot(torch.from_numpy(quat[i]).unsqueeze(0).repeat(num_point,1).unsqueeze(0), torch.from_numpy(cur_pts[i]).unsqueeze(0))
        gt_pts[i] = gt_pts[i] + center[i]


    # originals
    num_parts = cur_pts.shape[0]

    # Create a priority queue to store parts based on z-coordinate
    priority_queue = PriorityQueue()
    # gt_pts = gt_pts.mean(1).reshape(-1, 3) # Parts, 1, 3
    for i in range(num_parts):
        z_coord = gt_pts[i][:, 1].mean()
        priority_queue.put((z_coord, i))  # Use negative z_coord for descending order

    visited = set()
    parts_sequence = []

    while not priority_queue.empty():
        _, current_part = priority_queue.get()
        if current_part not in visited:
            visited.add(current_part)
            parts_sequence.append(current_part)
    return parts_sequence

input_folder = "./shape_data"
pcc_path = "./shape_data"
output_folder = "./pose_sequence_ascending_downtop"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# import pdb; pdb.set_trace()
for filename in tqdm(os.listdir(input_folder)):
    if not filename.startswith("pairs"):
        try:
            file_path = os.path.join(input_folder, filename)
            pc_path = os.path.join(pcc_path, filename)
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_folder, output_filename)
            if os.path.exists(output_path):
                continue
            parts_sequence = process_file(file_path, pc_path)

            with open(output_path, "w") as f:
                f.write(" ".join(map(str, parts_sequence)))
        except:
            print('filename error: ', filename)
    else:
        continue