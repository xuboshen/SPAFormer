
import os
import numpy as np
import pickle
import h5py
filefoldername = '../shape_data'
# cat_names = ['../Table_filtered.train.npy', '../Chair_filtered.train.npy', '../Lamp_filtered.train.npy',
#             '../Table_filtered.val.npy', '../Chair_filtered.val.npy', '../Lamp_filtered.val.npy',
#             '../Table_filtered.test.npy', '../Chair_filtered.test.npy', '../Lamp_filtered.test.npy',]
cat_names = ['Bag', 'Bed', 'Bottle', 'Bowl',
            'Chair', 'Clock', 'Dishwasher', 'Display', 
            'Door', 'Earphone', 'Faucet', 'Hat', 
            'Keyboard', 'Knife', 'Lamp', 'Laptop', 
            'Microwave', 'Mug', 'StorageFurniture', 'Refrigerator',
            'Scissors', 'TrashCan', 'Vase', 'Table']
level  =    [1, 3, 3, 1, 
            3, 3, 3, 3, 
            3, 3, 3, 1,
            1, 3, 3, 1, 
            3, 1, 3, 3,
            1, 3, 3, 3
]

data = np.load('./index_2_shape_ids.npy')
N = len(data)
with open('shape_ids_2_index.pkl', 'rb') as f:
    shape_ids_2_index = pickle.load(f)

with h5py.File("./shapes_all.hdf5", "w") as f:
    dset_part_pcs = f.create_dataset('part_pcs', (N, 20, 1000, 3), dtype='float64')
    dset_part_poses = f.create_dataset('part_poses', (N, 20, 7), dtype='float64')
    dset_part_ids = f.create_dataset('part_ids', (N, 20), dtype='i')
    dset_geo_part_ids = f.create_dataset('geo_part_ids', (N, 20), dtype='i')
    dset_sym = f.create_dataset('sym', (N, 20, 3), dtype='float64')
    dset_length = f.create_dataset('length', (N, 1), dtype='i')
    dset_label = f.create_dataset('label', (N, 1), dtype='i')
    dset_contact = f.create_dataset('contact_matrix', (N, 20, 20, 4), dtype='f')

    train_list = []
    from tqdm import tqdm
    train_below_20 = []
    for split in ['train', 'val', 'test']:
        for label, cat in enumerate(cat_names):
    # for ind, cat_name in enumerate(cat_names):
            cat_name = f'../{cat}_filtered.{split}.npy'
            try:
                file = np.load(cat_name)
            except:
                continue
            for index in tqdm(range(len(file))):
                shape_id = file[index]
                cur_data_fn = os.path.join('../prep_data', 'shape_data/%s_level' % shape_id + str(level[label]) + '.npy')
                cur_data = np.load(cur_data_fn, allow_pickle=True).item()   # assume data is stored in seperate .npz file
                cur_sym = cur_data['sym']
                cur_num_part = cur_sym.shape[0]
                if cur_num_part > 20:
                    continue         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item
                else:
                    dset_part_pcs[shape_ids_2_index[int(shape_id)], :cur_num_part, :, :] = cur_data['part_pcs']
                    dset_part_poses[shape_ids_2_index[int(shape_id)], :cur_num_part, :] = cur_data['part_poses']
                    dset_part_ids[shape_ids_2_index[int(shape_id)], :cur_num_part] = cur_data['part_ids']
                    dset_geo_part_ids[shape_ids_2_index[int(shape_id)], :cur_num_part] = cur_data['geo_part_ids']
                    dset_sym[shape_ids_2_index[int(shape_id)], :cur_num_part, :] = cur_data['sym']
                    dset_length[shape_ids_2_index[int(shape_id)], 0] = len(cur_data['sym'])
                    dset_label[shape_ids_2_index[int(shape_id)], 0] = label
                    # import pdb; pdb.set_trace()
                    cur_contact_fn = os.path.join('../prep_data', 'contact_points/pairs_with_contact_points_%s_level' % shape_id + str(level[label]) + '.npy')
                    cur_contact = np.load(cur_contact_fn)   # assume data is stored in seperate .npz file
                    dset_contact[shape_ids_2_index[int(shape_id)], :cur_num_part, :cur_num_part, :] = cur_contact[:, :, :]
