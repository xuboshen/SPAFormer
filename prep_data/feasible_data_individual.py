import numpy as np
import os
from tqdm import tqdm
import sys
part_names = ['Bag', 'Bed', 'Bottle', 'Bowl',
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
train_list = []
train_below_20 = []
split = str(sys.argv[1])
for ind, part in enumerate(part_names):
    filename = "./index/" + part
    file = np.load(filename + f".{split}.npy")
    for index in tqdm(range(len(file))):
        shape_id = file[index]
        cur_data_fn = os.path.join('./prep_data', 'shape_data/%s_level' % shape_id + str(level[ind]) + '.npy')
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()   # assume data is stored in seperate .npz file
        cur_sym = cur_data['sym']
        cur_num_part = cur_sym.shape[0]
        if cur_num_part > 20 or not cur_data['part_pcs'].shape[0] == len(cur_data['geo_part_ids']):
            continue         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item
        else:
            train_below_20.append(shape_id)
    print(len(train_below_20))
    np.save(f'./{part}_filtered.{split}.npy', np.array(train_below_20, dtype=np.int64))
