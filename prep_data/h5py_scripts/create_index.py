import os
import numpy as np
import pickle
filefoldername = '../shape_data'
count = 0
shape_ids = []
shape_id_2_index = {}
for i, file in enumerate(os.listdir(filefoldername)):
    if file.startswith('pairs'):
        continue
    try:
        shape_ids.append(int(file.split('_')[0]))
        shape_id_2_index[int(file.split('_')[0])] = count
        count += 1
    except:
        print(file)
# starts from 0, 0~n-1
print(len(shape_ids))
np.save('./index_2_shape_ids', np.array(shape_ids, dtype=np.int32))

with open('shape_ids_2_index.pkl', 'wb') as f:
    pickle.dump(shape_id_2_index, f)
with open('shape_ids_2_index.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
data = np.load('./index_2_shape_ids.npy')

