cd .. 
cd ..
source activate PartAssembly
categories=(all)
levels=('3' '3' '3')
epochs=(900 900 900)
for i in "${!categories[@]}"
do
    category=${categories[i]}
    level=${levels[i]}
    CUDA_VISIBLE_DEVICES="4,5" python ./train.py  \
        --exp_suffix '_train_all_cat' \
        --model_version 'encoder_abs_each_layer' \
        --category "$category" \
        --train_data_fn "${category}_filtered_21.train.npy" \
        --val_data_fn "${category}_filtered_21.val.npy" \
        --loss_weight_trans_l2 1.0 \
        --loss_weight_rot_l2 0.0 \
        --loss_weight_rot_cd 10 \
        --loss_weight_shape_cd 1.0 \
        --device cuda:0 \
        --num_epoch_every_visu 1000 \
        --level "$level" \
        --epochs 900 \
        --lr 1.5e-4 \
        --batch_size 64 \
        --num_workers 8 \
        --num_batch_every_visu 0 \
        --num_gpus 2 \
        --lr_decay_every 90 \
        --checkpoint_interval 20 \
        --lr_decay_by 0.8 \
        --encoder_num_layers 6 \
        --encoder_hidden_dim 1024 \
        --log_dir ./ckpts/all_cat_in_one \
        --pose_sequence 4 \
        --rope 1 \
        --MoN 0 \
        --repeat_times 1 \
        --random_noise_size 64 \
        --init_method tcp://localhost:9994 \
        --checkpoint_interval 20 \


    CUDA_VISIBLE_DEVICES="4,5" python ./train.py  \
        --exp_suffix '_train_all_cat' \
        --model_version 'encoder_abs_each_layer' \
        --category "$category" \
        --train_data_fn "${category}_filtered_21.train.npy" \
        --val_data_fn "${category}_filtered_21.val.npy" \
        --loss_weight_trans_l2 1.0 \
        --loss_weight_rot_l2 0.0 \
        --loss_weight_rot_cd 10 \
        --loss_weight_shape_cd 1.0 \
        --device cuda:0 \
        --num_epoch_every_visu 1000 \
        --level "$level" \
        --epochs 900 \
        --lr 1.5e-4 \
        --batch_size 128 \
        --num_workers 8 \
        --num_batch_every_visu 0 \
        --num_gpus 2 \
        --lr_decay_every 90 \
        --checkpoint_interval 20 \
        --lr_decay_by 0.8 \
        --encoder_num_layers 6 \
        --encoder_hidden_dim 1024 \
        --log_dir ./ckpts/all_cat_in_one \
        --pose_sequence 4 \
        --rope 1 \
        --MoN 0 \
        --repeat_times 1 \
        --random_noise_size 64 \
        --init_method tcp://localhost:9994 \
        --checkpoint_interval 20 \


    CUDA_VISIBLE_DEVICES=5 python ./test_paper.py  \
        --exp_suffix '_train_all_cat' \
        --model_version 'encoder_abs_each_layer' \
        --category $category \
        --train_data_fn "${category}_filtered_21.train.npy" \
        --val_data_fn "${category}_filtered_21.test.npy" \
        --device cuda:0 \
        --model_dir "./ckpts/all_cat_in_one/exp-${category}-encoder_abs_each_layer-level${level}_train_all_cat/ckpts/400_net_network.pth"\
        --level "$level" \
        --batch_size 16 \
        --num_batch_every_visu 0 \
        --pose_sequence 4 \
        --rope 1 \
        --MoN 0 \
        --repeat_times 1 \
        --log_dir ./ckpts/all_cat_in_one \
        --encoder_num_layers 6 \
        --encoder_hidden_dim 1024

    CUDA_VISIBLE_DEVICES=5 python ./test_paper.py  \
        --exp_suffix '_train_all_cat' \
        --model_version 'encoder_abs_each_layer' \
        --category $category \
        --train_data_fn "${category}_filtered_21.train.npy" \
        --val_data_fn "${category}_filtered_21.test.npy" \
        --device cuda:0 \
        --model_dir "./ckpts/all_cat_in_one/exp-${category}-encoder_abs_each_layer-level${level}_train_all_cat/ckpts/800_net_network.pth"\
        --level "$level" \
        --batch_size 16 \
        --num_batch_every_visu 0 \
        --pose_sequence 4 \
        --rope 1 \
        --MoN 0 \
        --repeat_times 1 \
        --log_dir ./ckpts/all_cat_in_one \
        --encoder_num_layers 6 \
        --encoder_hidden_dim 1024


    CUDA_VISIBLE_DEVICES=5 python ./test_paper.py  \
        --exp_suffix '_train_all_cat' \
        --model_version 'encoder_abs_each_layer' \
        --category $category \
        --train_data_fn "${category}_filtered_21.train.npy" \
        --val_data_fn "${category}_filtered_21.test.npy" \
        --device cuda:0 \
        --model_dir "./ckpts/all_cat_in_one/exp-${category}-encoder_abs_each_layer-level${level}_train_all_cat/best/net_network.pth"\
        --level "$level" \
        --batch_size 16 \
        --num_batch_every_visu 0 \
        --pose_sequence 4 \
        --rope 1 \
        --MoN 0 \
        --repeat_times 1 \
        --log_dir ./ckpts/all_cat_in_one \
        --encoder_num_layers 6 \
        --encoder_hidden_dim 1024

done