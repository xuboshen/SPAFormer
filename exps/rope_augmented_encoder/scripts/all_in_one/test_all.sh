cd .. 
cd ..
source activate PartAssembly
categories=(Bag Bed Bottle Bowl Chair Clock Dishwasher Display Door Earphone Faucet Hat Knife Lamp Laptop Mug StorageFurniture Refrigerator TrashCan Vase Table)
levels=('1' '3' '3' '1' '3' '3' '3' '3' '3' '3' '3' '1' '3' '3' '1' '1' '3' '3' '3' '3' '3')
epochs=(900 900 900)
for i in "${!categories[@]}"
do
    category=${categories[i]}
    level=${levels[i]}

    CUDA_VISIBLE_DEVICES=5 python ./test_paper.py  \
        --exp_suffix '_train_all_cat' \
        --model_version 'encoder_abs_each_layer' \
        --category $category \
        --train_data_fn "${category}_filtered.train.npy" \
        --val_data_fn "${category}_filtered.test.npy" \
        --device cuda:0 \
        --model_dir "./checkpoints/ours_multitask_ckpt.pth"\
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