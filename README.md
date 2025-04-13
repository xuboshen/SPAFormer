# SPAFormer: Sequential 3D Part Assembly with Transformers
Official code of [SPAFormer: Sequential 3D Part Assembly with Transformers](https://arxiv.org/abs/2403.05874) (3DV 2025).

## Overview

We introduce SPAFormer, an innovative model designed to overcome the combinatorial explosion challenge in the 3D Part Assembly (3D-PA) task.
This task requires accurate prediction of each part's pose and shape in sequential steps, and as the number of parts increases, the possible assembly combinations increase exponentially, leading to a combinatorial explosion that severely hinders the efficacy of 3D-PA.
SPAFormer addresses this problem by leveraging weak constraints from assembly sequences, effectively reducing the solution space's complexity. 

Our contributions can be summarized as:
- **Innovative Framework:**
    We propose SPAFormer, a transformer-based model for object assembly with sequential 3D parts. Our model particularly leverages sequential part information, and incorporates knowledge enhancement strategies to significantly improve the assembly performance.
- **Generalization of SPAFormer:** 
    SPAFormer shows superior generalization in object assembly from three crucial perspectives: a) category-specific, enabling it to handle various objects in the same category; b) multi-task, showcasing its versatility across diverse object categories using the same model;  and c) long-horizon, proving its ability in managing complex assembly tasks with numerous parts.
- **A More Comprehensive Benchmark:**
    To facilitate a thorough evaluation of different models, we introduce an extensive benchmark named PartNet-Assembly, covering up to 21 object categories and providing a broad spectrum of object assembly tasks for the first time.

## Install
We use CUDA=11.7 and python=3.7.

- clone this repository locally:
```
git clone $(this_repository)
cd ./$(this_repository)
```

- create your environment and install packages
```
conda create -n spaformer python=3.7
conda activate spaformer
pip install -r requirements.txt
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
cd exp/utils/cd
pip install .
```

## Train and Validate

### Data preparation
We release the 'diagonal' version of PartNet-Assembly data in [this link](https://pan.baidu.com/s/1jzbnmQLz4XkNR0nX7APl9Q?pwd=skb9&_at_=1739098127967#list/path=%2F).

If you'd like to create the data on your own, you could follow:
- Firstly, download the dataset from [PartNet](https://docs.google.com/forms/d/e/1FAIpQLSetsP7aj-Hy0gvP2FxRT3aTIrc_IMqSqR-5Xl8P3x2awDkQbw/viewform). We suggest putting PartNet under ./prep_data.

- Then, process the data by the scripts:
```
cd ./prep_data
bash prep_shape.sh
bash prep_contact_point.sh
bash prep_feasible_data.sh
bash prep_index.sh
cd ./h5py_scripts
bash create_dataset.sh
```

- To generate various assembly sequences (e.g. diagonal sequence):
```
cd ./prep_data
python prep_pose_sequence_diagonal.py
```

### Training
We provide our "multitask model" here for clarity. You can navigate other single-category models following the same format freely.
- train script: [train_seq.sh](exps/rope_augmented_encoder/scripts/all_in_one/train_seq.sh)

We provide our training and testing log under [train log](exps/rope_augmented_encoder/logs/train_log.txt) and [test log](exps/rope_augmented_encoder/logs/test_log.txt)

### Pretrained Model
We release the checkpoint of our "multitask model" in [checkpoints](checkpoints/ours_multitask_ckpt.pth) folder.

### Validating
- eval_script across all categories: [test_all.sh](exps/rope_augmented_encoder/scripts/all_in_one/test_all.sh)

## Acknowledgement

We are grateful for the following projects:
- [DGL](https://github.com/hyperplane-lab/Generative-3D-Part-Assembly): The codebase we built upon and benchmarking our baselines.
- [ScorePA](https://github.com/J-F-Cheng/Score-PA_Score-based-3D-Part-Assembly): We fix some evaluation errors caused by DGL, i.e. the total size of the dataset.
- [PartNet](https://partnet.cs.stanford.edu/): the dataset where we built PartNet-Assembly.

## Citation
If you find our code and research helpful, please cite our paper:
```
@inproceedings{
      xu2025spaformer,
      title={{SPAF}ormer: Sequential 3D Part Assembly with Transformers},
      author={Boshen Xu and Sipeng Zheng and Qin Jin},
      booktitle={International Conference on 3D Vision 2025},
      year={2025},
      url={https://openreview.net/forum?id=kryphH8cJP}
}
```
