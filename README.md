# SPAFormer: Sequential 3D Part Assembly with Transformers

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

dataset preprocessing
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
We provide our "versatile model" here for clarity. You can navigate other single-category models following the same format freely.
- train script: [train_seq.sh](exps/rope_augmented_encoder/scripts/all_in_one/train_seq.sh)

We provide our training and testing log under [train log](exps/rope_augmented_encoder/logs/train_log.txt) and [test log](exps/rope_augmented_encoder/logs/test_log.txt)

### Validating
- eval_script across all categories: [test_all.sh](exps/rope_augmented_encoder/scripts/all_in_one/test_all.sh)

## Acknowledgement

We are grateful for the following projects
- DGL: The codebase we built upon and benchmarking our baselines.
- ScorePA: We fix some evaluation errors caused by DGL, i.e. the total size of the dataset.
- PartNet: the dataset where we built PartNet-Assembly.