# HSTT-demo
Testing codes for 'Structure-aware Hierarchical Spatial-Temporal Transformer for Video Question Answering' with paper ID 2728 under review in CVPR 2022 

## Requirements 
We provide a Docker image for easier reproduction. Please install the following:
  - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+), 
  - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+), 
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

Our scripts require the user to have the [docker group membership](https://docs.docker.com/install/linux/linux-postinstall/)
so that docker commands can be run without sudo.
We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and NVIDIA TITAN RTX.
We use mixed-precision training hence GPUs with Tensor Cores are recommended.


## Getting Started

1. Create a folder that stores pretrained models, all the data, and results.
    ```bash
    PATH_TO_STORAGE=/path/to/your/data/
    mkdir -p $PATH_TO_STORAGE/txt_db  # annotations
    mkdir -p $PATH_TO_STORAGE/video_db  # event graph features
    mkdir -p $PATH_TO_STORAGE/finetune  # finetuning results
    mkdir -p $PATH_TO_STORAGE/pretrained  # pretrained models
    ```

2. Download test data and pretrained models.

    Our constructed event graphs of the sampled videos can be downloaded from Google Drive:          
    https://drive.google.com/drive/folders/1tQEU86FpNochqEpvSFFvJ3fI5Yza-Bd9?usp=sharing.
    Please put them under the $PATH_TO_STORAGE/video_db.
    
    The sampled QA pairs can be downloaded from Google Drive:
    https://drive.google.com/file/d/1MtEh491RNMXCWdnABa_fC0uiBLoJgfE5/view?usp=sharing.
    Please put the file under the $PATH_TO_STORAGE/txt_db.
    
    Our pretrained HSTT model (186MB), can be downloaded from Google Drive:
    https://drive.google.com/file/d/1A_wmjjvEoy1vbbNuLVtBQyVXD_zTp1DY/view?usp=sharing.
    Please put it under the $PATH_TO_STORAGE/finetune/agqa_expm_balanced_dfs/ckpt.
    
    The pretrained BERT model can be downloaded from Google Drive:
    https://drive.google.com/drive/folders/1t6Jax7I58UckbGXrsZa6hR2-qfgdsV_0?usp=sharing.
    Please put it under the $PATH_TO_STORAGE/pretrained.

3. Launch the Docker container for running the experiments.
    ```bash
    # docker image should be automatically pulled
    source launch_container.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/video_db \
        $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
    ```
    The launch script respects $CUDA_VISIBLE_DEVICES environment variable.
    Note that the source code is mounted into the container under `/clipbert` instead 
    of built into the image so that user modification will be reflected without
    re-building the image. (Data folders are mounted into the container separately
    for flexibility on folder structures.)

4. Run inference.
    ```bash
    sh inference.sh
    ```
    
    The results will be written under `$PATH_TO_STORAGE/pretrained/agqa_expm_balanced_dfs/results_agqa`.
    Please check the `scores.json` for all accuracy of different metrics.

## Acknowledgement

This code used resources from [ClipBERT](https://github.com/jayleicn/ClipBERT), [HERO](https://github.com/linjieli222/HERO), 
The code is implemented using [PyTorch](https://github.com/pytorch/pytorch), 
with multi-GPU support from [Horovod](https://github.com/horovod/horovod) 
and mixed precision support from [apex](https://github.com/NVIDIA/apex).  We thank the authors for open-sourcing their awesome projects.

