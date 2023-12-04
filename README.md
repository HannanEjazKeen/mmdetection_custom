# Configuration Files and Python Codes
This repository contains mmdetection configurations and codes for Custom Dataset.

Clone this folder within mmdetection/config and run.


## How to Configure Environment

### Installing Anaconda
First install anaconda from https://www.anaconda.com/download/

To update the latest version of anaconda
```bash
conda update -n base -c defaults conda
```

### Setting-up Conda Environment for MMDetection
Create a seperate conda environment for mmdetection with python 3.8 version
```bash
conda create --name mmdetection python=3.8 -y
```

Activate the environment
```bash
conda activate mmdetection
```

Install pytorch directly from pytorch channel (The command on mmdetection website have cuda conflict therefore, use command from https://pytorch.org/get-started/previous-versions/)
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install MMEngine and MMCV using MIM.
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
Install mmdetection from source:
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```
To check the installed pytorch libraries (specifically cuda), use the following command
```bash
python -m torch.utils.collect_env
```

Clone this repository in mmdetection/config
```bash
clone https://github.com/HannanEjazKeen/utad_config.git uatd
```


To start the training of any configuration (for instance, detr resnet 50 network)
```bash
python tools/train.py configs/uatd/detr_r50_8xb2-150e_coco_uatd.py
```

To train on multi-gpu
```bash
CUDA_VISIBLE_DEVICES=1,2,3 ./tools/dist_train.sh configs/uatd/faster-rcnn_r50_fpn_1x_coco_uatd.py 3
```

To start the test of any configuration (for instance, detr resnet 50 network)
```bash
python tools/test.py configs/uatd/detr_r50_8xb2-150e_coco_uatd.py work_dirs/detr_r50_8xb2-150e_coco_uatd/epoch_1.pth --show
```

If you would like to visualize the training process, edit the config/_base_/default_runtime.py.
```bash
vis_backends = [dict(type='TensorboardVisBackend')]
```

Now run the command on terminal,
```bash
tensorboard --logdir work_dirs/yolox_l_8xb8-300e_coco_uatd/ --host=localhost --port=8080
```

finally, run http://localhost:8080 in chrome.

