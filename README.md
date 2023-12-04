# UATD Configuration Files and Python Codes
This repository contains mmdetection configurations and codes for UATD Dataset.

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

Install pytorch directly from pytorch channel
```bash
conda install pytorch torchvision -c pytorch
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
