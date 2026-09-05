## 🔧Installation

We inherit the environement of [BONAI]([jwwangchn/BONAI: Official code for "Learning to Extract Building Footprints from Off-Nadir Aerial Images"](https://github.com/jwwangchn/BONAI)) and [MLS-BRN]([opendatalab/MLS-BRN: [CVPR 2024\] 3D Building Reconstruction from Monocular Remote Sensing Images with Multi-level Supervisions](https://github.com/opendatalab/MLS-BRN)), and here is a reference to deploy it:

```bash
# create & activate environment
conda create -n loft_foa_rofe python=3.8
conda activate loft_foa_rofe

# install pytorch-1.11.0 
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# install mmcv-full==1.7.0
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# install yapf-0.40.1 for compatibility with mmcf-full 1.7.0
pip install yapf==0.40.1

# install mmdet-2.28.1
cd rof/
pip install -v -e .
```
