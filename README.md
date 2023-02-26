# Project Description

This project is the code of WACV submission: MT-DETR: Robust End-to-end Multimodal Detection with Confidence Fusion. The code is based on mmdetection and cbnetv2.

To run the code:

1. Construct an environment first, please follow the cbnetv2 (https://github.com/VDIGPKU/CBNetV2) and mmdetection (https://github.com/open-mmlab/mmdetection) tutorial.
2. Download the dataset and model checkpoints. Please go to data/ and checkpoint/ and read the instructions there to download.
3. After preparation, type the following command in your terminal:
```bash
bash run_script/$script_name$
You can comment training/inference block in shell scripts if you want.

The following are the important directories of this project:

- `data`: download the dataset here
- `checkpoint`: download model weights here
- `run_script`: shell files for running models, change your path and GPU_id here
- `configs`: configs of models, adjust the models' setting here

