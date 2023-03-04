# MT-DETR

This project is the code of **WACV 2023** paper: [MT-DETR: Robust End-to-end Multimodal Detection with Confidence Fusion](https://openaccess.thecvf.com/content/WACV2023/papers/Chu_MT-DETR_Robust_End-to-End_Multimodal_Detection_With_Confidence_Fusion_WACV_2023_paper.pdf) by *Shih-Yun Chu*, *Ming-Sui Lee*. You can find more visualized result and details in [supplementary material](https://openaccess.thecvf.com/content/WACV2023/supplemental/Chu_MT-DETR_Robust_End-to-End_WACV_2023_supplemental.pdf).

In the application of autonomous driving, there are times when unexpected and severe weather occurs in outdoor environments, making the detection tasks less effective and unexpected. Therefore, this paper proposes a **novel multimodal object detection network called MT-DETR**, which adopts RFM, CFM, REM, and hierarchical fusion mechanism to perform crossmodal fusion and exchange. Additionally, MT-DETR employs a multistage loss to address the imbalance among vehicle sensors and learn to extract compelling features. MT-DETR achieves **state-of-the-art performance** using the camera, lidar and radar, and even better with additional time information and the proposed synthetic foggy training data. The experimental results demonstrate that the MT-DETR is **robust and performs well in various weather conditions**. The **good generalization and scalability** confirm future applicability to different multimodal tasks.

<div align=center>  
<img src='figure/architecture.png' width="95%">
</div>

<div align=center>  
<img src='figure/comparison.png' width="45%"> <img src='figure/input_sensor.png' width="45%">
</div>

## Visualized Results in different weather

<div align=center>  
<img src='figure/snow_day.png' width="45%"> <img src='figure/densefog_day.png' width="45%">
<img src='figure/lightfog_night.png' width="45%"> <img src='figure/clear_night.png' width="45%">
</div>



## Getting Started
The repository is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [cbnetv2](https://github.com/VDIGPKU/CBNetV2). Many thanks for their awesome open-source project.

To run the code:

1. Construct an environment first, please follow the cbnetv2 (https://github.com/VDIGPKU/CBNetV2) and mmdetection (https://github.com/open-mmlab/mmdetection) tutorial.
2. Download the dataset and model checkpoints. Please go to `data/` and `checkpoint/` and read the instructions there to download.
3. After preparation, type the following command in your terminal:
```
bash run_script/${script_name}
```
You can comment training/inference block in shell scripts if you want.

The following are the important directories of this project:

- `data`: download the dataset here
- `checkpoint`: download model weights here
- `run_script`: shell files for running models, change your path and GPU_id here
- `configs`: configs of models, adjust models' setting here
- `mmdet/models/backbones/mt_detr.py`,`mmdet/models/backbones/fusion_module.py`: core model architecture of MT-DETR (this paper)


## BibTeX

If you find our work useful in your research, please consider citing our [paper](https://openaccess.thecvf.com/content/WACV2023/papers/Chu_MT-DETR_Robust_End-to-End_Multimodal_Detection_With_Confidence_Fusion_WACV_2023_paper.pdf).
```
@InProceedings{Chu_2023_WACV,
    author    = {Chu, Shih-Yun and Lee, Ming-Sui},
    title     = {MT-DETR: Robust End-to-End Multimodal Detection With Confidence Fusion},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {5252-5261}
}
```