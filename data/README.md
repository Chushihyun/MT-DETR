# Prepare Dataset

## Download multimodal data from STF
The input of MT-DETR has 4 kinds of sensors: Camera, Lidar, Radar, and Time.

To get the input data, please download [STF dataset](https://www.uni-ulm.de/en/in/driveu/projects/dense-datasets/) (Seeing through fog), and follow the instrction of [STF Github repo](https://github.com/princeton-computational-imaging/SeeingThroughFog).

1. Camera: `cam_stereo_left_lut` in STF dataset provides the camera data we need, please move them into `cam_stereo_left_lut/` here.

2. Lidar: `lidar_hdl64_strongest` in STF dataset provides lidar pointcloud. With the visualization tools in the STF dataset, we can obtain 2d lidar points projection with different colors representing distances. Please move them into `lidar_projection/`

3. Radar: `radar_targets` in STF dataset provides radar pointcloud. With the visualization tools in the STF dataset, we can obtain 2d radar points projection with different colors representing distances. Please move them into `radar_projection/`

4. Time: Run `create_time_data.py` to create time image for each data, and save them in `time_image/`.


## Synthesize foggy data

Follow the steps below to generate foggy camera image and foggy lidar image.
- Foggy camera image
1. Follow this repo [DPT](https://github.com/isl-org/DPT) to generate depth image and save it in `depth_image/`.
2. Run `datagen_fog.py` to generate foggy camera image and save it in `foggy_camera/`.

- Lidar image
Follow this repo [LiDAR fog simulation](https://github.com/MartinHahner/LiDAR_fog_sim) to generate foggy lidar image and save it in `foggy_lidar/`.


## Introduce directories and functions

Briefly introduce the directories under this folder (data folder):

- `cam_stereo_left_lut`: store camera image data
- `lidar_projection`: store lidar image data
- `radar_projection`: store radar image data
- `time_image`: store time image data
- `splits`: record the weather condition for each data
- `coco_annotation`: record the object position and classification for each bounding box
- `create_time_data.py`: used to create time image

- `depth_image`: store depth image data
- `foggy_camera`: store foggy camera image data
- `foggy_lidar`: store foggy lidar image data
- `datagen_fog.py`: used to create foggy camera image

- `prepare_data.txt`: this file

## Acknowledgement
Special thanks to these excellent dataset and project:
- [STF dataset](https://www.uni-ulm.de/en/in/driveu/projects/dense-datasets/)
- [STF Github repo](https://github.com/princeton-computational-imaging/SeeingThroughFog)
- [DPT](https://github.com/isl-org/DPT)
- [LiDAR fog simulation](https://github.com/MartinHahner/LiDAR_fog_sim)
