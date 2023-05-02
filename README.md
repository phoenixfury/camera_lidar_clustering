# camera_lidar_clustering
A c++ based ROS2 package to cluster objects using the lidar pointcloud and the bounding boxes from a camera detector

- - -
## Dependencies
OpenCV:
```
$ sudo apt-get install python3-opencv
```
Image_transport:
```
$ sudo apt-get install ros-<distro>-image-transport
```
Eigen3:
```
$ sudo apt install libeigen3-dev
```
Open3d

## Usage
```
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select camera_lidar_clustering
```

## Assumption/ Known limits

This node assumes that the transformation between the camera and the lidar is published in the TF tree.

This node currently supports one lidar and one camera.
- - -
## Inputs / Outputs / API

## Launch file Inputs
Image_detector_topic - Image detector topic "tier4_perception_msgs::msg::DetectedObjectsWithFeature"

Lidar_topic - Pointcloud topic "PCL2"
- - -

## Outputs
An image displaying the projected pointcloud and bounding boxes

A list of detected objects in 3D.

A clustered pointcloud2
- - -

### Parameters

| Parameter | Description | default value |
| :----:|:---:| :-----:|
| lidar_name | Lidar frame in the TF tree | 'velodyne32' |
| Camera_name | Camera frame in the TF tree | 'sf_stereo_right' |
| k | intrinsic matrix of the calibrated camera | [1, 0, 0, 0, 1, 0, 0, 0, 1] |
| d | distortion vector of the calibrated camera | [0, 0, 0, 0, 0] |

- - -
## How to Run

```
ros2 launch
```

## Future extensions

--
