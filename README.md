# Merging_pointClouds
This Python script provides functionality for merging LiDAR point clouds obtained from positions using Iterative Closest Point (ICP) algorithm. The script also performs transformations, such as GPS to Cartesian conversion, camera projection, and color mapping to create a merged and colored point cloud map.

## Prerequisites

Make sure you have the required libraries installed. You can install them using the following:

```bash
pip install numpy pyproj pandas pillow plyfile opencv-python
```

## Input

The script expects LiDAR point cloud files in XYZ format and GPS data in CSV format. 4 Cameras are expected and their transformations could be edited in the main code. Camera images should be organized in folders named DEV0, DEV1, DEV2, and DEV3.


## Author
Amine Dhemaied
IFROS master student
