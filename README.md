# Bbox3d
3d Bounding box visualization of the 2d detected objects in RVIZ.

# Installation:
- Install the ROS dependencies
```sh
$ sudo apt-get install ros-kinetic-cv-bridge ros-kinetic-image-transport ros-kinetic-pcl-ros
``` 
- Get the [darknet_ros](https://github.com/leggedrobotics/darknet_ros) package. 
- Put both the packages inside ros workspace and use catkin_make to build them.
  ```sh
  $ cd BBox3d
  $ cp -r ./* ~/catkin_ws/src/
  $ roscd 
  $ catkin_make
  ```
# Running:
- First command starts the yolo detection. You should see the demo image with detection bbox after running it 
- Second command starts the 3d bounding box detection and RVIZ for visualization. You should be able
to see the point cloud and 3d bounding boxes in RVIZ around different objects. Please note that
rviz is not displaying marker text because of some bug.
```sh
$ roslaunch darknet_ros darknet_ros.launch
$ roslaunch bbox3d bbox3d.launch
```
# Running Version 2:
Less delay and better bounding box computation. Please run the following commands to start detection:
```sh
$ roslaunch darknet_ros darnet_ros.launch
$ roslaunch bbox3d bbox3d_ver2.launch
```

# Running Version 3:
Version-3 provides tracking along with bounding box computation. It saves the labels of tracked objects in Kitti data fromat in the file /tmp/labels.txt and also saves the segmented point cloud in /tmp directory. To enable writing the point clouds please set the enable_pcd_write argument to true inside bbox3d_ver3.launch file. To clearly see the segmented point cloud (White) in RVIZ just unsubscribe from /velodyne_points topic.
```sh
$ roslaunch darknet_ros darknet_ros.launch
$ roslaunch bbox3d bbox3d_ver3.launch
```



 

