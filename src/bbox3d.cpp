#include "tf_conversions/tf_eigen.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cv_bridge/cv_bridge.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <string>
#include <tf/tf.h>
#include <visualization_msgs/MarkerArray.h>
#define LEAF_SIZE 0.1f
typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::PointCloud2, sensor_msgs::Image,
    darknet_ros_msgs::BoundingBoxes>
    MySyncPolicy;
class DrawBbox {
public:
  DrawBbox() {
    nh_ = new ros::NodeHandle("~");
    // initialize all the parameters

    r = {0.015453, 0.010533, -0.999825, 0.863324, -0.504586,
         0.008027, 0.504413, 0.863297,  0.016891};
    t = {-0.098300, -0.024300, 0.024600};
    k = {424.256690, 0.000000, 0.000000, 0.0000, 407.246190,
         0.000,      0.000,    0.000,    1.0000};
    c = {239.500000, 319.500000};
    d = {-0.295620, 0.082190, 0.000540, -0.011990, 0.000000};

    nh_->param("in_cloud_topic", in_cloud_topic,
               std::string("/velodyne_points"));
    nh_->param("out_cloud_topic", out_cloud_topic, std::string("/bbox_cloud"));
    nh_->param("in_image_topic", in_image_topic,
               std::string("/camera0/image_raw"));
    nh_->param("depth_image_topic", depth_image_topic,
               std::string("/velodyne_depth_image"));
    nh_->param("lidar_frame_id", lidar_frame_id, std::string("/velodyne"));
    nh_->param("bbox2d_topic", bbox2d_topic,
               std::string("/darknet_ros/bounding_boxes"));
    nh_->param("marker_array_topic", marker_array_topic,
               std::string("/bbox3d_marker_array"));
    nh_->param("cloud_z_thresh", cloud_z_thresh, float(0));
    nh_->getParam("R", r);
    nh_->getParam("T", t);
    nh_->getParam("K", k);
    nh_->getParam("C", c);
    nh_->getParam("D", d);
    R = cv::Mat(3, 3, CV_64FC1, r.data());
    T = cv::Mat(3, 1, CV_64FC1, t.data());
    K = cv::Mat(3, 3, CV_64FC1, k.data());
    C = cv::Mat(2, 1, CV_64FC1, c.data());
    D = cv::Mat(5, 1, CV_64FC1, d.data());

    std::cout << "using rotation matrix R = " << std::endl
              << " " << R << std::endl
              << std::endl;
    std::cout << "using translation vector = " << std::endl
              << " " << T << std::endl
              << std::endl;
    std::cout << "using Intrinsic matrix K = " << std::endl
              << " " << R << std::endl
              << std::endl;
    std::cout << "using  principal points vector C = " << std::endl
              << " " << C << std::endl
              << std::endl;
    std::cout << "using distortion coefficients D = " << std::endl
              << " " << D << std::endl
              << std::endl;
    // create pointcloud and bbox3d marker array publishers
    cloud_pub = nh_->advertise<sensor_msgs::PointCloud2>(out_cloud_topic, 1);
    bbox3d_pub =
        nh_->advertise<visualization_msgs::MarkerArray>(marker_array_topic, 1);

    // create image, bbox2d and cloud subscribers
    image_sub = new message_filters::Subscriber<sensor_msgs::Image>(
        *nh_, in_image_topic, 1);
    cloud_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(
        *nh_, in_cloud_topic, 1);
    bbox_sub = new message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes>(
        *nh_, bbox2d_topic, 1);
    sync = new message_filters::Synchronizer<MySyncPolicy>(
        MySyncPolicy(10), *cloud_sub, *image_sub, *bbox_sub);
    // bind the callbacks
    sync->registerCallback(boost::bind(&DrawBbox::SyncCB, this, _1, _2, _3));
  }

  void PubColorCloud(const pcl::PCLPointCloud2 &cloud,
                     const cv_bridge::CvImagePtr &cv_ptr,
                     const darknet_ros_msgs::BoundingBoxesConstPtr &bbox2d) {
    //  cv::imshow("image", cv_ptr->image);
    //  cv::waitKey(1);
    // extract point cloud corresponding to each bbox2d
    int num_bbox = bbox2d->boundingBoxes.size();
    int *cluster_index_counts = new int[num_bbox];
    pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_out_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr *cloud_clusters(
        new pcl::PointCloud<pcl::PointXYZRGB>::Ptr[num_bbox]);

    pcl::fromPCLPointCloud2(cloud, *in_cloud);
    // ROS_INFO("conversion to points done");
    temp_out_cloud->points.resize(in_cloud->points.size());
    for (int i = 0; i < num_bbox; i++) {
      cluster_index_counts[i] = 0;
      cloud_clusters[i].reset(new pcl::PointCloud<pcl::PointXYZRGB>);
      cloud_clusters[i]->points.resize(in_cloud->points.size());
      bbox_markers.markers.resize(i);
    }

    // iterate over all the points and assign them to correspoindg bbox cloud
    cv::Mat point3d(3, 1, CV_64FC1);
    for (size_t i = 0; i < in_cloud->points.size(); i++) {
      if (in_cloud->points[i].z > cloud_z_thresh) {
        continue;
      }
      /*  temp_out_cloud->points[i].x = in_cloud->points[i].x;
        temp_out_cloud->points[i].y = in_cloud->points[i].y;
        temp_out_cloud->points[i].z = in_cloud->points[i].z;
        temp_out_cloud->points[i].r = 255;
        temp_out_cloud->points[i].g = 255;
        temp_out_cloud->points[i].b = 255;*/

      //   ROS_INFO("image rows %d", cv_ptr->image.rows);
      point3d.at<double>(0, 0) = in_cloud->points[i].x;
      point3d.at<double>(1, 0) = in_cloud->points[i].y;
      point3d.at<double>(2, 0) = in_cloud->points[i].z;
      cv::Mat point2d(3, 1, CV_64FC1);
      // get the corresponding image pixel
      point2d = (R * point3d + T);
      double w = point2d.at<double>(2, 0);
      if (w > 0) {
        point2d = point2d / w;
        float x_1 = point2d.at<double>(0, 0);
        float y_1 = point2d.at<double>(1, 0);
        float r = x_1 * x_1 + y_1 * y_1;
        x_1 = (x_1 * (1.0 + d[0] * r * r + d[1] * r * r * r * r +
                      d[4] * r * r * r * r * r * r) +
               2 * d[2] * x_1 * y_1 + d[3] * (r * r + 2 * x_1));
        y_1 = (y_1 * (1.0 + d[0] * r * r + d[1] * r * r * r * r +
                      d[4] * r * r * r * r * r * r) +
               2 * d[3] * x_1 * y_1 + d[2] * (r * r + 2 * y_1));
        point2d.at<double>(0, 0) = x_1;
        point2d.at<double>(1, 0) = y_1;
        point2d = K * point2d;
        int x = floor(point2d.at<double>(0, 0) + C.at<double>(0, 0));
        int y = floor(point2d.at<double>(1, 0) + C.at<double>(1, 0));
        // check if pojection lies inside any bbox... if yes assign them to
        // corresponding cluster
        for (int j = 0; j < bbox2d->boundingBoxes.size(); j++) {
          // if (x >= 0 && x < cv_ptr->image.rows && y >= 0 &&
          //    y < cv_ptr->image.cols) {
          int xmin = bbox2d->boundingBoxes[j].ymin;
          int xmax = bbox2d->boundingBoxes[j].ymax;
          int ymin = bbox2d->boundingBoxes[j].xmin;
          int ymax = bbox2d->boundingBoxes[j].xmax;
          int bbox_offset = 0;
          if (x >= xmin + bbox_offset && x < xmax - bbox_offset &&
              y >= ymin + bbox_offset && y < ymax - bbox_offset) {
            //    ROS_INFO("x = %lf y = %lf", x, y);
            cv::Vec3b color_info = cv_ptr->image.at<cv::Vec3b>(x, y);
            /*   temp_out_cloud->points[i].r = 0;
               temp_out_cloud->points[i].g = 255;
               temp_out_cloud->points[i].b = 0;*/
            cloud_clusters[j]->points[cluster_index_counts[j]].x =
                in_cloud->points[i].x;
            cloud_clusters[j]->points[cluster_index_counts[j]].y =
                in_cloud->points[i].y;
            cloud_clusters[j]->points[cluster_index_counts[j]].z =
                in_cloud->points[i].z;
            cloud_clusters[j]->points[cluster_index_counts[j]].r =
                color_info[2];
            cloud_clusters[j]->points[cluster_index_counts[j]].g =
                color_info[1];
            cloud_clusters[j]->points[cluster_index_counts[j]].b =
                color_info[0];
            cluster_index_counts[j]++;
          }
        }
      }
    }
    /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr projected_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(cloud_clusters[0]);
    pca.project(*cloud_clusters[0], *projected_cloud);*/
    // temp_out_cloud = cloud_clusters[0];
    // get subclusters of each bbox point cloud
    for (int i = 0; i < num_bbox; i++) {
      cloud_clusters[i]->points.resize(cluster_index_counts[i]);
      cloud_clusters[i]->height = cluster_index_counts[i];
      pcl::VoxelGrid<pcl::PointXYZRGB> vg;
      // ROS_INFO("starting clustering");
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>);
      vg.setInputCloud(cloud_clusters[i]);
      vg.setLeafSize(0.1f, 0.1f, 0.1f);
      vg.filter(*filtered_cloud);
      *cloud_clusters[i] = *filtered_cloud;
      filtered_cloud->points.resize(0);
      // initialize Kd tree and Clustering algorithm
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(
          new pcl::search::KdTree<pcl::PointXYZRGB>);
      tree->setInputCloud(cloud_clusters[i]);
      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
      ec.setClusterTolerance(0.3);
      ec.setMinClusterSize(100);
      ec.setMaxClusterSize(15000);
      ec.setSearchMethod(tree);
      ec.setInputCloud(cloud_clusters[i]);
      ec.extract(cluster_indices);
      // get the cluster index corresponding to closest cluster (most of the
      // time
      // this will be the detected object .. // needs to be better
      int significant_cluster_idx = 0;
      int t = 0;
      float closest_centroid = std::numeric_limits<float>::max();
      for (std::vector<pcl::PointIndices>::const_iterator
               it = cluster_indices.begin();
           it != cluster_indices.end(); it++, t++) {
        Eigen::Vector4f sub_cluster_centroid;
        pcl::compute3DCentroid(*cloud_clusters[i], *it, sub_cluster_centroid);
        if (closest_centroid > sub_cluster_centroid.squaredNorm()) {
          closest_centroid = sub_cluster_centroid.squaredNorm();
          significant_cluster_idx = t;
        }
      }
      if (t != 0) {
        /* pcl::ExtractIndices<pcl::PointXYZRGB> extract;
         extract.setInputCloud(cloud_clusters[i]);
         ROS_INFO("num of clusters = %d", t);
         ROS_INFO("cluster index size = %d",
                  cluster_indices[significant_cluster_idx].indices.size());
         pcl::PointIndices::Ptr cluster_index_pointer(new pcl::PointIndices);
         //  cluster_index_pointer.reset(
         //    &(cluster_indices[significant_cluster_idx]));
         //  ROS_INFO("cluster index size2 = %d",
         //         cluster_index_pointer->indices.size());
         //  extract.setIndices(cluster_index_pointer);
         extract.setNegative(false);*/
        //   extract.filter(*filtered_cloud);
        int cluster_size =
            cluster_indices[significant_cluster_idx].indices.size();
        for (int j = 0; j < cluster_size; j++) {
          filtered_cloud->points.push_back(
              cloud_clusters[i]->points[cluster_indices[significant_cluster_idx]
                                            .indices[j]]);
        }
      } else {
        // if no significant clusters found return
        return;
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloud_clusters[i]);
        sor.setMeanK(10);
        sor.setStddevMulThresh(0.1);
        // ROS_INFO("applying filtering");
        sor.filter(*filtered_cloud);
      }
      //  cloud_clusters[i]->reset(filtered_cloud);
      //     filtered_cloud = cloud_clusters[i];
      //  ROS_INFO("filter applied");
      // start PCA for significant cluster to get the oreinted bounding box
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr projected_cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PCA<pcl::PointXYZRGB> pca;
      pca.setInputCloud(filtered_cloud);
      pca.project(*filtered_cloud, *projected_cloud);
      Eigen::Matrix3f eigen_vector_pca = pca.getEigenVectors();
      Eigen::Vector3f eigen_values = pca.getEigenValues();
      Eigen::Matrix3d eigen_vector_pca_double = eigen_vector_pca.cast<double>();
      pcl::PointXYZRGB min_point, max_point;
      pcl::getMinMax3D(*projected_cloud, min_point, max_point);
      // std::cout << "Eigen vectors" << std::endl
      //         << eigen_vector_pca << std::endl;
      Eigen::Vector4f cluster_centroid;
      pcl::compute3DCentroid(*filtered_cloud, cluster_centroid);
      //    *temp_out_cloud += *filtered_cloud;
      //  std::cout << "cluster_centroid" << std::endl
      //          << cluster_centroid << std::endl;
      tf::Quaternion quat;
      tf::Matrix3x3 tf_rotation;
      tf::matrixEigenToTF(eigen_vector_pca_double, tf_rotation);
      double roll, pitch, yaw;
      //   tf_rotation.getRPY(roll, pitch, yaw);
      tf_rotation.getRotation(quat);
      // quat.setRPY(0, 0, yaw);
      // create marker correspoinding to the bbox
      visualization_msgs::Marker marker;
      marker.header.frame_id = lidar_frame_id;
      // marker.text = bbox2d->boundingBoxes[i].Class;
      marker.text = "car";
      marker.header.stamp = ros::Time::now();
      marker.id = i;
      marker.type = visualization_msgs::Marker::CUBE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.lifetime = ros::Duration(0.2);
      marker.pose.position.x = cluster_centroid(0);
      marker.pose.position.y = cluster_centroid(1);
      marker.pose.position.z = cluster_centroid(2);
      marker.pose.orientation.x = quat.getAxis().getX();
      marker.pose.orientation.y = quat.getAxis().getY();
      marker.pose.orientation.z = quat.getAxis().getZ();
      marker.pose.orientation.w = quat.getW();
      marker.scale.x = max_point.x - min_point.x + 0.2;
      marker.scale.y = max_point.y - min_point.y + 0.2;
      marker.scale.z = max_point.z - min_point.z + 0.2;
      //  marker.scale.x = 3;
      //  marker.scale.y = 2;
      //  marker.scale.z = 2;
      marker.color.a = 0.2;
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 0;
      bbox_markers.markers.push_back(marker);
      //  ROS_INFO("pushed the marker");
      //   break;

      // *temp_out_cloud += *cloud_clusters[i];
    }
    //    for (int j = 0; j < bbox2d->boundingBoxes.size(); j++) {
    //   }
    /*    pcl::PCLPointCloud2 *temp_cloud = new pcl::PCLPointCloud2;
        // ROS_INFO("converting to pcl cloud");
        pcl::toPCLPointCloud2(*temp_out_cloud, *temp_cloud);
        pcl_conversions::fromPCL(*temp_cloud, out_cloud);
        out_cloud.header.stamp = ros::Time::now();
        out_cloud.header.frame_id = lidar_frame_id; */
    // publish point cluster cloud with color and bbox
    bbox3d_pub.publish(bbox_markers);
    // cloud_pub.publish(out_cloud);
  }
  void SyncCB(const sensor_msgs::PointCloud2ConstPtr &cloud,
              const sensor_msgs::ImageConstPtr &img,
              const darknet_ros_msgs::BoundingBoxesConstPtr &bbox2d) {
    if ((cloud->width * cloud->height) == 0) {
      return; // cloud is not dense;
    }
    // converting to PCL point cloud type
    //   ROS_INFO("converting point cloud type");
    pcl::PCLPointCloud2 *in_cloud = new pcl::PCLPointCloud2;
    pcl_conversions::toPCL(*cloud, *in_cloud);
    //   ROS_INFO("conversion done");

    /* pcl::PCLPointCloud2 cloud_filtered;
     pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
     pcl::PCLPointCloud2ConstPtr cloudPtr(in_cloud);
     ROS_INFO("setting input cloud");
     sor.setInputCloud(cloudPtr);
     sor.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
     ROS_INFO("applying filter");
     sor.filter(cloud_filtered);
     ROS_INFO("publishing filtered cloud");
     pcl_conversions::fromPCL(cloud_filtered, out_cloud);*/
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    // pass cloud, image and bbox2d to get 3d bbox
    PubColorCloud(*in_cloud, cv_ptr, bbox2d);
  }

private:
  ros::NodeHandle *nh_;
  sensor_msgs::PointCloud2 out_cloud;
  // topic subscribers
  std::string in_cloud_topic, out_cloud_topic, depth_image_topic,
      in_image_topic, lidar_frame_id, bbox2d_topic, marker_array_topic;
  // pointcloud, detection image , bounding box subscribers
  message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud_sub;
  message_filters::Subscriber<sensor_msgs::Image> *image_sub;
  message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> *bbox_sub;
  // sync policy
  message_filters::Synchronizer<MySyncPolicy> *sync;
  ros::Publisher cloud_pub, depth_image_pub, bbox3d_pub;
  // remove the cloud above this value
  float cloud_z_thresh;
  // camera lidar calibration data, ros parameters
  std::vector<double> r, t, k, c, d;
  // camera lidar calibration data , cv mat
  cv::Mat R, T, K, C, D;
  visualization_msgs::MarkerArray bbox_markers;
};
int main(int argc, char **argv) {
  ros::init(argc, argv, "bbox");
  DrawBbox draw_bbox;
  ros::spin();
}
