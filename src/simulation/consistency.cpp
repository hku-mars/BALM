#include "tools.hpp"
// #include "BAs.hpp"
#include "BAs_left.hpp"
#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Eigenvalues>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_broadcaster.h>
#include <random>

using namespace std;

template <typename T>
void pub_pl_func(T &pl, ros::Publisher &pub)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "camera_init";
  output.header.stamp = ros::Time::now();
  pub.publish(output);
}

void pub_odom_func(IMUST &xcurr)
{
  Eigen::Quaterniond q_this(xcurr.R);
  Eigen::Vector3d t_this(xcurr.p);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(t_this.x(), t_this.y(), t_this.z()));
  q.setW(q_this.w());
  q.setX(q_this.x());
  q.setY(q_this.y());
  q.setZ(q_this.z());
  transform.setRotation(q);
  ros::Time ct = ros::Time::now();
  br.sendTransform(tf::StampedTransform(transform, ct, "/camera_init", "/aft_mapped"));
}

ros::Publisher pub_test, pub_curr, pub_full, pub_path;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "simu2");
  ros::NodeHandle n;

  pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 100);
  pub_curr = n.advertise<sensor_msgs::PointCloud2>("/map_curr", 100);
  pub_full = n.advertise<sensor_msgs::PointCloud2>("/map_full", 100);
  pub_path = n.advertise<sensor_msgs::PointCloud2>("/map_path", 100);

  n.param<double>("pnoise", pnoise, 0.02);
  string file_path;
  n.param<string>("file_path", file_path, "");

  int pose_size = 101;
  fstream inFile(file_path + "/datas/consistency/lidarPose.csv", ios::in);
  int jump_num = 1;

  string lineStr, str;
  PLV(3) poss(pose_size);
  PLM(3) rots(pose_size);
  Eigen::Matrix4d aff;
  vector<double> nums;
  
  for(int i=0; i<pose_size; i++)
  {
    nums.clear();
    for(int j=0; j<4; j++)
    {
      getline(inFile, lineStr);
      stringstream ss(lineStr);
      while(getline(ss, str, ','))
        nums.push_back(stod(str));
    }

    for(int j=0; j<16; j++)
      aff(j) = nums[j];
    
    Eigen::Matrix4d affT = aff.transpose();

    static Eigen::Vector3d pos0 = affT.block<3, 1>(0, 3);

    rots[i] = affT.block<3, 3>(0, 0);
    poss[i] = affT.block<3, 1>(0, 3) - pos0;
  }

  ros::Rate rate(10);
  pcl::PointCloud<pcl::PointXYZ> pl_orig;
  pcl::PointCloud<PointType> pl_full, pl_surf, pl_path, pl_send, pl_send2;

  sleep(1.5);
  for(int iterCount=0; iterCount<1 && n.ok(); iterCount++)
  {
    int win_count = 0;
    int win_base = 0;
    unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
    vector<IMUST> x_buf(win_size+fix_size);

    printf("Show the point cloud generated by simulator...\n");
    for(int m=0; m<pose_size&&n.ok(); m++)
    {
      string filename = file_path + "/datas/consistency/" + to_string(m+1) + ".pcd";
      pl_surf.clear();
      pcl::io::loadPCDFile(filename, pl_surf);

      win_count++;
      x_buf[win_count-1].R = rots[m];
      x_buf[win_count-1].p = poss[m];

      pl_send = pl_surf;
      pl_transform(pl_send,x_buf[win_count-1].R, x_buf[win_count-1].p);
      PointType ap;
      ap.x = x_buf[win_count-1].p[0];
      ap.y = x_buf[win_count-1].p[1];
      ap.z = x_buf[win_count-1].p[2];
      pl_path.push_back(ap);
      pub_pl_func(pl_path, pub_path);
      pub_pl_func(pl_send, pub_curr);
      pub_pl_func(pl_send, pub_full);
      pub_odom_func(x_buf[win_count-1]);
      rate.sleep();

      cut_voxel(surf_map, pl_surf, x_buf[win_count-1], win_count-1);

      if(win_count < win_size + fix_size) continue;

      vector<IMUST> x_buf2;
      for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
      {
        iter->second->recut(win_count);
        iter->second->marginalize(fix_size, x_buf2, win_count);
      }

      x_buf2.resize(win_size);
      for(int i=0; i<win_size; i++)
        x_buf2[i] = x_buf[fix_size + i];
      x_buf = x_buf2;

      win_count -= fix_size;

      printf("The size of poses: %d\n", win_count);

      default_random_engine e(ros::Time::now().toSec());
      // default_random_engine e(200);

      /* Corrept the point cloud with random noise */
      for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
        iter->second->corrupt(e, x_buf, win_count);
      
      VOX_HESS voxhess;
      for(auto iter=surf_map.begin(); iter!=surf_map.end(); iter++)        
        iter->second->tras_opt(voxhess, win_count);
      
      printf("Begin to optimize...\n");
      
      Eigen::MatrixXd Rcov(6*win_size, 6*win_size); Rcov.setZero();
      BALM2 opt_lsv;
      opt_lsv.damping_iter(x_buf, voxhess, Rcov);

      for(auto iter=surf_map.begin(); iter!=surf_map.end(); iter++)
        delete iter->second;
      surf_map.clear();

      Eigen::VectorXd err(6*win_size); err.setZero();
      for(int i=0; i<win_size; i++)
      {
        // err.block<3, 1>(6*i, 0) = Log(x_buf[i].R.transpose() * x_buf2[i].R);
        // err.block<3, 1>(6*i+3, 0) = (x_buf2[i].p - x_buf[i].p);
        err.block<3, 1>(6*i, 0) = Log(x_buf2[i].R * x_buf[i].R.transpose());
        err.block<3, 1>(6*i+3, 0) = -x_buf2[i].R * x_buf[i].R.transpose() * x_buf[i].p + x_buf2[i].p;
      }

      double nees = err.transpose() * Rcov.inverse() * err;
      printf("The expected NEES is 6*%d = %d.\n", win_size, 6*win_size);
      printf("The NEES for this Monto-Carlo experiment is %lf.\n", nees);

      /* 3 sigma bounds */
      // for(int i=0; i<win_size; i++)
      // {
      //   cout << i << ", " << err(6*i) << ", " << err(6*i+1) << ", " << err(6*i+2) << ", " << err(6*i+3) << ", " << err(6*i+4) << ", " << err(6*i+5) << ", ";
      //   cout << sqrt(Rcov(6*i, 6*i)) << "," << sqrt(Rcov(6*i+1, 6*i+1)) << "," << sqrt(Rcov(6*i+2, 6*i+2)) << "," << sqrt(Rcov(6*i+3, 6*i+3)) << "," << sqrt(Rcov(6*i+4, 6*i+4)) << "," << sqrt(Rcov(6*i+5, 6*i+5)) << endl;
      // }

      /* NEES for each pose (Require multiple Monte-Carlo experiments) */
      // for(int i=0; i<win_size; i++)
      // {
      //   Eigen::Matrix<double, 6, 1> err6 = err.block<6, 1>(6*i, 0);
      //   Eigen::Matrix<double, 6, 6> Rcov6 = Rcov.block<6, 6>(6*i, 6*i);
      //   double nees = err6.transpose() * Rcov6.inverse() * err6;
      //   double neesR = err6.head(3).transpose() * Rcov6.block<3, 3>(0, 0).inverse() * err6.head(3);
      //   double neesT = err6.tail(3).transpose() * Rcov6.block<3, 3>(3, 3).inverse() * err6.tail(3);
      //   cout << i << " " << nees << " " << neesR << " " << neesT << endl;
      // }

      break;
    }

  }

  ros::spin();
}
