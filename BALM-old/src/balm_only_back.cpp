#include "balmclass.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <queue>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseArray.h>

using namespace std;
double voxel_size[2] = {1, 1};

mutex mBuf;
queue<sensor_msgs::PointCloud2ConstPtr> surf_buf, corn_buf, full_buf;
queue<nav_msgs::Odometry::ConstPtr> odom_buf;

void surf_handler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  mBuf.lock();
  surf_buf.push(msg);
  mBuf.unlock();
}

void corn_handler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  mBuf.lock();
  corn_buf.push(msg);
  mBuf.unlock();
}

void full_handler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  mBuf.lock();
  full_buf.push(msg);
  mBuf.unlock();
}

void odom_handler(const nav_msgs::Odometry::ConstPtr &msg)
{
  mBuf.lock();
  odom_buf.push(msg);
  mBuf.unlock();
}

void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE*> &feat_map, pcl::PointCloud<PointType>::Ptr pl_feat, Eigen::Matrix3d R_p, Eigen::Vector3d t_p, int feattype, int fnum, int capacity)
{
  uint plsize = pl_feat->size();
  for(uint i=0; i<plsize; i++)
  {
    PointType &p_c = pl_feat->points[i];
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = R_p*pvec_orig + t_p;

    float loc_xyz[3];
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = pvec_tran[j] / voxel_size[feattype];
      if(loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->plvec_orig[fnum]->push_back(pvec_orig);
      iter->second->plvec_tran[fnum]->push_back(pvec_tran);
      iter->second->is2opt = true;
    }
    else
    {
      OCTO_TREE *ot = new OCTO_TREE(feattype, capacity);
      ot->plvec_orig[fnum]->push_back(pvec_orig);
      ot->plvec_tran[fnum]->push_back(pvec_tran);

      ot->voxel_center[0] = (0.5+position.x) * voxel_size[feattype];
      ot->voxel_center[1] = (0.5+position.y) * voxel_size[feattype];
      ot->voxel_center[2] = (0.5+position.z) * voxel_size[feattype];
      ot->quater_length = voxel_size[feattype] / 4.0;
      feat_map[position] = ot;
    }
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "balm_only_back");
  ros::NodeHandle n;

  ros::Subscriber sub_corn = n.subscribe<sensor_msgs::PointCloud2>("/corn_last", 100, corn_handler);
  ros::Subscriber sub_surf = n.subscribe<sensor_msgs::PointCloud2>("/surf_last", 100, surf_handler);
  ros::Subscriber sub_full = n.subscribe<sensor_msgs::PointCloud2>("/full_last", 100, full_handler);
  ros::Subscriber sub_odom = n.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, odom_handler);

  ros::Publisher pub_corn = n.advertise<sensor_msgs::PointCloud2>("/map_corn", 10);
  ros::Publisher pub_surf = n.advertise<sensor_msgs::PointCloud2>("/map_surf", 10);
  ros::Publisher pub_full = n.advertise<sensor_msgs::PointCloud2>("/map_full", 10);
  ros::Publisher pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 10);
  ros::Publisher pub_odom = n.advertise<nav_msgs::Odometry>("/odom_rviz_last", 10);
  ros::Publisher pub_pose = n.advertise<geometry_msgs::PoseArray>("/poseArrayTopic", 10);

  double surf_filter_length = 0.4;
  double corn_filter_length = 0.2;

  int window_size = 20;
  int margi_size = 5;
  int filter_num = 1;
  int thread_num = 4;
  int skip_num = 0;
  int pub_skip = 1;

  n.param<double>("surf_filter_length", surf_filter_length, 0.4);
  n.param<double>("corn_filter_length", corn_filter_length, 0.2);
  n.param<double>("root_surf_voxel_size", voxel_size[0], 1);
  n.param<double>("root_corn_voxel_size", voxel_size[1], 1);
  n.param<int>("skip_num", skip_num, 0);
  n.param<double>("surf_feat_eigen_limit", feat_eigen_limit[0], 9);
  n.param<double>("corn_feat_eigen_limit", feat_eigen_limit[0], 4);
  n.param<double>("surf_opt_feat_eigen_limit", feat_eigen_limit[0], 16);
  n.param<double>("corn_opt_feat_eigen_limit", feat_eigen_limit[0], 9);
  n.param<int>("pub_skip", pub_skip, 1);

  int jump_flag = skip_num;
  printf("%d\n", skip_num);
  LM_SLWD_VOXEL opt_lsv(window_size, filter_num, thread_num);

  pcl::PointCloud<PointType>::Ptr pl_corn(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr pl_surf(new pcl::PointCloud<PointType>);
  vector<pcl::PointCloud<PointType>::Ptr> pl_full_buf;

  vector<Eigen::Quaterniond> q_poses;
  vector<Eigen::Vector3d> t_poses;
  Eigen::Quaterniond q_odom, q_gather_pose(1, 0, 0, 0) ,q_last(1, 0, 0, 0); 
  Eigen::Vector3d t_odom, t_gather_pose(0, 0, 0), t_last(0, 0, 0);
  int plcount = 0, window_base = 0;
  
  unordered_map<VOXEL_LOC, OCTO_TREE*> surf_map, corn_map;
  Eigen::Matrix4d trans(Eigen::Matrix4d::Identity());
  geometry_msgs::PoseArray parray;
  parray.header.frame_id = "camera_init";

  while(n.ok())
  {
    ros::spinOnce();
    if(corn_buf.empty() || surf_buf.empty() || full_buf.empty() || odom_buf.empty())
    {
      continue;
    }

    mBuf.lock();
    uint64_t time_corn = corn_buf.front()->header.stamp.toNSec();
    uint64_t time_surf = surf_buf.front()->header.stamp.toNSec();
    uint64_t time_full = full_buf.front()->header.stamp.toNSec();
    uint64_t time_odom = odom_buf.front()->header.stamp.toNSec();

    if(time_odom != time_corn)
    {
      time_odom < time_corn ? odom_buf.pop() : corn_buf.pop();
      mBuf.unlock();
      continue;
    }
    if(time_odom != time_surf)
    {
      time_odom < time_surf ? odom_buf.pop() : surf_buf.pop();
      mBuf.unlock();
      continue;
    }
    if(time_odom != time_full)
    {
      time_odom < time_full ? odom_buf.pop() : full_buf.pop();
      mBuf.unlock();
      continue;
    }

    ros::Time ct(full_buf.front()->header.stamp);
    pcl::PointCloud<PointType>::Ptr pl_full(new pcl::PointCloud<PointType>);
    rosmsg2ptype(*surf_buf.front(), *pl_surf);
    rosmsg2ptype(*corn_buf.front(), *pl_corn);
    rosmsg2ptype(*full_buf.front(), *pl_full);
    corn_buf.pop(); surf_buf.pop(); full_buf.pop();

    q_odom.w() = odom_buf.front()->pose.pose.orientation.w;
    q_odom.x() = odom_buf.front()->pose.pose.orientation.x;
    q_odom.y() = odom_buf.front()->pose.pose.orientation.y;
    q_odom.z() = odom_buf.front()->pose.pose.orientation.z;
    t_odom.x() = odom_buf.front()->pose.pose.position.x;
    t_odom.y() = odom_buf.front()->pose.pose.position.y;
    t_odom.z() = odom_buf.front()->pose.pose.position.z;
    odom_buf.pop();
    mBuf.unlock();

    Eigen::Vector3d delta_t(q_last.matrix().transpose()*(t_odom-t_last));
    Eigen::Quaterniond delta_q(q_last.matrix().transpose() * q_odom.matrix());
    q_last = q_odom; t_last = t_odom;

    t_gather_pose = t_gather_pose + q_gather_pose * delta_t;
    q_gather_pose = q_gather_pose * delta_q;
    if(jump_flag < skip_num)
    {
      jump_flag++;
      continue;
    }
    jump_flag = 0;

    if(plcount == 0)
    {
      q_poses.push_back(q_gather_pose);
      t_poses.push_back(t_gather_pose);
    }
    else
    {
      t_poses.push_back(t_poses[plcount-1] + q_poses[plcount-1] * t_gather_pose);
      q_poses.push_back(q_poses[plcount-1]*q_gather_pose);
    }

    parray.header.stamp = ct;
    geometry_msgs::Pose apose;
    apose.orientation.w = q_poses[plcount].w();
    apose.orientation.x = q_poses[plcount].x();
    apose.orientation.y = q_poses[plcount].y();
    apose.orientation.z = q_poses[plcount].z();
    apose.position.x = t_poses[plcount].x();
    apose.position.y = t_poses[plcount].y();
    apose.position.z = t_poses[plcount].z();
    parray.poses.push_back(apose);
    pub_pose.publish(parray);

    pl_full_buf.push_back(pl_full);
    plcount++;
    OCTO_TREE::voxel_windowsize = plcount - window_base;
    q_gather_pose.setIdentity(); t_gather_pose.setZero();

    down_sampling_voxel(*pl_corn, corn_filter_length);
    down_sampling_voxel(*pl_surf, surf_filter_length);

    int frame_head = plcount-1-window_base;
    cut_voxel(surf_map, pl_surf, q_poses[plcount-1].matrix(), t_poses[plcount-1], 0, frame_head, window_size);
    cut_voxel(corn_map, pl_corn, q_poses[plcount-1].matrix(), t_poses[plcount-1], 1, frame_head, window_size);

    for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
    {
      if(iter->second->is2opt)
      {
        iter->second->root_centors.clear();
        iter->second->recut(0, frame_head, iter->second->root_centors);
      }
    }

    for(auto iter=corn_map.begin(); iter!=corn_map.end(); ++iter)
    {
      if(iter->second->is2opt)
      {
        iter->second->root_centors.clear();
        iter->second->recut(0, frame_head, iter->second->root_centors);
      }
    }

    if(plcount >= window_base+window_size)
    {
      for(int i=0; i<window_size; i++)
      {
        opt_lsv.so3_poses[i].setQuaternion(q_poses[window_base + i]);
        opt_lsv.t_poses[i] = t_poses[window_base + i];
      }

      if(window_base != 0)
      {
        for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
        {
          if(iter->second->is2opt)
          {
            iter->second->traversal_opt(opt_lsv);
          }
        }

        for(auto iter=corn_map.begin(); iter!=corn_map.end(); ++iter)
        {
          if(iter->second->is2opt)
          {
            iter->second->traversal_opt(opt_lsv);
          }
        }

        opt_lsv.damping_iter();
      }

      pcl::PointCloud<PointType> pl_send;
      
      for(int i=0; i<margi_size; i+=pub_skip)
      {
        trans.block<3, 3>(0, 0) = opt_lsv.so3_poses[i].matrix();
        trans.block<3, 1>(0, 3) = opt_lsv.t_poses[i];

        pcl::PointCloud<PointType> pcloud;
        pcl::transformPointCloud(*pl_full_buf[window_base + i], pcloud, trans);
        pl_send += pcloud;
      }

      pub_func(pl_send, pub_full, ct);

      for(int i=0; i<margi_size; i++)
      {
        pl_full_buf[window_base + i] = nullptr;
      }

      for(int i=0; i<window_size; i++)
      {
        q_poses[window_base + i] = opt_lsv.so3_poses[i].unit_quaternion();
        t_poses[window_base + i] = opt_lsv.t_poses[i];
      }

      for(int i=window_base; i<window_base+window_size; i++)
      {
        parray.poses[i].orientation.w = q_poses[i].w();
        parray.poses[i].orientation.x = q_poses[i].x();
        parray.poses[i].orientation.y = q_poses[i].y();
        parray.poses[i].orientation.z = q_poses[i].z();
        parray.poses[i].position.x = t_poses[i].x();
        parray.poses[i].position.y = t_poses[i].y();
        parray.poses[i].position.z = t_poses[i].z();
      }
      pub_pose.publish(parray);

      for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
      {
        if(iter->second->is2opt)
        {
          iter->second->root_centors.clear();
          iter->second->marginalize(0, margi_size, q_poses, t_poses, window_base, iter->second->root_centors);
        }
      }

      for(auto iter=corn_map.begin(); iter!=corn_map.end(); ++iter)
      {
        if(iter->second->is2opt)
        {
          iter->second->root_centors.clear();
          iter->second->marginalize(0, margi_size, q_poses, t_poses, window_base, iter->second->root_centors);
        }
      }


      window_base += margi_size;
      opt_lsv.free_voxel();
    }
  }
}


