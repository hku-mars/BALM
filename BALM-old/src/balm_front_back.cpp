#include "balmclass.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <queue>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseArray.h>

using namespace std;
double voxel_size[2] = {1, 1}; // {surf, corn}

mutex mBuf;
queue<sensor_msgs::PointCloud2ConstPtr> surf_buf, corn_buf, full_buf;

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

// Put feature points into root voxel
// feat_map: The hash table which manages voxel map
// pl_feat: Current feature pointcloud
// R_p, t_p: Current pose
// feattype: 0 is surf, 1 is corn
// fnum: The position in sliding window
// capacity: The capacity of sliding window, a little bigger than windowsize
void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE*> &feat_map, pcl::PointCloud<PointType>::Ptr pl_feat, Eigen::Matrix3d R_p, Eigen::Vector3d t_p, int feattype, int fnum, int capacity)
{
  uint plsize = pl_feat->size();
  for(uint i=0; i<plsize; i++)
  {
    // Transform point to world coordinate
    PointType &p_c = pl_feat->points[i];
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = R_p*pvec_orig + t_p;

    // Determine the key of hash table
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

    // Find corresponding voxel
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->plvec_orig[fnum]->push_back(pvec_orig);
      iter->second->plvec_tran[fnum]->push_back(pvec_tran);
      iter->second->is2opt = true;
    }
    else // If not finding, build a new voxel
    {
      OCTO_TREE *ot = new OCTO_TREE(feattype, capacity);
      ot->plvec_orig[fnum]->push_back(pvec_orig);
      ot->plvec_tran[fnum]->push_back(pvec_tran);

      // Voxel center coordinate
      ot->voxel_center[0] = (0.5+position.x) * voxel_size[feattype];
      ot->voxel_center[1] = (0.5+position.y) * voxel_size[feattype];
      ot->voxel_center[2] = (0.5+position.z) * voxel_size[feattype];
      ot->quater_length = voxel_size[feattype] / 4.0; // A quater of side length
      feat_map[position] = ot;
    }
  }
}


int main(int argc, char **argv) 
{
  ros::init(argc, argv, "balm_front_back");
  ros::NodeHandle n;

  ros::Subscriber sub_surf, sub_corn, sub_full;
  
  sub_surf = n.subscribe<sensor_msgs::PointCloud2>("/pc2_surfN", 100, surf_handler);
  sub_corn = n.subscribe<sensor_msgs::PointCloud2>("/pc2_cornN", 100, corn_handler);
  sub_full = n.subscribe<sensor_msgs::PointCloud2>("/pc2_fullN", 100, full_handler);
  
  ros::Publisher pub_full = n.advertise<sensor_msgs::PointCloud2>("/map_full", 10);
  ros::Publisher pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 10);
  ros::Publisher pub_odom = n.advertise<nav_msgs::Odometry>("/odom_rviz_last", 10);
  ros::Publisher pub_pose = n.advertise<geometry_msgs::PoseArray>("/poseArrayTopic", 10);


  int accumulate_window = 3;
  double surf_filter_length = 0.4;
  double corn_filter_length = 0.2;
  int window_size = 20; // sliding window size
  int margi_size = 5; // margilization size
  int filter_num = 1; // for map-refine LM optimizer
  int thread_num = 4; // for map-refine LM optimizer
  int scan2map_on = 10;
  int pub_skip = 1;

  n.param<int>("accumulate_window", accumulate_window, 3);
  n.param<double>("surf_filter_length", surf_filter_length, 0.4);
  n.param<double>("corn_filter_length", corn_filter_length, 0.2);
  n.param<double>("root_surf_voxel_size", voxel_size[0], 1);
  n.param<double>("root_corn_voxel_size", voxel_size[1], 1);
  n.param<int>("scan2map_on", scan2map_on, 10);
  n.param<int>("pub_skip", pub_skip, 1);
  printf("%lf %lf\n", voxel_size[0], voxel_size[1]);

  Eigen::Quaterniond q_curr(1, 0, 0, 0);
  Eigen::Vector3d t_curr(0, 0, 0);

  pcl::PointCloud<PointType>::Ptr pl_corn(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr pl_surf(new pcl::PointCloud<PointType>);
  vector<pcl::PointCloud<PointType>::Ptr> pl_full_buf;

  // Number of received scans
  int plcount = 0; 
  // The sequence of frist scan in the sliding window. 
  // Plus margi_size after every map-refine
  int window_base = 0; 

  Eigen::Quaterniond delta_q(1, 0, 0, 0);
  Eigen::Vector3d delta_t(0, 0, 0);

  // The hash table of voxel map
  unordered_map<VOXEL_LOC, OCTO_TREE*> surf_map, corn_map;

  vector<Eigen::Quaterniond> delta_q_buf, q_buf;
  vector<Eigen::Vector3d> delta_t_buf, t_buf;

  // LM optimizer for map-refine
  LM_SLWD_VOXEL opt_lsv(window_size, filter_num, thread_num);

  pcl::PointCloud<PointType> pl_surf_centor_map, pl_corn_centor_map;
  pcl::PointCloud<PointType> pl_corn_fil_map, pl_surf_fil_map;

  pcl::KdTreeFLANN<PointType>::Ptr kdtree_surf(new pcl::KdTreeFLANN<PointType>());
  pcl::KdTreeFLANN<PointType>::Ptr kdtree_corn(new pcl::KdTreeFLANN<PointType>());

  pcl::PointCloud<PointType> pl_send;
  Eigen::Vector3d p_orig, aft_tran, kervec, orient, v_ac;
  uint a_size;
  PointType apy;
  vector<int> pointSearchInd; vector<float> pointSearchSqDis;
  double range;
  Eigen::Matrix4d trans(Eigen::Matrix4d::Identity());
  ros::Time ct;
  geometry_msgs::PoseArray parray;
  parray.header.frame_id = "camera_init";
  thread *map_refine_thread = nullptr;

  while(n.ok())
  {
    ros::spinOnce();

    // The thread of map_refine is done
    if(opt_lsv.read_refine_state() == 2)
    {
      nav_msgs::Odometry laser_odom;
      laser_odom.header.frame_id = "camera_init";
      laser_odom.header.stamp = ct;

      // Publish points whose pose is fixed
      pl_send.clear();
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
        q_buf[window_base+i] = opt_lsv.so3_poses[i].unit_quaternion();
        t_buf[window_base+i] = opt_lsv.t_poses[i];
      }

      // for(int i=window_base+window_size; i<plcount; i++)
      // {
      //   t_buf[i] = t_buf[i-1] + q_buf[i-1] * delta_t_buf[i];
      //   q_buf[i] = q_buf[i-1] * delta_q_buf[i];
      // }

      // Publish poses
      for(int i=window_base; i<plcount; i++)
      {
        parray.poses[i].orientation.w = q_buf[i].w();
        parray.poses[i].orientation.x = q_buf[i].x();
        parray.poses[i].orientation.y = q_buf[i].y();
        parray.poses[i].orientation.z = q_buf[i].z();
        parray.poses[i].position.x = t_buf[i].x();
        parray.poses[i].position.x = t_buf[i].x();
        parray.poses[i].position.x = t_buf[i].x();
      }
      pub_pose.publish(parray);
      
      // Marginalization and update voxel map
      pl_surf_centor_map.clear(); pl_corn_centor_map.clear();
      for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
      {
        if(iter->second->is2opt)
        {
          iter->second->root_centors.clear();
          iter->second->marginalize(0, margi_size, q_buf, t_buf, window_base, iter->second->root_centors);
        }
        pl_surf_centor_map += iter->second->root_centors;
      }

      for(auto iter=corn_map.begin(); iter!=corn_map.end(); ++iter)
      {
        if(iter->second->is2opt)
        {
          iter->second->root_centors.clear();
          iter->second->marginalize(0, margi_size, q_buf, t_buf, window_base, iter->second->root_centors);
        }
        pl_corn_centor_map += iter->second->root_centors;
      }
      
      // If use multithreading mode, clear memory
      if(map_refine_thread != nullptr)
      {
        delete map_refine_thread;
        map_refine_thread = nullptr;
      }

      // window size of every voxel
      OCTO_TREE::voxel_windowsize -= margi_size;

      opt_lsv.free_voxel();
      window_base += margi_size; // as definition of window_base
      opt_lsv.set_refine_state(0);
    }
    
    if(surf_buf.empty() || corn_buf.empty() || full_buf.empty())
    {
      continue;
    }

    mBuf.lock();
    uint64_t time_surf = surf_buf.front()->header.stamp.toNSec();
    uint64_t time_corn = corn_buf.front()->header.stamp.toNSec();
    uint64_t time_full = full_buf.front()->header.stamp.toNSec();

    if(time_corn != time_surf)
    {
      time_corn < time_surf ? corn_buf.pop() : surf_buf.pop();
      mBuf.unlock();
      continue;
    }
    if(time_corn != time_full)
    {
      time_corn < time_full ? corn_buf.pop() : full_buf.pop();
      mBuf.unlock();
      continue;
    }
    
    pcl::PointCloud<PointType>::Ptr pl_full(new pcl::PointCloud<PointType>);

    ct = full_buf.front()->header.stamp;

    // Convert PointCloud2 to PointType(PointXYZINormal)
    rosmsg2ptype(*surf_buf.front(), *pl_surf);
    rosmsg2ptype(*corn_buf.front(), *pl_corn);
    rosmsg2ptype(*full_buf.front(), *pl_full);
    corn_buf.pop(); surf_buf.pop(); full_buf.pop();

    if(pl_full->size() < 5000)
    {
      mBuf.unlock();
      continue;
    }
    pl_full_buf.push_back(pl_full);
    mBuf.unlock();
    plcount++;
    // The number of scans in the sliding window
    OCTO_TREE::voxel_windowsize = plcount - window_base;
    // Down sampling like PCL voxelgrid filter
    down_sampling_voxel(*pl_corn, corn_filter_length);
    
    double time_scan2map = ros::Time::now().toSec();
    // Scan2map module
    if(plcount > accumulate_window)
    {
      down_sampling_voxel(*pl_surf, surf_filter_length);

      // The new scan2map method needs several scans to initialize for Velodyne lidar
      if(plcount <= scan2map_on)
      {
        // Similar with loam mapping
        kdtree_surf->setInputCloud(pl_surf_fil_map.makeShared());
        kdtree_corn->setInputCloud(pl_corn_fil_map.makeShared());
      }
      else
      {
        // The new scan2map
        kdtree_surf->setInputCloud(pl_surf_centor_map.makeShared());
        kdtree_corn->setInputCloud(pl_corn_centor_map.makeShared());
      }
      
      // Two-step method
      for(int itercount=0; itercount<2; itercount++)
      {
        // LM optimizer for scan2map
        VOXEL_DISTANCE sld;
        sld.so3_pose.setQuaternion(q_curr);
        sld.t_pose = t_curr;

        a_size = pl_surf->size(); 
        if(plcount <= scan2map_on)
        {
          // The method is similar with loam mapping.
          for(uint i=0; i<a_size; i++)
          {
            int ns = 5;
            p_orig << (*pl_surf)[i].x, (*pl_surf)[i].y, (*pl_surf)[i].z;
            aft_tran = q_curr*p_orig + t_curr;
            apy.x = aft_tran[0]; apy.y = aft_tran[1]; apy.z = aft_tran[2];

            kdtree_surf->nearestKSearch(apy, ns, pointSearchInd, pointSearchSqDis);

            if(pointSearchSqDis[ns-1] > 5)
            {
              continue;
            }

            Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
            Eigen::Vector3d center(0, 0, 0);
            for(int j=0; j<ns; j++)
            {
              Eigen::Vector3d tvec;
              tvec[0] = pl_surf_fil_map[pointSearchInd[j]].x;
              tvec[1] = pl_surf_fil_map[pointSearchInd[j]].y;
              tvec[2] = pl_surf_fil_map[pointSearchInd[j]].z;
              covMat += tvec * tvec.transpose();
              center += tvec;
            }

            center /= ns;
            covMat -= ns * center * center.transpose();
            covMat /= ns;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
            if(saes.eigenvalues()[2] < 25*saes.eigenvalues()[0])
            {
              continue;
            }

            kervec = center;
            orient = saes.eigenvectors().col(0);

            range = fabs(orient.dot(aft_tran - kervec));
            
            if(range > 1)
            {
              continue;
            }
            
            sld.push_surf(p_orig, kervec, orient, (1-0.75*range));
          }

          a_size = pl_corn->size();
          for(uint i=0; i<a_size; i++)
          {
            int ns = 5;
            p_orig << (*pl_corn)[i].x, (*pl_corn)[i].y, (*pl_corn)[i].z;
            aft_tran = q_curr*p_orig + t_curr;

            apy.x = aft_tran[0]; apy.y = aft_tran[1]; apy.z = aft_tran[2];
            kdtree_corn->nearestKSearch(apy, ns, pointSearchInd, pointSearchSqDis);

            if((pointSearchSqDis[ns-1]) > 5)
            {
              continue;
            }

            Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
            Eigen::Vector3d center(0, 0, 0);
            for(int j=0; j<ns; j++)
            {
              Eigen::Vector3d tvec;
              tvec[0] = pl_corn_fil_map[pointSearchInd[j]].x;
              tvec[1] = pl_corn_fil_map[pointSearchInd[j]].y;
              tvec[2] = pl_corn_fil_map[pointSearchInd[j]].z;
              covMat += tvec * tvec.transpose();
              center += tvec;
            }
            center /= ns;
            covMat -= ns * center * center.transpose();
            covMat /= ns;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
            if(saes.eigenvalues()[2] < 4*saes.eigenvalues()[1])
            {
              continue;
            }

            kervec = center;
            orient = saes.eigenvectors().col(2);
            v_ac = aft_tran - kervec;
            range = (v_ac - orient*orient.transpose()*v_ac).norm();
            if(range > 1.0)
            {
              continue;
            }
            sld.push_line(p_orig, kervec, orient, 0.5*(1-0.75*range));
          }


        }
        else
        {
          // The scan2map method described in the paper.
          for(uint i=0; i<a_size; i++)
          {
            int ns = 5;
            p_orig << (*pl_surf)[i].x, (*pl_surf)[i].y, (*pl_surf)[i].z;
            aft_tran = q_curr*p_orig + t_curr;
            apy.x = aft_tran[0]; apy.y = aft_tran[1]; apy.z = aft_tran[2];

            kdtree_surf->nearestKSearch(apy, ns, pointSearchInd, pointSearchSqDis);
          
            if((pointSearchSqDis[0]) > 1.0)
            {
              continue;
            }

            // Find the nearest plane.
            range = 10;
            for(int j=0; j<ns; j++)
            {
              // The point in "pl_surf_centor_map" is defined below. 
              PointType &ay = pl_surf_centor_map[pointSearchInd[j]];
              Eigen::Vector3d center(ay.x, ay.y, ay.z);
              Eigen::Vector3d direct(ay.normal_x, ay.normal_y, ay.normal_z);
              double dista = fabs(direct.dot(aft_tran - center));
              if(dista <= range && pointSearchSqDis[j] < 4.0)
              {
                kervec = center;
                orient = direct;
                range = dista;
              }
            }

            // Push points into optimizer
            sld.push_surf(p_orig, kervec, orient, (1-0.75*range));
          }

          // Corn features
          a_size = pl_corn->size();
          for(uint i=0; i<a_size; i++)
          {
            int ns = 3;
            p_orig << (*pl_corn)[i].x, (*pl_corn)[i].y, (*pl_corn)[i].z;
            aft_tran = q_curr*p_orig + t_curr;
            apy.x = aft_tran[0]; apy.y = aft_tran[1]; apy.z = aft_tran[2];

            kdtree_corn->nearestKSearch(apy, ns, pointSearchInd, pointSearchSqDis);
            if((pointSearchSqDis[0]) > 1)
            {
              continue;
            }

            range = 10;
            double dis_record = 10;
            for(int j=0; j<ns; j++)
            {
              PointType &ay = pl_corn_centor_map[pointSearchInd[j]];
              Eigen::Vector3d center(ay.x, ay.y, ay.z);
              Eigen::Vector3d direct(ay.normal_x, ay.normal_y, ay.normal_z);
              v_ac = aft_tran - center;
              double dista = (v_ac - direct*direct.transpose()*v_ac).norm();
              if(dista <= range)
              {
                kervec = center;
                orient = direct;
                range = dista;
                dis_record = pointSearchSqDis[j];
              }
            }

            if(range < 0.2 && sqrt(dis_record) < 1)
            {
              sld.push_line(p_orig, kervec, orient, (1-0.75*range));
            }
          }
        }   

        sld.damping_iter();
        q_curr = sld.so3_pose.unit_quaternion();
        t_curr = sld.t_pose;
      }
    }

    time_scan2map = ros::Time::now().toSec() - time_scan2map;
    // printf("Scan2map time: %lfs\n", ros::Time::now().toSec()-time_scan2map);

    if(plcount <= scan2map_on)
    {
      trans.block<3, 3>(0, 0) = q_curr.matrix();
      trans.block<3, 1>(0, 3) = t_curr;

      pcl::transformPointCloud(*pl_surf, pl_send, trans);
      pl_surf_fil_map += pl_send;
      pcl::transformPointCloud(*pl_corn, pl_send, trans);
      pl_corn_fil_map += pl_send;
      down_sampling_voxel(pl_surf_fil_map, 0.2);
      down_sampling_voxel(pl_corn_fil_map, 0.2);
    }
    
    // Put new pose into posearray which will be further modified
    parray.header.stamp = ct;
    geometry_msgs::Pose apose;
    apose.orientation.w = q_curr.w();
    apose.orientation.x = q_curr.x();
    apose.orientation.y = q_curr.y();
    apose.orientation.z = q_curr.z();
    apose.position.x = t_curr.x();
    apose.position.y = t_curr.y();
    apose.position.z = t_curr.z();
    parray.poses.push_back(apose);
    pub_pose.publish(parray);

    // Use "rostopic echo" to check the odometry
    nav_msgs::Odometry laser_odom;
    laser_odom.header.frame_id = "camera_init";
    laser_odom.header.stamp = ct;
    laser_odom.pose.pose.orientation.x = q_curr.x();
    laser_odom.pose.pose.orientation.y = q_curr.y();
    laser_odom.pose.pose.orientation.z = q_curr.z();
    laser_odom.pose.pose.orientation.w = q_curr.w();
    laser_odom.pose.pose.position.x = t_curr.x();
    laser_odom.pose.pose.position.y = t_curr.y();
    laser_odom.pose.pose.position.z = t_curr.z();
    pub_odom.publish(laser_odom);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(t_curr.x(), t_curr.y(), t_curr.z()));
    q.setW(q_curr.w());
    q.setX(q_curr.x());
    q.setY(q_curr.y());
    q.setZ(q_curr.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, laser_odom.header.stamp, "/camera_init", "/aft_mapped"));
    
    trans.block<3, 3>(0, 0) = q_curr.matrix();
    trans.block<3, 1>(0, 3) = t_curr;
    pcl::transformPointCloud(*pl_full, pl_send, trans);
    pub_func(pl_send, pub_test, ct);
    
    // Get the variation of pose
    if(plcount > 1)
    {
      delta_t = q_buf[plcount-2].matrix().transpose() * (t_curr-t_buf[plcount-2]);
      delta_q = q_buf[plcount-2].matrix().transpose() * q_curr.matrix();
    }
  
    q_buf.push_back(q_curr);
    t_buf.push_back(t_curr);
    delta_q_buf.push_back(delta_q);
    delta_t_buf.push_back(delta_t);
    
    // For multiple thread, if computer runs slow, the memory may be out of range.
    // 10 is decided in "cut_voxel"
    if(plcount-window_base-window_size > 10)
    {
      printf("Out of size\n");
      exit(0);
    }

    // Put current feature points into root voxel node
    cut_voxel(surf_map, pl_surf, q_curr.matrix(), t_curr, 0, plcount-1-window_base, window_size+10);
    cut_voxel(corn_map, pl_corn, q_curr.matrix(), t_curr, 1, plcount-1-window_base, window_size+10);
    
    // The center point of surf points and corn points
    // The normal_x(yz) in each point is normal vector for plane
    // or direction vector for line.
    pl_surf_centor_map.clear(); pl_corn_centor_map.clear();
    
    // Points in new scan have been distributed in corresponding root node voxel
    // Then continue to cut the root voxel until right size
    for(auto iter=surf_map.begin(); iter!=surf_map.end(); ++iter)
    {
      if(iter->second->is2opt) // Sliding window of root voxel should have points
      {
        iter->second->root_centors.clear();
        iter->second->recut(0, plcount-1-window_base, iter->second->root_centors);
      }

      // Add up surf centor points.
      pl_surf_centor_map += iter->second->root_centors;
      // You can add some distance restrictions in case that pl_surf_centor_map is too large.
      // You can also push points in root voxel into kdtree (loam mapping)
      // You can use "surf_map.erase(iter++)" to erase voxel for saving memory
    }

    for(auto iter=corn_map.begin(); iter!=corn_map.end(); ++iter)
    {
      if(iter->second->is2opt)
      {
        iter->second->root_centors.clear();
        iter->second->recut(0, plcount-1-window_base, iter->second->root_centors);
      }
      pl_corn_centor_map += iter->second->root_centors;
    }
    
    // Begin map refine module
    if(plcount>=window_base+window_size && opt_lsv.read_refine_state()==0)
    {
      for(int i=0; i<window_size; i++)
      {
        opt_lsv.so3_poses[i].setQuaternion(q_buf[window_base + i]);
        opt_lsv.t_poses[i] = t_buf[window_base + i];
      }

      // Do not optimize first sliding window 
      if(window_base == 0)
      {
        opt_lsv.set_refine_state(2);
      }
      else
      {
        // Push voxel map into optimizer
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
        
        // Begin iterative optimization
        // You can use multithreading or not.
        // We do not recommend use multiple thread on computer with poor performance

        // multithreading
        // map_refine_thread = new thread(&LM_SLWD_VOXEL::damping_iter, &opt_lsv);
        // map_refine_thread->detach();
        
        // non multithreading
        opt_lsv.damping_iter();
      }
    }

    // pose prediction
    t_curr = t_curr + q_curr * delta_t;
    q_curr = q_curr * delta_q;    
  }
}


