#include "tools.hpp"
#include <ros/ros.h>
#include <Eigen/Eigenvalues>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/PoseArray.h>
#include <random>
#include <ctime>
#include "SE3/SE3.hpp"
using namespace std;

const double one_three = (1.0 / 3.0);
vector<IMUST> xBuf_gt;
int winSize = 20;
int sufSize = 20;
int ptsSize = 40;

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

ros::Publisher pub_test, pub_curr, pub_full, pub_pose;

bool iter_stop(const Eigen::VectorXd &dx, double thre = 1e-6, int win_size = 0)
{
  // int win_size = dx.rows() / 6;
  if(win_size == 0)
    win_size = dx.rows() / 6;

  double angErr = 0, tranErr = 0;
  for(int i=0; i<win_size; i++)
  {
    angErr += dx.block<3, 1>(6*i, 0).norm();
    tranErr += dx.block<3, 1>(6*i+3, 0).norm();
  }

  angErr /= win_size; tranErr /= win_size;

  return (angErr < thre) && (tranErr < thre);
}

void rsme(vector<IMUST> &xBuf_es, double &rot, double &tran)
{
  rot = 0; tran = 0;
  int win_size = xBuf_es.size();
  
  for(int i=0; i<win_size; i++)
  {
    rot += Log(xBuf_gt[i].R.transpose() * xBuf_es[i].R).squaredNorm();
    tran += (xBuf_es[i].p - xBuf_gt[i].p).squaredNorm();
  }

  rot = sqrt(rot / win_size);
  tran = sqrt(tran / win_size);
}

void data_show(const vector<IMUST> &xBuf, vector<pcl::PointCloud<PointType>::Ptr> &plSurfs)
{
  vector<IMUST> xBuf2 = xBuf;

  geometry_msgs::PoseArray parray;
  parray.header.frame_id = "camera_init";
  for(int i=0; i<xBuf2.size(); i++)
  {
    Eigen::Quaterniond q_curr(xBuf2[i].R);
    Eigen::Vector3d t_curr(xBuf2[i].p);
    geometry_msgs::Pose apose;
    apose.orientation.w = q_curr.w();
    apose.orientation.x = q_curr.x();
    apose.orientation.y = q_curr.y();
    apose.orientation.z = q_curr.z();
    apose.position.x = t_curr.x();
    apose.position.y = t_curr.y();
    apose.position.z = t_curr.z();
    parray.poses.push_back(apose);
  }
  pub_pose.publish(parray);
  
  pcl::PointCloud<PointType> pl_send;
  for(int i=0; i<plSurfs.size(); i++)
  {
    for(PointType ap: plSurfs[i]->points)
    {
      Eigen::Vector3d pvec(ap.x, ap.y, ap.z);
      int fn = ap.intensity;
      pvec = xBuf2[fn].R * pvec + xBuf2[fn].p;
      ap.x = pvec[0];
      ap.y = pvec[1];
      ap.z = pvec[2];
      pl_send.push_back(ap);
    }
  }

  pub_pl_func(pl_send, pub_test);
}

class EigenFactor
{
public:
  int winSize, plaSize;
  vector<PLM(4)*> plvecMat4s;

  vector<vector<PointCluster>*> plvecVoxels;

  void Calculate_jacobian(const vector<mrob::SE3> &se3Poses, Eigen::VectorXd &JacT, double &residual, double &allsize)
  {
    JacT.setZero(); residual = 0; allsize = 0;

    int gpsSize = plvecMat4s.size();
    for(int a=0; a<gpsSize; a++)
    {
      PLM(4) &S = *plvecMat4s[a];
      PLM(4) matrixQ_(winSize);
      Eigen::Matrix4d Qall;
      Qall.setZero();
      
      for(int i=0; i<winSize; i++)
      {
        matrixQ_[i] = se3Poses[i].T_ * S[i] * se3Poses[i].T_.transpose();
        Qall += matrixQ_[i];
      }

      Eigen::JacobiSVD<Eigen::Matrix4d> svd(Qall, Eigen::ComputeFullU);
      Eigen::Vector4d planeEstimation_ = svd.matrixU().col(3);
      double lambda_ = svd.singularValues()(3);
      
      Eigen::VectorXd jact(6*winSize);
      for(int i=0; i<winSize; i++)
      {
        allsize += S[i](3, 3);

        Eigen::Matrix4d dQ;
        Eigen::Matrix4d &Q = matrixQ_[i];

        dQ.setZero();
        dQ.row(1) << -Q.row(2);
        dQ.row(2) <<  Q.row(1);
        dQ += dQ.transpose().eval();
        jact(6*i) = planeEstimation_.dot(dQ*planeEstimation_);

        dQ.setZero();
        dQ.row(0) <<  Q.row(2);
        dQ.row(2) << -Q.row(0);
        dQ += dQ.transpose().eval();
        jact(6*i+1) = planeEstimation_.dot(dQ*planeEstimation_);

        dQ.setZero();
        dQ.row(0) << -Q.row(1);
        dQ.row(1) <<  Q.row(0);
        dQ += dQ.transpose().eval();
        jact(6*i+2) = planeEstimation_.dot(dQ*planeEstimation_);

        dQ.setZero();
        dQ.row(0) << Q.row(3);
        dQ += dQ.transpose().eval();
        jact(6*i+3) = planeEstimation_.dot(dQ*planeEstimation_);

        dQ.setZero();
        dQ.row(1) << Q.row(3);
        dQ += dQ.transpose().eval();
        jact(6*i+4) = planeEstimation_.dot(dQ*planeEstimation_);

        dQ.setZero();
        dQ.row(2) << Q.row(3);
        dQ += dQ.transpose().eval();
        jact(6*i+5) = planeEstimation_.dot(dQ*planeEstimation_);
      }

      residual += lambda_;
      JacT += jact;
    }

  }

  void Residual(const vector<mrob::SE3> &se3Poses, double &residual)
  {
    residual = 0;
    int gpsSize = plvecMat4s.size();
    for(int a=0; a<gpsSize; a++)
    {
      PLM(4) &S = *plvecMat4s[a];
      PLM(4) matrixQ_(winSize);
      Eigen::Matrix4d Qall;
      Qall.setZero();
      
      for(int i=0; i<winSize; i++)
      {
        matrixQ_[i] = se3Poses[i].T_ * S[i] * se3Poses[i].T_.transpose();
        Qall += matrixQ_[i];
      }

      Eigen::JacobiSVD<Eigen::Matrix4d> svd(Qall, Eigen::ComputeFullU);
      Eigen::Vector4d planeEstimation_ = svd.matrixU().col(3);
      double lambda_ = svd.singularValues()(3);
      residual += lambda_;
    }

  }
  
  void gradient_evaluate(vector<IMUST> &x_stats, vector<pcl::PointCloud<PointType>::Ptr> &plSurfs)
  {
    // GRADIENT_DESCENT_NAIVE
    double alpha = 0.5;
    double alpha_init = alpha;
    winSize = x_stats.size();
    plaSize = plSurfs.size();

    for(pcl::PointCloud<PointType>::Ptr plPtr : plSurfs)
    {
      plvecVoxels.push_back(new vector<PointCluster>(winSize));
      vector<PointCluster> &plvecVoxel = *(plvecVoxels.back());

      for(PointType &ap : plPtr->points)
      {
        int fn = ap.intensity;
        Eigen::Vector3d pvec(ap.x, ap.y, ap.z);
        plvecVoxel[fn].push(pvec);
      }
    }

    for(pcl::PointCloud<PointType>::ConstPtr plPtr : plSurfs)
    {
      plvecMat4s.push_back(new PLM(4)(winSize));
      PLM(4) &plvecVoxel = *plvecMat4s.back();

      for(Eigen::Matrix4d &mat : plvecVoxel)
        mat.setZero();
      
      for(PointType ap : plPtr->points)
      {
        int fn = ap.intensity;
        Eigen::Vector4d pvec(ap.x, ap.y, ap.z, 1);
        plvecVoxel[fn] += pvec * pvec.transpose();
      }
    }

    int jacLeng = winSize * 6;
    Eigen::VectorXd JacT(jacLeng), dxi(jacLeng);

    double residual1, residual2, q;
    bool is_calc_hess = true;
    double res1, res2;
    int rcount = 0;

    vector<mrob::SE3> se3Poses(winSize), se3PosesTemp(winSize);

    for(int i=0; i<winSize; i++)
      se3Poses[i].T_ << x_stats[i].R, x_stats[i].p, 0, 0, 0, 1;
    se3PosesTemp = se3Poses;

    double tt1 = ros::Time::now().toSec();
    int count = 0;
    for(int i=0; i<1000; i++)
    {
      count++;
      double allsize = 0;
      Calculate_jacobian(se3Poses, JacT, residual1, allsize);

      dxi = -alpha/allsize * JacT;

      for(int j=0; j<winSize; j++)
      {
        se3PosesTemp[j] = se3Poses[j];
        se3PosesTemp[j].update(dxi.block<6, 1>(6*j, 0));
      }

      Residual(se3PosesTemp, residual2);

      printf("iter%d: (%lf %lf)\n", i, residual1, residual2);

      if(residual1 > residual2)
      {
        se3Poses = se3PosesTemp;
        alpha = alpha_init;
      }
      else
      {
        alpha *= 0.5;
      }

      if(iter_stop(dxi, 1e-5))
        break;

    }

    for(int i=0; i<winSize; i++)
    {
      x_stats[i].R = se3Poses[i].T_.block<3, 3>(0, 0);
      x_stats[i].p = se3Poses[i].T_.block<3, 1>(0, 3);
    }

    for(int j=1; j<winSize; j++)
    {
      x_stats[j].p = x_stats[0].R.transpose() * (x_stats[j].p - x_stats[0].p);
      x_stats[j].R = x_stats[0].R.transpose() * x_stats[j].R;
    }

    x_stats[0].R.setIdentity();
    x_stats[0].p.setZero();
  }

};

void method_test(vector<IMUST> &xBuf, vector<pcl::PointCloud<PointType>::Ptr> &plSurfs, default_random_engine &e, int tseed)
{
  double rsme_rot, rsme_tran, time_cost;
  xBuf_gt = xBuf;

  normal_distribution<double> randRot(0, 2 / 57.3);
  normal_distribution<double> randTra(0, 0.1);

  for(int i=0; i<xBuf.size(); i++)
  {
    Eigen::Vector3d rotvec(randRot(e), randRot(e), randRot(e));
    Eigen::Vector3d travec(randTra(e), randTra(e), randTra(e));

    rotvec /= 1.732; travec /= 1.732;

    xBuf[i].R = xBuf[i].R * Exp(rotvec);
    xBuf[i].p = xBuf[i].p + travec;
  }

  printf("\n************************\n");
  printf("EF test.\n");
  printf("************************\n\n");

  sleep(2);
  data_show(xBuf, plSurfs);
  printf("Display the point cloud and trajectory with noises.\n");
  printf("Input '1' to continue...\n");
  int a; cin >> a; if(a==0) exit(0);

  vector<IMUST> xBuf2 = xBuf;
  xBuf2 = xBuf;
  EigenFactor ef;
  ef.gradient_evaluate(xBuf2, plSurfs);
  rsme(xBuf2, rsme_rot, rsme_tran);
  printf("RSME: %lfdeg, %lfm\n", rsme_rot*57.3, rsme_tran);
  xBuf = xBuf2;

  data_show(xBuf, plSurfs);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "benchmark");
  ros::NodeHandle n;

  pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 100);
  pub_curr = n.advertise<sensor_msgs::PointCloud2>("/map_curr", 100);
  pub_full = n.advertise<sensor_msgs::PointCloud2>("/map_full", 100);
  pub_pose = n.advertise<geometry_msgs::PoseArray>("/poseAray", 10);

  double point_noise = 0.01;
  double surf_range = 2.0;
  
  int tseed = time(0);
  // int tseed = 10;
  default_random_engine e(tseed);
  uniform_real_distribution<double> randNorml(-M_PI, M_PI);
  uniform_real_distribution<double> randRange(-surf_range, surf_range);
  normal_distribution<double> randThick(0.0, point_noise);
  uniform_real_distribution<double> randVoxel(-0.5, 0.5);
  uniform_real_distribution<double> randpsize(3, 40);

  vector<pcl::PointCloud<PointType>::Ptr> plSurfs(sufSize);
  PLV(3) surfDirect(sufSize), surfCenter(sufSize);
  vector<IMUST> xBuf(winSize);

  normal_distribution<double> rand_traj(-1, 1);
  Eigen::Vector3d rotEnd(rand_traj(e), rand_traj(e), rand_traj(e));
  Eigen::Vector3d traEnd(rand_traj(e), rand_traj(e), rand_traj(e));
  rotEnd = rotEnd.normalized() * 0.5;
  traEnd = traEnd.normalized() * 1;

  for(int i=1; i<winSize; i++)
  {
    double ratio = 1.0*i/winSize;
    xBuf[i].R = Exp(ratio * rotEnd);
    xBuf[i].p = ratio * traEnd;
  }

  PointType ap;
  for(int i=0; i<sufSize; i++)
  {
    plSurfs[i].reset(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType> &plSurf = *plSurfs[i];
    
    Eigen::Matrix3d rot;
    if(i<3)
    {
      Eigen::Vector3d fd(0, 0, 0);
      fd[i] = M_PI_2;
      rot = Exp(fd);
    }
    else
      rot = Exp({randNorml(e), randNorml(e), randNorml(e)});

    surfDirect[i] = rot.col(2);
    surfCenter[i] << randRange(e), randRange(e), randRange(e);

    for(int j=0; j<winSize; j++)
    {
      ap.intensity = j;
      for(int k=0; k<ptsSize; k++)
      {
        Eigen::Vector3d pvec(randVoxel(e), randVoxel(e), randThick(e));
        pvec = rot * pvec + surfCenter[i];
        pvec = xBuf[j].R.transpose() * (pvec - xBuf[j].p);

        ap.x = pvec.x();
        ap.y = pvec.y();
        ap.z = pvec.z();
        plSurf.push_back(ap);
      }
    }
  }

  method_test(xBuf, plSurfs, e, tseed);
  ros::spin();
}


