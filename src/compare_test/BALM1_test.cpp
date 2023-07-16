#include "tools.hpp"
#include <ros/ros.h>
#include <Eigen/Eigenvalues>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/PoseArray.h>
#include <random>
#include <ctime>
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

class BALM1
{
public:
  vector<PLV(3)*> plvec_voxels;
  vector<vector<int>*> slwd_nums;
  vector<double> coeffs;
  int win_size;

  vector<vector<PointCluster>*> plvecVoxels;

  void down_sample_order(pcl::PointCloud<PointType> &pl, int num)
  {
    int plsize = pl.size();
    if(plsize <= num) return;

    pcl::PointCloud<PointType> pl_down;
    int segsize = plsize / num;
    // for(int i=0; i<num; i++)
    // {
    //   Eigen::Vector3d pvec(0, 0, 0);
    //   for(int j=i*segsize; j<(i+1)*segsize; j++)
    //   {
    //     pvec += Eigen::Vector3d(pl[j].x, pl[j].y, pl[j].z);
    //   }
    //   pvec /= segsize;
    //   PointType ap;
    //   ap.x = pvec[0]; ap.y = pvec[1]; ap.z = pvec[2];
    //   pl_down.push_back(ap);
    // }

    for(int i=0; i<plsize; i+=segsize)
    {
      pl_down.push_back(pl[i]);
      // Eigen::Vector3d pvec(0, 0, 0);
      // for(int j=i; j<i+segsize; j++)
      // {
      //   pvec += Eigen::Vector3d(pl[j].x, pl[j].y, pl[j].z);
      // }
      // pvec /= segsize;
      // PointType ap;
      // ap.x = pvec[0]; ap.y = pvec[1]; ap.z = pvec[2];
      // pl_down.push_back(ap);
    }

    pl.swap(pl_down);
  }

  void acc_evaluate(const vector<IMUST> &x_ps, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    // int gpsSize = plvec_voxels.size();

    Hess.setZero(); JacT.setZero(); residual = 0;
    Eigen::MatrixXd _hess(Hess);
    Eigen::VectorXd _jact(JacT);

    // for(int a=0; a<gpsSize; a++)
    for(int a=head; a<end; a++)
    {
      // printf("\rprogess: %.2lf%%", 100.0 * (a-0)/(gpsSize-0));
      // fflush(stdout); 

      // VOX_FACTOR &sig_vec = sig_vecs[a];
      PLV(3) &plvec_voxel = *plvec_voxels[a];
      vector<int> &slwd_num = *slwd_nums[a];
      uint backnum = plvec_voxel.size();

      Eigen::Vector3d vec_tran;
      PLV(3) plvec_back(backnum);
      vector<Eigen::Matrix3d> point_xis(backnum);
      Eigen::Vector3d center(Eigen::Vector3d::Zero());
      Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());

      for(uint i=0; i<backnum; i++)
      {
        vec_tran = x_ps[slwd_num[i]].R * plvec_voxel[i];
        point_xis[i] = -hat(vec_tran);
        plvec_back[i] = vec_tran + x_ps[slwd_num[i]].p;

        center += plvec_back[i];
        covMat += plvec_back[i] * plvec_back[i].transpose();
      }

      // double N_points = backnum + sig_vec.N;
      // center += sig_vec.v;
      // covMat += sig_vec.P;

      double N_points = backnum;

      center /= N_points;
      covMat = covMat/N_points - center*center.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
      Eigen::Vector3d eigen_value = saes.eigenvalues();
      Eigen::Matrix3d U = saes.eigenvectors();
      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)}; // eigenvectors

      // uint k = lam_types[a] * 2;
      uint k = 0;
      Eigen::Matrix3d ukukT(u[k] * u[k].transpose());
      // if(k == 2)
      // {
      //   ukukT -= I33;
      // }
      Eigen::Vector3d vec_Jt;
      for(uint i=0; i<backnum; i++)
      {
        plvec_back[i] = plvec_back[i] - center;
        vec_Jt = 2.0/N_points * ukukT * plvec_back[i];
        _jact.block<3, 1>(DVEL*slwd_num[i]+3, 0) += vec_Jt;
        _jact.block<3, 1>(DVEL*slwd_num[i], 0) -= point_xis[i] * vec_Jt;
      }

      Eigen::Matrix3d Hessian33;
      Eigen::Matrix3d C_k;
      vector<Eigen::Matrix3d> C_k_np(3);
      for(uint i=0; i<3; i++)
      {
        if(i == k)
        {
          C_k_np[i].setZero();
          continue;
        }
        Hessian33 = u[i]*u[k].transpose();
        // part of F matrix in paper
        C_k_np[i] = -1.0/N_points/(eigen_value[i]-eigen_value[k])*(Hessian33 + Hessian33.transpose());
      }

      Eigen::Matrix3d h33;
      uint rownum, colnum;

      Eigen::Matrix3d ukukT_same = (N_points-1)/N_points * ukukT;
      Eigen::Matrix3d ukukT_diff = -1.0 / N_points * ukukT;

      for(uint j=0; j<backnum; j++)
      {
        for(int f=0; f<3; f++)
        {
          C_k.block<1, 3>(f, 0) = plvec_back[j].transpose() * C_k_np[f];
        }
        C_k = U * C_k;
        colnum = DVEL*slwd_num[j];
        // block matrix operation, half Hessian matrix
        for(uint i=j; i<backnum; i++)
        {
          Hessian33 = u[k]*(plvec_back[i]).transpose()*C_k + u[k].dot(plvec_back[i])*C_k;

          rownum = DVEL*slwd_num[i];
          if(i == j) // 改
          {
            // Hessian33 += (N_points-1)/N_points * ukukT;
            Hessian33 += ukukT_same;
          }
          else
          {
            // Hessian33 -= 1.0/N_points * ukukT;
            Hessian33 += ukukT_diff;
          }
          Hessian33 = 2.0/N_points * Hessian33; // Hessian matrix of lambda and point

          // Hessian matrix of lambda and pose
          if(rownum==colnum && i!=j)
          {
            _hess.block<3, 3>(rownum+3, colnum+3) += Hessian33 + Hessian33.transpose();

            h33 = -point_xis[i]*Hessian33;
            _hess.block<3, 3>(rownum, colnum+3) += h33;
            _hess.block<3, 3>(rownum+3, colnum) += h33.transpose();
            h33 = Hessian33*point_xis[j];
            _hess.block<3, 3>(rownum+3, colnum) += h33;
            _hess.block<3, 3>(rownum, colnum+3) += h33.transpose();
            h33 = -point_xis[i] * h33;
            _hess.block<3, 3>(rownum, colnum) += h33 + h33.transpose();
          }
          else
          {
            _hess.block<3, 3>(rownum+3, colnum+3) += Hessian33;
            h33 = Hessian33*point_xis[j];
            _hess.block<3, 3>(rownum+3, colnum) += h33;
            _hess.block<3, 3>(rownum, colnum+3) -= point_xis[i]*Hessian33;
            _hess.block<3, 3>(rownum, colnum) -= point_xis[i]*h33;
          } 
        }
      }

      residual += coeffs[a] * eigen_value[k];
      Hess += coeffs[a] * _hess; JacT += coeffs[a] * _jact;
      
      _hess.setZero(); _jact.setZero();
    }

  }

  void only_residual(vector<IMUST> &x_ps, double &residual)
  {
    int gps_size = plvec_voxels.size();
    residual = 0;

    for(uint a=0; a<gps_size; a++)
    {
      // VOX_FACTOR &sig_vec = sig_vecs[a];
      PLV(3) &plvec_voxel = *plvec_voxels[a];
      vector<int> &slwd_num = *slwd_nums[a];
      uint backnum = plvec_voxel.size();

      Eigen::Vector3d vec_tran;
      PLV(3) plvec_back(backnum);
      
      Eigen::Vector3d center(Eigen::Vector3d::Zero());
      Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());

      for(uint i=0; i<backnum; i++)
      {
        vec_tran = x_ps[slwd_num[i]].R * plvec_voxel[i];
        plvec_back[i] = vec_tran + x_ps[slwd_num[i]].p;

        center += plvec_back[i];
        covMat += plvec_back[i] * plvec_back[i].transpose();
      }

      // double N_points = backnum + sig_vec.N;
      // center += sig_vec.v;
      // covMat += sig_vec.P;

      double N_points = backnum;

      center /= N_points;
      covMat = covMat/N_points - center*center.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
      Eigen::Vector3d eigen_value = saes.eigenvalues();
      
      residual += coeffs[a] * eigen_value[0];
    }

  }

  void damping_iter(vector<IMUST> &x_stats, vector<pcl::PointCloud<PointType>::Ptr> &plSurfs)
  {
    win_size = x_stats.size();
    int surf_size = plSurfs.size();
    double coe = 1;

    for(pcl::PointCloud<PointType>::Ptr plPtr : plSurfs)
    {
      plvecVoxels.push_back(new vector<PointCluster>(win_size));
      vector<PointCluster> &plvecVoxel = *(plvecVoxels.back());

      for(PointType &ap : plPtr->points)
      {
        int fn = ap.intensity;
        Eigen::Vector3d pvec(ap.x, ap.y, ap.z);
        plvecVoxel[fn].push(pvec);
      }
    }

    vector<pcl::PointCloud<PointType>::Ptr> pl_buff(win_size);
    for(int i=0; i<win_size; i++)
      pl_buff[i].reset(new pcl::PointCloud<PointType>());

    for(pcl::PointCloud<PointType>::Ptr plPtr : plSurfs)
    {
      PLV(3) *plvec_voxel = new PLV(3)();
      vector<int> *slwd_num = new vector<int>();
      
      for(int i=0; i<win_size; i++)
        pl_buff[i]->clear();

      for(PointType &ap : plPtr->points)
      {
        int fn = ap.intensity;
        pl_buff[fn]->push_back(ap);
      }

      for(int i=0; i<win_size; i++)
      {
        // down_sampling_voxel(*pl_buff[i], 0.5);
        down_sample_order(*pl_buff[i], 5);
        for(PointType &ap : pl_buff[i]->points)
        {
          plvec_voxel->push_back({ap.x, ap.y, ap.z});
          slwd_num->push_back(i);
        }
      }
        
      plvec_voxels.push_back(plvec_voxel);
      slwd_nums.push_back(slwd_num);
      coeffs.push_back(coe);
    }

    double u = 0.1, v = 2;
    int jacLeng = win_size * 6;
    Eigen::MatrixXd D(jacLeng, jacLeng), Hess(jacLeng, jacLeng);
    Eigen::VectorXd JacT(jacLeng), dxi(jacLeng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp = x_stats;

    double tt1 = ros::Time::now().toSec();
    int max_iter = 10;

    for(int i=0; i<max_iter; i++)
    {
      acc_evaluate(x_stats, 0, plvec_voxels.size(), Hess, JacT, residual1);
      // divide_thread(x_stats, Hess, JacT, residual1);

      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      for(int j=0; j<win_size; j++)
      {
        x_stats_temp[j].R = Exp(dxi.block<3, 1>(DVEL*j, 0)) * x_stats[j].R;
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL*j+3, 0);
      }

      double q1 = 0.5*dxi.dot(u*D*dxi-JacT);
      only_residual(x_stats_temp, residual2);
      q = (residual1-residual2);

      printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.3lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

      if(q > 0)
      {
        x_stats = x_stats_temp;

        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;
      }

      if(iter_stop(dxi, 1e-6))
        break;

      // if(fabs(residual1-residual2)<1e-9)  
      //   break;

    }

    for(int i=0; i<plvec_voxels.size(); i++)
      delete plvec_voxels[i];
    plvec_voxels.clear();

    for(int i=0; i<slwd_nums.size(); i++)
      delete slwd_nums[i];
    slwd_nums.clear();

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
  printf("BALM1 test.\n");
  printf("************************\n\n");

  sleep(2);
  data_show(xBuf, plSurfs);
  printf("Display the point cloud and trajectory with noises.\n");
  printf("Input '1' to continue...\n");
  int a; cin >> a; if(a==0) exit(0);

  vector<IMUST> xBuf2 = xBuf;
  xBuf2 = xBuf;
  BALM1 bm;
  bm.damping_iter(xBuf2, plSurfs);
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


