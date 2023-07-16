#include "tools.hpp"
#include <ros/ros.h>
#include <Eigen/Eigenvalues>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/PoseArray.h>
#include <random>
#include <ctime>
#include <ceres/ceres.h>
#include "factors_pr.h"
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

struct ParamSO3 : public ceres::LocalParameterization 
{
  virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const 
  {
    Eigen::Map<const Eigen::Vector3d> tangent(x);
    Eigen::Map<const Eigen::Vector3d> drot(delta);
    Eigen::Map<Eigen::Vector3d> out(x_plus_delta);

    out = Log(Exp(tangent) * Exp(drot));

    return true;
  }

  virtual bool ComputeJacobian(const double *x, double *jacobian) const 
  {
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> j(jacobian);
    j.setIdentity();
    return true;
  }

  virtual int GlobalSize() const { return 3; };
  virtual int LocalSize() const { return 3; };
};

class BAREG_test
{
public:
  vector<vector<PointCluster>*> plvecVoxels;
  int winSize;
  vector<PLM(3)> Rsigma_ks;
  vector<PLV(3)> scalar_lbds, muks;
  PLV(3) normals, mus;

  void refine_normal(PLV(3) &rot_params, PLV(3) &trans_params)
  {
    int surf_size = plvecVoxels.size();
    vector<PointCluster> sig_tran(winSize);

    for(int a=0; a<surf_size; a++)
    {
      const vector<PointCluster> &sig_orig = *plvecVoxels[a];

      PointCluster sig;
      for(int i=0; i<winSize; i++)
      if(sig_orig[i].N != 0)
      {
        sig_tran[i].transform(sig_orig[i], Exp(rot_params[i]), trans_params[i]);
        sig += sig_tran[i];
      }

      mus[a] = sig.v / sig.N;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.cov());
      normals[a] = saes.eigenvectors().col(0);
    }

  }

  void ceres_init(vector<IMUST> &x_stats, vector<pcl::PointCloud<PointType>::Ptr> &plSurfs)
  {
    winSize = x_stats.size();
    int surf_size = plSurfs.size();
    Rsigma_ks.resize(surf_size);
    scalar_lbds.resize(surf_size);
    muks.resize(surf_size);
    normals.resize(surf_size); mus.resize(surf_size);

    for(int i=0; i<surf_size; i++)
    {
      pcl::PointCloud<PointType>::Ptr plPtr = plSurfs[i];
      plvecVoxels.push_back(new vector<PointCluster>(winSize));
      vector<PointCluster> &plvecVoxel = *(plvecVoxels.back());

      for(PointType &ap : plPtr->points)
      {
        int fn = ap.intensity;
        Eigen::Vector3d pvec(ap.x, ap.y, ap.z);
        plvecVoxel[fn].push(pvec);
      }
      
      Rsigma_ks[i].resize(winSize);
      scalar_lbds[i].resize(winSize);
      muks[i].resize(winSize);

      for(int j=0; j<winSize; j++)
      {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(plvecVoxel[j].cov());
        Rsigma_ks[i][j] = saes.eigenvectors();
        scalar_lbds[i][j] = saes.eigenvalues() * plvecVoxel[j].N;
        muks[i][j] = plvecVoxel[j].v / plvecVoxel[j].N;
      }
    }

    PLV(3) rot_params(winSize), trans_params(winSize);
    for(int i=0; i<winSize; i++)
    {
      rot_params[i] = Log(x_stats[i].R);
      trans_params[i] = x_stats[i].p;
    }
    PLV(3) last_rot = rot_params;
    PLV(3) last_tra = trans_params;
    Eigen::VectorXd dx(6*winSize);

    int iteration = 0;
    for(int iterCount=0; iterCount<100; iterCount++)
    {
      refine_normal(rot_params, trans_params);

      ceres::Problem problem;
      for(int i=0; i<winSize; i++)
      {
        ceres::LocalParameterization *parametrization = new ParamSO3();

        problem.AddParameterBlock(rot_params[i].data(), 3, parametrization);
        problem.AddParameterBlock(trans_params[i].data(), 3);
      }

      // problem.SetParameterBlockConstant(rot_params[0].data());
      // problem.SetParameterBlockConstant(trans_params[0].data());
      
      for(int a=0; a<surf_size; a++)
      {
        const vector<PointCluster> &sig_orig = *plvecVoxels[a];
        PLV(3) &muk = muks[a];
        PLV(3) &scalar_lbd = scalar_lbds[a];
        PLM(3) &Rsigma_k = Rsigma_ks[a];

        for(int i=0; i<winSize; i++)
        {
          double scalar = sqrt(sig_orig[i].N);
          ceres::CostFunction *ft = new EigenFactorTrans2(normals[a], mus[a], muk[i], scalar);
          problem.AddResidualBlock(ft, NULL, rot_params[i].data(), trans_params[i].data());

          double rs1 = sqrt(scalar_lbd[i][1]);
          double rs2 = sqrt(scalar_lbd[i][2]);

          ceres::CostFunction *fr1 = new EigenFactorRotAxis(normals[a], Rsigma_k[i].col(1), rs1);
          ceres::CostFunction *fr2 = new EigenFactorRotAxis(normals[a], Rsigma_k[i].col(2), rs2);
          
          problem.AddResidualBlock(fr1, NULL, rot_params[i].data());
          problem.AddResidualBlock(fr2, NULL, rot_params[i].data());
        }
      }

      ceres::Solver::Options options;
      options.use_nonmonotonic_steps = true;
      options.max_consecutive_nonmonotonic_steps = 5;
      options.update_state_every_iteration = true;
      // options.linear_solver_type = ceres::DENSE_SCHUR;
      options.linear_solver_type = ceres::SPARSE_SCHUR;
      options.minimizer_progress_to_stdout = false;
      options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
      options.max_num_iterations = 100;
      options.num_threads = 1;
      options.function_tolerance = 1e-10;
      options.parameter_tolerance = 1e-10;

      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);

      iteration += summary.num_successful_steps;

      for(int i=0; i<winSize; i++)
      {
        dx.block<3, 1>(6*i, 0) = Log(Exp(-last_rot[i]) * Exp(rot_params[i]));
        dx.block<3, 1>(6*i+3, 0) = trans_params[i] - last_tra[i];
      }
      if(iter_stop(dx, 1e-6))
        break;

      last_rot = rot_params;
      last_tra = trans_params;
    }

    printf("max iterations: %d\n", iteration);

    for(int i=0; i<winSize; i++)
    {
      x_stats[i].R = Exp(rot_params[i]);
      x_stats[i].p = trans_params[i];
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
  printf("BAREG test.\n");
  printf("************************\n\n");

  sleep(2);
  data_show(xBuf, plSurfs);
  printf("Display the point cloud and trajectory with noises.\n");
  printf("Input '1' to continue...\n");
  int a; cin >> a; if(a==0) exit(0);

  vector<IMUST> xBuf2 = xBuf;
  xBuf2 = xBuf;
  BAREG_test bareg;
  bareg.ceres_init(xBuf2, plSurfs);
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


