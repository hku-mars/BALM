#include "tools.hpp"
#include <ros/ros.h>
#include <Eigen/Eigenvalues>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/PoseArray.h>
#include <random>
#include <ctime>
#include <ceres/ceres.h>
using namespace std;

const double one_three = (1.0 / 3.0);
vector<IMUST> xBuf_gt;
int winSize = 10;
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

class PACeresFactor: public ceres::SizedCostFunction<4, 3, 3, 3>
{
public:
  Eigen::Matrix4d Gmat;

  PACeresFactor(Eigen::Matrix4d &mat): Gmat(mat){}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const 
  {
    double tt1 = ros::Time::now().toSec();

    Eigen::Vector3d rot_vec(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d pos(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Matrix3d rot = Exp(rot_vec);
    Eigen::Vector3d piFeature(parameters[2][0], parameters[2][1], parameters[2][2]);

    double d = piFeature.norm();
    Eigen::Vector3d n = piFeature / d;

    Eigen::Map<Eigen::Vector4d> residual(residuals);
    residual.block<3, 1>(0, 0) = rot.transpose() * n;
    residual(3, 0) = pos.dot(n) + d;
    residual = Gmat * residual;

    if(jacobians)
    {
      Eigen::Matrix<double, 4, 3> jac;
      if(jacobians[0])
      {
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> jac_R(jacobians[0]);
        jac.setZero();
        jac.block<3, 3>(0, 0) = hat(rot.transpose() * n);
        jac_R = Gmat * jac;
      }

      if(jacobians[1])
      {
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> jac_p(jacobians[1]);
        jac.setZero();
        jac.block<1, 3>(3, 0) = n.transpose();
        jac_p = Gmat * jac;
      }

      if(jacobians[2])
      {
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> jac_c(jacobians[2]);
        jac.setZero();
        jac.block<3, 3>(0, 0) = (rot.transpose() - n*n.transpose()) / d;
        jac.block<1, 3>(3, 0) = (pos.transpose() - pos.dot(n)*n.transpose()) / d + n.transpose();
        jac_c = Gmat * jac;
      }

    }

    return true;
  }

};

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

class PlaneAdjustmentCeres
{
public:
  void ceres_iteration(vector<IMUST> &x_stats, vector<pcl::PointCloud<PointType>::Ptr> &plSurfs)
  {
    PLV(3) rot_params, pos_params, pla_params;
    vector<PLM(4)*> plvecMat4s;

    int win_size = x_stats.size();
    for(int i=0; i<win_size; i++)
    {
      rot_params.push_back(Log(x_stats[i].R));
      pos_params.push_back(x_stats[i].p);
    }

    for(pcl::PointCloud<PointType>::Ptr plPtr : plSurfs)
    {
      plvecMat4s.push_back(new PLM(4)(winSize));
      PLM(4) &plvecVoxel = *plvecMat4s.back();

      for(Eigen::Matrix4d &mat : plvecVoxel)
        mat.setZero();
      
      PointCluster vf;
      for(PointType &ap : plPtr->points)
      {
        int fn = ap.intensity;
        Eigen::Vector4d pvec(ap.x, ap.y, ap.z, 1);
        plvecVoxel[fn] += pvec * pvec.transpose();
        vf.push(x_stats[fn].R * pvec.head(3) + x_stats[fn].p);
      }

      for(Eigen::Matrix4d &mat : plvecVoxel)
      {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> saes(mat);
        Eigen::Vector4d evalue = saes.eigenvalues();
        Eigen::Matrix4d mleft = saes.eigenvectors();
        Eigen::Matrix4d mvalue; mvalue.setZero();
        for(int i=0; i<4; i++)
        {
          if(evalue[i] > 0)
            mvalue(i, i) = sqrt(evalue[i]);
        }
        mat = (mleft * mvalue).transpose();
        // mat = mat.llt().matrixL().transpose();
      }
        
      Eigen::Vector3d center = vf.v / vf.N;
      Eigen::Matrix3d covMat = vf.P / vf.N - center * center.transpose();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
      Eigen::Vector3d pi = saes.eigenvectors().col(0);
      double d = - pi.dot(center);
      pla_params.push_back(d * pi);
    }

    ceres::Problem problem;
    for(int i=0; i<win_size; i++)
    {
      ceres::LocalParameterization *parametrization = new ParamSO3();
      problem.AddParameterBlock(rot_params[i].data(), 3, parametrization);
      problem.AddParameterBlock(pos_params[i].data(), 3);
    }
    // problem.SetParameterBlockConstant(rot_params[0].data());
    // problem.SetParameterBlockConstant(pos_params[0].data());

    for(int i=0; i<plvecMat4s.size(); i++)
    {
      PLM(4) &plvecVoxel = *plvecMat4s[i];
      for(int j=0; j<win_size; j++)
      {
        PACeresFactor *f = new PACeresFactor(plvecVoxel[j]);
        problem.AddResidualBlock(f, NULL, rot_params[j].data(), pos_params[j].data(), pla_params[i].data());
      }
    }

    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 1000;
    options.minimizer_progress_to_stdout = true;

    options.function_tolerance = 1e-10;
    options.parameter_tolerance = 1e-10;

    options.use_inner_iterations = true;
    ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering;
    for(int i=0; i<win_size; i++)
    {
      ordering->AddElementToGroup(rot_params[i].data(), 0);
      ordering->AddElementToGroup(pos_params[i].data(), 1);
    }
    for(int i=0; i<pla_params.size(); i++)
    {
      ordering->AddElementToGroup(pla_params[i].data(), 2);
    }
    options.inner_iteration_ordering.reset(ordering);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    for(int i=0; i<win_size; i++)
    {
      x_stats[i].R = Exp(rot_params[i]);
      x_stats[i].p = pos_params[i];
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
  printf("Plane Adjustment (PA) test.\n");
  printf("************************\n\n");

  sleep(2);
  data_show(xBuf, plSurfs);
  printf("Display the point cloud and trajectory with noises.\n");
  printf("Input '1' to continue...\n");
  int a; cin >> a; if(a==0) exit(0);

  vector<IMUST> xBuf2 = xBuf;
  xBuf2 = xBuf;
  PlaneAdjustmentCeres padc;
  padc.ceres_iteration(xBuf2, plSurfs);
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


