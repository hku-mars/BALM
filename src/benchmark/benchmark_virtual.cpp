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
int sufSize = 150;
int ptsSize = 30;

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

class BALM2
{
public:
  vector<PointCluster*> sig_vecs;
  vector<vector<PointCluster>*> plvecVoxels;
  vector<double> coeffs;
  int winSize;

  void accEvaluate2(const vector<IMUST> &xs, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    Hess.setZero(); JacT.setZero(); residual = 0;
    vector<PointCluster> sig_tran(winSize);
    const int kk = 0;

    PLV(3) viRiTuk(winSize);
    PLM(3) viRiTukukT(winSize);

    #define Matrix36 Eigen::Matrix<double, 3, 6>
    vector<Matrix36, Eigen::aligned_allocator<Matrix36>> Auk(winSize);
    Eigen::Matrix3d umumT;

    int gpsSize = plvecVoxels.size();
    for(int a=0; a<gpsSize; a++)
    {
      const vector<PointCluster> &sig_orig = *plvecVoxels[a];
      double coe = coeffs[a];

      PointCluster sig = *sig_vecs[a];
      for(int i=0; i<winSize; i++)
      if(sig_orig[i].N != 0)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }
      
      const Eigen::Vector3d &vBar = sig.v / sig.N;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      const Eigen::Vector3d &lmbd = saes.eigenvalues();
      const Eigen::Matrix3d &U = saes.eigenvectors();
      int NN = sig.N;
      
      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};

      const Eigen::Vector3d &uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for(int i=0; i<3; i++)
        if(i != kk)
          umumT += 2.0/(lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for(int i=0; i<winSize; i++)
      if(sig_orig[i].N != 0)
      {
        Eigen::Matrix3d Pi = sig_orig[i].P;
        Eigen::Vector3d vi = sig_orig[i].v;
        Eigen::Matrix3d Ri = xs[i].R;
        double ni = sig_orig[i].N;

        Eigen::Matrix3d vihat; vihat << SKEW_SYM_MATRX(vi);
        Eigen::Vector3d RiTuk = Ri.transpose() * uk;
        Eigen::Matrix3d RiTukhat; RiTukhat << SKEW_SYM_MATRX(RiTuk);

        Eigen::Vector3d PiRiTuk = Pi * RiTuk;
        viRiTuk[i] = vihat * RiTuk;
        viRiTukukT[i] = viRiTuk[i] * uk.transpose();
        
        Eigen::Vector3d ti_v = xs[i].p - vBar;
        double ukTti_v = uk.dot(ti_v);

        Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
        Eigen::Vector3d combo2 = Ri*vi + ni*ti_v;
        Auk[i].block<3, 3>(0, 0) = (Ri*Pi + ti_v*vi.transpose()) * RiTukhat - Ri*combo1;
        Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
        Auk[i] /= NN;

        const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
        JacT.block<6, 1>(6*i, 0) += coe * jjt;

        const Eigen::Matrix3d &HRt = 2.0/NN * (1.0-ni/NN) * viRiTukukT[i];
        Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
        Hb.block<3, 3>(0, 0) += 2.0/NN * (combo1 - RiTukhat*Pi) * RiTukhat - 2.0/NN/NN * viRiTuk[i] * viRiTuk[i].transpose() - 0.5*hat(jjt.block<3, 1>(0, 0));
        Hb.block<3, 3>(0, 3) += HRt;
        Hb.block<3, 3>(3, 0) += HRt.transpose();
        Hb.block<3, 3>(3, 3) += 2.0/NN * (ni - ni*ni/NN) * ukukT;

        Hess.block<6, 6>(6*i, 6*i) += coe * Hb;
      }
      
      for(int i=0; i<winSize-1; i++)
      if(sig_orig[i].N != 0)
      {
        double ni = sig_orig[i].N;
        for(int j=i+1; j<winSize; j++)
        if(sig_orig[j].N != 0)
        {
          double nj = sig_orig[j].N;
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
          Hb.block<3, 3>(0, 0) += -2.0/NN/NN * viRiTuk[i] * viRiTuk[j].transpose();
          Hb.block<3, 3>(0, 3) += -2.0*nj/NN/NN * viRiTukukT[i];
          Hb.block<3, 3>(3, 0) += -2.0*ni/NN/NN * viRiTukukT[j].transpose();
          Hb.block<3, 3>(3, 3) += -2.0*ni*nj/NN/NN * ukukT;

          Hess.block<6, 6>(6*i, 6*j) += coe * Hb;
        }
      }
      
      residual += coe * lmbd[kk];
    }

    for(int i=1; i<winSize; i++)
      for(int j=0; j<i; j++)
        Hess.block<6, 6>(6*i, 6*j) = Hess.block<6, 6>(6*j, 6*i).transpose();
    
  }

  void left_evaluate_acc2(const vector<IMUST> &xs, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    Hess.setZero(); JacT.setZero(); residual = 0;
    int win_size = winSize;
    int l = 0;
    PLM(4) T(win_size);
    for(int i=0; i<win_size; i++)
      T[i] << xs[i].R, xs[i].p, 0, 0, 0, 1;

    vector<PLM(4)*> Cs;
    for(int a=0; a<plvecVoxels.size(); a++)
    {
      const vector<PointCluster> &sig_orig = *plvecVoxels[a];
      PLM(4) *Co = new PLM(4)(win_size, Eigen::Matrix4d::Zero());
      for(int i=0; i<win_size; i++)
        Co->at(i) << sig_orig[i].P, sig_orig[i].v, sig_orig[i].v.transpose(), sig_orig[i].N;
      Cs.push_back(Co);
    }
    
    int gpsSize = plvecVoxels.size();
    for(int a=0; a<gpsSize; a++)
    {
      double coe = coeffs[a];
      Eigen::Matrix4d C;
      PointCluster sig = *sig_vecs[a];
      C << sig.P, sig.v, sig.v.transpose(), sig.N;

      vector<int> Ns(win_size, 0);

      PLM(4) &Co = *Cs[a];
      PLM(4) TC(win_size), TCT(win_size);
      for(int j=0; j<win_size; j++)
      if((int)Co[j](3, 3) > 0)
      {
        TC[j] = T[j] * Co[j];
        TCT[j] = TC[j] * T[j].transpose();
        C += TCT[j];

        Ns[j] = Co[j](3, 3);
      }

      double NN = C(3, 3);
      C = C / NN;
      Eigen::Vector3d v_bar = C.block<3, 1>(0, 3);

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(C.block<3, 3>(0, 0) - v_bar * v_bar.transpose());
      Eigen::Vector3d lmbd = saes.eigenvalues();
      Eigen::Matrix3d Uev = saes.eigenvectors();

      residual += coe * lmbd[l];
      
      Eigen::Vector3d u[3] = {Uev.col(0), Uev.col(1), Uev.col(2)};
      Eigen::Matrix<double, 6, 4> U[3];
      PLV(6) g_kl[3];
      for(int k=0; k<3; k++)
      {
        g_kl[k].resize(win_size);
        U[k].setZero();
        U[k].block<3, 3>(0, 0) = hat(-u[k]);
        U[k].block<3, 1>(3, 3) = u[k];
      }

      PLV(6) UlTCF(win_size, Eigen::Matrix<double, 6, 1>::Zero());

      Eigen::VectorXd JacT_iter(6*win_size);
      for(int i=0; i<win_size; i++)
      if(Ns[i] != 0)
      {
        Eigen::Matrix<double, 3, 4> temp = T[i].block<3, 4>(0, 0);
        temp.block<3, 1>(0, 3) -= v_bar;
        Eigen::Matrix<double, 4, 3> TC_TCFSp = TC[i] * temp.transpose();
        for(int k=0; k<3; k++)
        {
          Eigen::Matrix<double, 6, 1> g1, g2;
          g1 = U[k] * TC_TCFSp * u[l];
          g2 = U[l] * TC_TCFSp * u[k];

          g_kl[k][i] = (g1 + g2) / NN;
        }

        UlTCF[i] = (U[l] * TC[i]).block<6, 1>(0, 3);
        JacT.block<6, 1>(6*i, 0) += coe * g_kl[l][i];

        // Eigen::Matrix<double, 6, 6> Hb(2.0/NN * U[l] * TCT[i] * U[l].transpose());

        Eigen::Matrix<double, 6, 6> Ha(-2.0/NN/NN * UlTCF[i] * UlTCF[i].transpose());

        Eigen::Matrix3d Ell = 1.0/NN * hat(TC_TCFSp.block<3, 3>(0, 0) * u[l]) * hat(u[l]);
        Ha.block<3, 3>(0, 0) += Ell + Ell.transpose();

        for(int k=0; k<3; k++)
          if(k != l)
            Ha += 2.0/(lmbd[l] - lmbd[k]) * g_kl[k][i] * g_kl[k][i].transpose();
          
        Hess.block<6, 6>(6*i, 6*i) += coe * Ha;
      }

      for(int i=0; i<win_size; i++)
      if(Ns[i] != 0)
      {
        Eigen::Matrix<double, 6, 6> Hb = U[l] * TCT[i] * U[l].transpose();
        Hess.block<6, 6>(6*i, 6*i) += 2.0 / NN * coe * Hb;
      }

      for(int i=0; i<win_size-1; i++)
      if(Ns[i] != 0)
      {
        for(int j=i+1; j<win_size; j++)
        if(Ns[j] != 0)
        {
          Eigen::Matrix<double, 6, 6> Ha = -2.0/NN/NN * UlTCF[i] * UlTCF[j].transpose();

          for(int k=0; k<3; k++)
            if(k != l)
              Ha += 2.0/(lmbd[l] - lmbd[k]) * g_kl[k][i] * g_kl[k][j].transpose();

          Hess.block<6, 6>(6*i, 6*j) += coe * Ha;
        }
      }
    
    }
    
    for(int i=1; i<win_size; i++)
      for(int j=0; j<i; j++)
        Hess.block<6, 6>(6*i, 6*j) = Hess.block<6, 6>(6*j, 6*i).transpose();
    
    for(int i=0; i<Cs.size(); i++)
      delete Cs[i];
    Cs.clear();

  }

  void only_residual(const vector<IMUST> &xs, double &residual)
  {
    residual = 0;
    vector<PointCluster> sig_tran(winSize);
    int kk = 0; // The kk-th lambda value

    int gps_size = plvecVoxels.size();
    for(int a=0; a<gps_size; a++)
    {
      const vector<PointCluster> &sig_orig = *plvecVoxels[a];
      PointCluster sig = *sig_vecs[a];

      for(int i=0; i<winSize; i++)
      if(sig_orig[i].N != 0)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.cov());
      residual += coeffs[a] * saes.eigenvalues()[kk];
    }

  }

  double dampingIter(vector<IMUST> &x_stats, vector<pcl::PointCloud<PointType>::Ptr> &plSurfs)
  {
    for(int i=0; i<plSurfs.size(); i++)
      sig_vecs.push_back(new PointCluster());

    double u = 0.1, v = 2;
    winSize = x_stats.size();
    int jacLeng = winSize * 6;
    Eigen::MatrixXd D(jacLeng, jacLeng), Hess(jacLeng, jacLeng);
    Eigen::VectorXd JacT(jacLeng), dxi(jacLeng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp = x_stats;

    coeffs.resize(plSurfs.size(), winSize * ptsSize);
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

    left_evaluate_acc2(x_stats, Hess, JacT, residual1);

    double tt1 = ros::Time::now().toSec();
    for(int i=0; i<20; i++)
    {
      if(is_calc_hess)
      {
        // accEvaluate2(x_stats, Hess, JacT, residual1);
        left_evaluate_acc2(x_stats, Hess, JacT, residual1);
      }

      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      for(int j=0; j<winSize; j++)
      {
        // x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL*j, 0));
        // x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL*j+3, 0);

        Eigen::Matrix3d dR = Exp(dxi.block<3, 1>(DVEL*j, 0));
        x_stats_temp[j].R = dR * x_stats[j].R;
        x_stats_temp[j].p = dR * x_stats[j].p + dxi.block<3, 1>(DVEL*j+3, 0);
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
    }
    double tt2 = ros::Time::now().toSec();

    for(int i=0; i<sig_vecs.size(); i++)
      delete sig_vecs[i];
    sig_vecs.clear();

    for(int i=0; i<plvecVoxels.size(); i++)
      delete plvecVoxels[i];
    plvecVoxels.clear();
  
    for(int j=1; j<winSize; j++)
    {
      x_stats[j].p = x_stats[j].p - x_stats[j].R * x_stats[0].R.transpose() * x_stats[0].p;
      x_stats[j].R = x_stats[j].R * x_stats[0].R.transpose();
    }

    x_stats[0].R.setIdentity();
    x_stats[0].p.setZero();

    return tt2 - tt1;
  }

};

void method_test(vector<IMUST> &xBuf, vector<pcl::PointCloud<PointType>::Ptr> &plSurfs, default_random_engine &e, int tseed, int full_flag)
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

  sleep(2);
  data_show(xBuf, plSurfs);
  printf("Display the point cloud and trajectory with noises.\n");
  printf("Input '1' to continue...\n");
  int a; cin >> a; if(a==0) exit(0);

  vector<IMUST> xBuf2 = xBuf;
  xBuf2 = xBuf;
  BALM2 bm;
  time_cost = bm.dampingIter(xBuf2, plSurfs);
  rsme(xBuf2, rsme_rot, rsme_tran);
  printf("RSME: %lfdeg, %lfm\n", rsme_rot*57.3, rsme_tran);
  printf("time: %lfs\n", time_cost);
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
  int full_flag = 0;
  double surf_range = 2.0;
  n.param<int>("winSize", winSize, 20);
  n.param<int>("sufSize", sufSize, 150);
  n.param<int>("ptsSize", ptsSize, 40);
  n.param<int>("full_flag", full_flag, 0);
  n.param<double>("point_noise", point_noise, 0.05);
  n.param<double>("surf_range", surf_range, 2.0);
  printf("winSize: %d\n", winSize);
  printf("sufSize: %d\n", sufSize);
  printf("pstSize: %d\n", ptsSize);
  
  int tseed = time(0);
  // int tseed = 1662299903;
  
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

  method_test(xBuf, plSurfs, e, tseed, full_flag);
  ros::spin();
}


