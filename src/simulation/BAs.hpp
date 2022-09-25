#ifndef BAs_HPP
#define BAs_HPP

#include "tools.hpp"
#include <thread>
#include <Eigen/Eigenvalues>
#include "preintegration2.hpp"
#include <fstream>
// #include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <ros/ros.h>
#include <random>

int win_size = 30;
int fix_size = 1;

const double one_three = (1.0 / 3.0);

int layer_limit = 0;
int layer_size[3] = {30, 25, 20};
// float eigen_value_array[3] = {0.0004, 0.0004, 0.0004};
float eigen_value_array[3] = {1.0/64, 1.0/64, 1.0/64};
// float eigen_value_array[3] = {1.0/225, 1.0/225, 1.0/225};
int min_ps = 10;

double voxel_size = 1;
int life_span = 1000;
int thd_num = 4;

class VOX_HESS
{
public:
  vector<const VOX_FACTOR*> sig_vecs;
  vector<const vector<VOX_FACTOR>*> plvec_voxels;
  vector<double> coeffs;

  void push_voxel(vector<VOX_FACTOR> *vec_orig, const VOX_FACTOR *fix, double feat_eigen, int layer)
  {
    int process_size = 0;
    for(int i=0; i<win_size; i++)
      if((*vec_orig)[i].N != 0)
        process_size++;

    if(process_size < 3) return; // 改

    double coe = 1 - feat_eigen/eigen_value_array[layer];
    coe = coe * coe;
    coe = 1;

    plvec_voxels.push_back(vec_orig);
    sig_vecs.push_back(fix);
    coeffs.push_back(coe);
  }

  void acc_evaluate(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    Hess.setZero(); JacT.setZero(); residual = 0;
    vector<VOX_FACTOR> sig_tran(win_size);
    const int kk = 0;

    PLV(3) viRiTuk(win_size);
    PLM(3) viRiTukukT(win_size);

    vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    for(int a=head; a<end; a++)
    {
      const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[a];
      // double coe = coeffs[a];

      VOX_FACTOR sig = *sig_vecs[a];
      for(int i=0; i<win_size; i++)
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

      for(int i=0; i<win_size; i++)
      // for(int i=1; i<win_size; i++)
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
        JacT.block<6, 1>(6*i, 0) += jjt;

        const Eigen::Matrix3d &HRt = 2.0/NN * (1.0-ni/NN) * viRiTukukT[i];
        Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
        Hb.block<3, 3>(0, 0) += 2.0/NN * (combo1 - RiTukhat*Pi) * RiTukhat - 2.0/NN/NN * viRiTuk[i] * viRiTuk[i].transpose() - 0.5*hat(jjt.block<3, 1>(0, 0));
        Hb.block<3, 3>(0, 3) += HRt;
        Hb.block<3, 3>(3, 0) += HRt.transpose();
        Hb.block<3, 3>(3, 3) += 2.0/NN * (ni - ni*ni/NN) * ukukT;

        Hess.block<6, 6>(6*i, 6*i) += Hb;
      }
      
      for(int i=0; i<win_size-1; i++)
      // for(int i=1; i<win_size-1; i++)
      if(sig_orig[i].N != 0)
      {
        double ni = sig_orig[i].N;
        for(int j=i+1; j<win_size; j++)
        if(sig_orig[j].N != 0)
        {
          double nj = sig_orig[j].N;
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
          Hb.block<3, 3>(0, 0) += -2.0/NN/NN * viRiTuk[i] * viRiTuk[j].transpose();
          Hb.block<3, 3>(0, 3) += -2.0*nj/NN/NN * viRiTukukT[i];
          Hb.block<3, 3>(3, 0) += -2.0*ni/NN/NN * viRiTukukT[j].transpose();
          Hb.block<3, 3>(3, 3) += -2.0*ni*nj/NN/NN * ukukT;

          Hess.block<6, 6>(6*i, 6*j) += Hb;
        }
      }
      
      residual += lmbd[kk];
    }

    for(int i=1; i<win_size; i++)
      for(int j=0; j<i; j++)
        Hess.block<6, 6>(6*i, 6*j) = Hess.block<6, 6>(6*j, 6*i).transpose();
    
  }

  void evaluate_only_residual(const vector<IMUST> &xs, double &residual)
  {
    residual = 0;
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value

    int gps_size = plvec_voxels.size();

    for(int a=0; a<gps_size; a++)
    {
      const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[a];
      VOX_FACTOR sig = *sig_vecs[a];

      for(int i=0; i<win_size; i++)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P/sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residual += coeffs[a] * lmbd[kk];
    }

  }

  void give_second(const vector<IMUST> &xs, int beg, int end, Eigen::MatrixXd &Rcov)
  {
    Rcov.setZero();

    vector<VOX_FACTOR> sig_tran(win_size);
    const int kk = 0;

    PLV(3) viRiTuk(win_size);
    PLM(3) viRiTukukT(win_size);

    vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    for(int a=beg; a<end; a++)
    {
      if(a%5 == 0)
      {
        printf("\rprogess: %.2lf%%", 100.0 * (a-beg)/(end-beg));
        fflush(stdout);
      }

      const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[a];
      VOX_FACTOR sig = *sig_vecs[a];

      for(int i=0; i<win_size; i++)
      if(sig_orig[i].N != 0)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      Eigen::Vector3d lmbd = saes.eigenvalues();
      Eigen::Matrix3d U = saes.eigenvectors();
      int NN = sig.N;

      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};

      const Eigen::Vector3d &uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for(int i=0; i<3; i++)
        if(i != kk)
          umumT += 2.0/(lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      Eigen::MatrixXd Mi(6*win_size, 6*win_size);
      Eigen::MatrixXd Ki(6*win_size, 3*win_size);
      Mi.setZero(); Ki.setZero();

      vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> uk_rhos(win_size);
      PLM(3) uk_vs(win_size);

      for(int i=0; i<win_size; i++)
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


        Eigen::Matrix<double, 3, 6> PRuk_rho; PRuk_rho.setZero();
        PRuk_rho << RiTuk[0], RiTuk[1], RiTuk[2],        0,        0,        0,
                           0, RiTuk[0],        0, RiTuk[1], RiTuk[2],        0,
                           0,        0, RiTuk[0],        0, RiTuk[1], RiTuk[2];
        uk_rhos[i] = Ri * PRuk_rho / NN;
        Eigen::Matrix<double, 6, 6> Ma = Auk[i].transpose() * umumT * uk_rhos[i];
        Ma.block<3, 6>(0, 0) += -2.0/NN * RiTukhat * PRuk_rho;
        Mi.block<6, 6>(6*i, 6*i) = Ma;
        
        uk_vs[i] = (ukTti_v*I33 + ti_v * uk.transpose()) * Ri / NN;
        Eigen::Matrix<double, 6, 3> Ka = Auk[i].transpose() * umumT * uk_vs[i];
        Ka.block<3, 3>(0, 0) += -2.0/NN * (ukTti_v * RiTukhat + vihat * RiTuk * RiTuk.transpose()/NN);
        Ka.block<3, 3>(3, 0) += 2.0* (NN-ni) /NN/NN * uk * RiTuk.transpose();
        Ki.block<6, 3>(6*i, 3*i) = Ka;
      }

      for(int i=0; i<win_size-1; i++)
      if(sig_orig[i].N != 0)
      {
        double ni = sig_orig[i].N;
        Eigen::Matrix3d Ri = xs[i].R;

        for(int j=i+1; j<win_size; j++)
        if(sig_orig[j].N != 0)
        {
          double nj = sig_orig[j].N;
          Eigen::Matrix3d Rj = xs[j].R;

          Mi.block<6, 6>(6*i, 6*j) = Auk[i].transpose() * umumT * uk_rhos[j];
          Mi.block<6, 6>(6*j, 6*i) = Auk[j].transpose() * umumT * uk_rhos[i];

          Eigen::Matrix<double, 6, 3> Ka = Auk[i].transpose() * umumT * uk_vs[j];
          Ka.block<3, 3>(0, 0) += -2.0/NN/NN * viRiTukukT[i] * Rj;
          Ka.block<3, 3>(3, 0) += -2.0*ni/NN/NN * ukukT * Rj;
          Ki.block<6, 3>(6*i, 3*j) = Ka;

          Ka = Auk[j].transpose() * umumT * uk_vs[i];
          Ka.block<3, 3>(0, 0) += -2.0/NN/NN * viRiTukukT[j] * Ri;
          Ka.block<3, 3>(3, 0) += -2.0*nj/NN/NN * ukukT * Ri;
          Ki.block<6, 3>(6*j, 3*i) = Ka;
        }

      }

      Eigen::MatrixXd cov_BpB(6*win_size, 6*win_size);
      Eigen::MatrixXd cov_Bp(6*win_size, 3*win_size);
      Eigen::MatrixXd cov_p(3*win_size, 3*win_size);
      cov_BpB.setZero(); cov_Bp.setZero(); cov_p.setZero();

      for(int i=0; i<win_size; i++)
      if(sig_orig[i].N != 0)
      {
        cov_BpB.block<6, 6>(6*i, 6*i) = sig_orig[i].P_cov;
        cov_Bp.block<6, 3>(6*i, 3*i) = sig_orig[i].BP_cov;
        cov_p.block<3, 3>(3*i, 3*i) = sig_orig[i].v_cov;
      }

      Eigen::MatrixXd Rcov2 = Mi * cov_Bp.sparseView() * Ki.transpose();
      Rcov += Mi*cov_BpB.sparseView()*Mi.transpose() + Rcov2 + Rcov2.transpose() + Ki*cov_p.sparseView()*Ki.transpose();
    }

  }

};

class OCTO_TREE_NODE
{
public:
  int octo_state; // 0(unknown), 1(mid node), 2(plane)
  int push_state;
  int layer;
  vector<PLV(3)> vec_orig, vec_tran;
  vector<VOX_FACTOR> sig_orig, sig_tran;
  VOX_FACTOR fix_point;
  PLV(3) vec_fix;

  OCTO_TREE_NODE *leaves[8];
  float voxel_center[3];
  float quater_length;

  Eigen::Vector3d center, direct, value_vector; // temporal
  double decision, ref, max_dis;

  OCTO_TREE_NODE()
  {
    value_vector.setZero(); max_dis = 0;
    octo_state = 0; push_state = 0;
    vec_orig.resize(win_size+fix_size); vec_tran.resize(win_size+fix_size);
    sig_orig.resize(win_size+fix_size); sig_tran.resize(win_size+fix_size);
    for(int i=0; i<8; i++) leaves[i] = nullptr;
    ref = 255.0*rand()/(RAND_MAX + 1.0f);
    layer = 0;
  }

  bool judge_eigen(int win_count)
  {
    VOX_FACTOR covMat = fix_point;
    // VOX_FACTOR covMat; covMat.clear();
    for(int i=0; i<win_count; i++)
      covMat += sig_tran[i];
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
    value_vector = saes.eigenvalues();
    center = covMat.v / covMat.N;
    direct = saes.eigenvectors().col(0);

    max_dis = 0;
    for(int i=0; i<win_count; i++)
    {
      for(Eigen::Vector3d &pvec : vec_tran[i])
      {
        double dis = fabs(direct.dot(pvec - center));
        if(dis > max_dis) max_dis = dis;
      }
    }

    // decision = saes.eigenvalues()[0] / saes.eigenvalues()[2];
    decision = saes.eigenvalues()[0] / saes.eigenvalues()[1];
    // decision = saes.eigenvalues()[0];

    return decision < eigen_value_array[layer] && max_dis < 0.001 && saes.eigenvalues()[2] / saes.eigenvalues()[1] < 25.0 && value_vector[0] < 1e-10;
  }

  void cut_func(int ci)
  {
    PLV(3) &pvec_orig = vec_orig[ci];
    PLV(3) &pvec_tran = vec_tran[ci];

    uint a_size = pvec_tran.size();
    for(uint j=0; j<a_size; j++)
    {
      int xyz[3] = {0, 0, 0};
      for(uint k=0; k<3; k++)
        if(pvec_tran[j][k] > voxel_center[k])
          xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OCTO_TREE_NODE();
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
        leaves[leafnum]->layer = layer + 1;
      }

      leaves[leafnum]->vec_orig[ci].push_back(pvec_orig[j]);
      leaves[leafnum]->vec_tran[ci].push_back(pvec_tran[j]);
      
      if(leaves[leafnum]->octo_state != 1)
      {
        leaves[leafnum]->sig_orig[ci].push(pvec_orig[j]);
        leaves[leafnum]->sig_tran[ci].push(pvec_tran[j]);
      }
    }

    PLV(3)().swap(pvec_orig); PLV(3)().swap(pvec_tran);
  }

  void recut(int win_count)
  {
    if(octo_state != 1)
    {
      int point_size = fix_point.N;
      for(int i=0; i<win_count; i++)
        point_size += sig_orig[i].N;

      push_state = 0; 
      if(point_size <= min_ps)
        return;

      if(judge_eigen(win_count))
      {
        if(octo_state==0 && point_size>layer_size[layer])
          octo_state = 2;

        point_size -= fix_point.N;
        if(point_size > min_ps)
          push_state = 1; 
        return;
      }
      else if(layer == layer_limit)
      {
        octo_state = 2; return;
      }

      octo_state = 1; 
      vector<VOX_FACTOR>().swap(sig_orig);
      vector<VOX_FACTOR>().swap(sig_tran);
      for(int i=0; i<win_count; i++)
        cut_func(i);
    }
    else
      cut_func(win_count-1);
    
    for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
        leaves[i]->recut(win_count);
  }

  void to_margi(int mg_size, vector<IMUST> &x_poses, int win_count)
  {
    if(octo_state != 1)
    {
      if(!x_poses.empty())
      for(int i=0; i<win_count; i++)
      {
        sig_tran[i].transform(sig_orig[i], x_poses[i]);
        plvec_trans(vec_orig[i], vec_tran[i], x_poses[i]);
      }
          
      if(fix_point.N<30 && push_state==1)
      for(int i=0; i<mg_size; i++)
      {
        fix_point += sig_tran[i];
        vec_fix.insert(vec_fix.end(), vec_tran[i].begin(), vec_tran[i].end());
      }
          
      for(int i=mg_size; i<win_count; i++)
      {
        sig_orig[i-mg_size] = sig_orig[i];
        sig_tran[i-mg_size] = sig_tran[i];
        vec_orig[i-mg_size].swap(vec_orig[i]);
        vec_tran[i-mg_size].swap(vec_tran[i]);
      }
        
      for(int i=win_count-mg_size; i<win_count; i++)
      {
        sig_orig[i].clear(); sig_tran[i].clear();
        vec_orig[i].clear(); vec_tran[i].clear();
      }

    }
    else
      for(int i=0; i<8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->to_margi(mg_size, x_poses, win_count);

  }

  void tras_opt(VOX_HESS &vox_opt, int win_count)
  {
    if(octo_state != 1)
    {
      int points_size = 0;
      for(int i=0; i<win_count; i++)
        points_size += sig_orig[i].N;
      
      if(points_size < min_ps)
        return;

      if(push_state == 1)
        vox_opt.push_voxel(&sig_orig, &fix_point, decision, layer);
    }
    else
    {
      for(int i=0; i<8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt, win_count);
    }

  }

  ~OCTO_TREE_NODE()
  {
    for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
        delete leaves[i];
  }

  void tras_display(pcl::PointCloud<PointType> &pl_feat, int win_count)
  {
    PointType ap; ap.intensity = ref;

    if(octo_state != 1)
    {
      if(push_state != 1)
        return;

      judge_eigen(win_count);

      for(int i=0; i<win_count; i++)
      for(uint j=0; j<vec_tran[i].size(); j++)
      {
        Eigen::Vector3d &pvec = vec_tran[i][j];
        ap.x = pvec.x();
        ap.y = pvec.y();
        ap.z = pvec.z();
        ap.normal_x = sqrt(value_vector[1] / value_vector[0]);
        ap.normal_y = sqrt(value_vector[2] / value_vector[0]);
        ap.normal_z = value_vector[0];
        ap.curvature = direct.dot(pvec - center);
        pl_feat.push_back(ap);
      }

    }
    else
    {
      for(int i=0; i<8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_display(pl_feat, win_count);
    }

  }

  void tras_fix(pcl::PointCloud<PointType> &pl_feat, int win_count)
  {
    PointType ap; ap.intensity = ref;

    if(octo_state != 1)
    {
      if(push_state != 1)
        return;

      for(Eigen::Vector3d &pvec : vec_fix)
      {
        ap.x = pvec.x();
        ap.y = pvec.y();
        ap.z = pvec.z() + 5;
        pl_feat.push_back(ap);
      }
    }
    else
    {
      for(int i=0; i<8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_fix(pl_feat, win_count);
    }


  }

  void corrupt(default_random_engine &e, vector<IMUST> &x_poses, int win_count)
  {
    normal_distribution<double> range_rand(0.0, 0.02);
    normal_distribution<double> angle_rand(0.0, 0.01 / 57.3);

    for(int i=0; i<win_count; i++)
    {
      sig_orig[i].clear(); sig_tran[i].clear();
      int psize = vec_orig[i].size();
      for(int j=0; j<psize; j++)
      {
        vec_orig[i][j].x() += range_rand(e);
        vec_orig[i][j].y() += range_rand(e);
        vec_orig[i][j].z() += range_rand(e);
        
        // Eigen::Vector3d &pvec = vec_orig[i][j];
        // double range = pvec.norm();
        // pvec = pvec / range;
        // range += range_rand(e);
        // Eigen::Vector3d raxi(angle_rand(e), angle_rand(e), angle_rand(e));
        // pvec = Exp(raxi) * pvec;
        // pvec = range * pvec;

        vec_tran[i][j] = x_poses[i].R * vec_orig[i][j] + x_poses[i].p;
        sig_orig[i].push(vec_orig[i][j]);
        sig_tran[i].push(vec_tran[i][j]);
      }
    }

  }

};

class OCTO_TREE_ROOT: public OCTO_TREE_NODE
{
public:
  bool is2opt;
  int life;
  vector<int> each_num;

  OCTO_TREE_ROOT()
  {
    is2opt = true;
    life = life_span;
    each_num.resize(win_size+fix_size);
    for(int i=0; i<win_size+fix_size; i++) each_num[i] = 0;
  }

  void marginalize(int mg_size, vector<IMUST> &x_poses, int win_count)
  {
    to_margi(mg_size, x_poses, win_count);

    int left_size = 0;
    for(int i=mg_size; i<win_count; i++)
    {
      each_num[i-mg_size] = each_num[i];
      left_size += each_num[i-mg_size];
    }

    if(left_size == 0) is2opt = false;

    for(int i=win_count-mg_size; i<win_count; i++)
      each_num[i] = 0;
  }

};

double vel_coef = 0.1;
double imu_coef = 0.1;

class VOX_OPTIMIZER
{
public:
  VOX_OPTIMIZER(){}

  double divide_thread(vector<IMUST> &x_stats, VOX_HESS &voxhess, vector<IMUST> &x_ab, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num); 
    PLV(-1) jacobins(thd_num);

    for(int i=0; i<thd_num; i++)
    {
      hessians[i].resize(6*win_size, 6*win_size);
      jacobins[i].resize(6*win_size);
    }

    int tthd_num = thd_num;
    vector<double> resis(tthd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;

    vector<thread*> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    for(int i=0; i<tthd_num; i++)
      mthreads[i] = new thread(&VOX_HESS::acc_evaluate, &voxhess, x_stats, part*i, part*(i+1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    // Eigen::Matrix<double, DVEL, 1> rr;
    // Eigen::Matrix<double, DVEL, DVEL> joca, jocb;

    // for(int i=1; i<win_size; i++)
    // {
    //   joca.setZero(); jocb.setZero();

    //   Eigen::Matrix3d &Ra = x_stats[i-1].R;
    //   Eigen::Vector3d &pa = x_stats[i-1].p;
    //   Eigen::Matrix3d &Rb = x_stats[i].R;
    //   Eigen::Vector3d &pb = x_stats[i].p;
    //   Eigen::Matrix3d &Rab = x_ab[i].R;
    //   Eigen::Vector3d &pab = x_ab[i].p;

    //   Eigen::Matrix3d res_r = Rab.transpose() * Ra.transpose() * Rb;
    //   Eigen::Vector3d res_p = pb - pa;
    //   rr.setZero();
    //   rr.block<3, 1>(0, 0) = Log(res_r);
    //   rr.block<3, 1>(3, 0) = Ra.transpose() * res_p - pab;

    //   residual += rr.squaredNorm();

    //   Eigen::Matrix3d JR_inv = jr_inv(res_r);
    //   joca.block<3, 3>(0, 0) = -JR_inv * Rb.transpose() * Ra;
    //   jocb.block<3, 3>(0, 0) =  JR_inv;
    //   joca.block<3, 3>(3, 0) =  hat(Ra.transpose() * res_p);
    //   joca.block<3, 3>(3, 3) = -Ra.transpose();
    //   jocb.block<3, 3>(3, 3) =  Ra.transpose();

    //   Eigen::Matrix<double, 6, 12> joc;
    //   joc.block<6, 6>(0, 0) = joca;
    //   joc.block<6, 6>(0, 6) = jocb;

    //   Hess.block<12, 12>((i-1)*6, (i-1)*6) += joc.transpose() * joc;
    //   JacT.block<12, 1>((i-1)*6, 0) += joc.transpose() * rr;
    // }

    // rr.setZero(); joca.setIdentity();
    // Eigen::Matrix3d res_r = x_ab[0].R.transpose() * x_stats[0].R;
    // rr.block<3, 1>(0, 0) = Log(res_r);
    // rr.block<3, 1>(3, 0) = x_stats[0].p - x_ab[0].p;
    // joca.block<3, 3>(0, 0) = jr_inv(res_r);
    // Hess.block<DVEL, DVEL>(0, 0) += joca.transpose() * joca;
    // JacT.block<DVEL, 1>(0, 0) += joca.transpose() * rr;
    // residual += rr.squaredNorm();

    // Hess *= vel_coef;
    // JacT *= vel_coef;
    // residual *= (vel_coef * 0.5);

    for(int i=0; i<tthd_num; i++)
    {
      mthreads[i]->join();
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double only_residual(vector<IMUST> &x_stats, VOX_HESS &voxhess, vector<IMUST> &x_ab)
  {
    double residual1 = 0, residual2 = 0;

    // Eigen::Matrix<double, DVEL, 1> rr;
    // for(int i=1; i<win_size; i++)
    // {
    //   Eigen::Matrix3d &Ra = x_stats[i-1].R;
    //   Eigen::Vector3d &pa = x_stats[i-1].p;
    //   Eigen::Matrix3d &Rb = x_stats[i].R;
    //   Eigen::Vector3d &pb = x_stats[i].p;
    //   Eigen::Matrix3d &Rab = x_ab[i].R;
    //   Eigen::Vector3d &pab = x_ab[i].p;

    //   Eigen::Matrix3d res_r = Rab.transpose() * Ra.transpose() * Rb;
    //   Eigen::Vector3d res_p = pb - pa;
    //   rr.setZero();
    //   rr.block<3, 1>(0, 0) = Log(res_r);
    //   rr.block<3, 1>(3, 0) = Ra.transpose() * res_p - pab;

    //   residual1 += rr.squaredNorm();
    // }

    // rr.setZero();
    // Eigen::Matrix3d res_r = x_ab[0].R.transpose() * x_stats[0].R;
    // rr.block<3, 1>(0, 0) = Log(res_r);
    // rr.block<3, 1>(3, 0) = x_stats[0].p - x_ab[0].p;
    // residual1 += rr.squaredNorm();

    // residual1 *= (vel_coef * 0.5);

    voxhess.evaluate_only_residual(x_stats, residual2);
    return (residual1 + residual2);
  }

  void multi_second(vector<IMUST> &x_stats, Eigen::MatrixXd &Rcov, VOX_HESS &voxhess)
  {
    double tnum = 6;
    PLM(-1) Rcovs(tnum);
    for(int i=0; i<tnum; i++)
    {
      Rcovs[i].setZero(6*win_size, 6*win_size);
    }

    int g_size = voxhess.plvec_voxels.size();
    vector<thread*> mthreads(tnum);
    double part = 1.0 * g_size / tnum;
    for(int i=0; i<tnum; i++)
      mthreads[i] = new thread(&VOX_HESS::give_second, &voxhess, x_stats, part*i, part*(i+1), ref(Rcovs[i]));
    
    for(int i=0; i<tnum; i++)
    {
      mthreads[i]->join();
      Rcov += Rcovs[i];
      delete mthreads[i];
    }

    printf("\rprogess: 100.00%%\n");

  }

  void damping_iter(vector<IMUST> &x_stats, VOX_HESS &voxhess, Eigen::MatrixXd &Rcov, int covEnable = 1)
  {
    double u = 0.01, v = 2;
    Eigen::MatrixXd D(6*win_size, 6*win_size), Hess(6*win_size, 6*win_size);
    Eigen::VectorXd JacT(6*win_size), dxi(6*win_size);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp = x_stats;

    vector<IMUST> x_ab(win_size);
    x_ab[0] = x_stats[0];
    for(int i=1; i<win_size; i++)
    {
      x_ab[i].p = x_stats[i-1].R.transpose() * (x_stats[i].p - x_stats[i-1].p);
      x_ab[i].R = x_stats[i-1].R.transpose() * x_stats[i].R;
    }

    for(int i=0; i<10; i++)
    {
      if(is_calc_hess)
        residual1 = divide_thread(x_stats, voxhess, x_ab, Hess, JacT);
      
      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      for(int j=0; j<win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL*j+3, 0);
      }
      double q1 = 0.5*dxi.dot(u*D*dxi-JacT);

      residual2 = only_residual(x_stats_temp, voxhess, x_ab);

      q = (residual1-residual2);
      printf("iter%d: (%lf %lf) u: %lf v: %lf q: %lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

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

      if(fabs(residual1-residual2)<1e-9)  
        break;
    }

    divide_thread(x_stats, voxhess, x_ab, Hess, JacT);

    if(covEnable)
    {
      multi_second(x_stats, Rcov, voxhess);
      Eigen::MatrixXd hess_inv = Hess.inverse();
      Rcov = hess_inv * Rcov * hess_inv.transpose();
    }
    
  }

};

void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> &feat_map, pcl::PointCloud<PointType> &pl_feat, const IMUST &x_key, int fnum)
{
  float loc_xyz[3];
  for(PointType &p_c : pl_feat.points)
  {
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = x_key.R*pvec_orig + x_key.p;

    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = pvec_tran[j] / voxel_size;
      if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->vec_orig[fnum].push_back(pvec_orig);
      iter->second->vec_tran[fnum].push_back(pvec_tran);
      
      if(iter->second->octo_state != 1)
      {
        iter->second->sig_orig[fnum].push(pvec_orig);
        iter->second->sig_tran[fnum].push(pvec_tran);
      }

      iter->second->is2opt = true;
      iter->second->life = life_span;
      iter->second->each_num[fnum]++;
    }
    else
    {
      OCTO_TREE_ROOT *ot = new OCTO_TREE_ROOT();
      ot->vec_orig[fnum].push_back(pvec_orig);
      ot->vec_tran[fnum].push_back(pvec_tran);
      ot->sig_orig[fnum].push(pvec_orig);
      ot->sig_tran[fnum].push(pvec_tran);
      ot->each_num[fnum]++;

      ot->voxel_center[0] = (0.5+position.x) * voxel_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->layer = 0;
      feat_map[position] = ot;
    }

  }

}

#endif
