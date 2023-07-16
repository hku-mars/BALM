#pragma once

#include "tools.hpp"
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <fstream>

class EigenFactorTrans2 : public ceres::SizedCostFunction<1, 3, 3> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  EigenFactorTrans2(const Eigen::Vector3d nhat, const Eigen::Vector3d mu_tgt, const Eigen::Vector3d &mu, const double s) : nhat_(nhat), mu_tgt_(mu_tgt), mu_(mu), scalar_(s) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
  {
    Eigen::Map<const Eigen::Vector3d> rot_log(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> t(parameters[1]);
    const auto &rot = Exp(rot_log);

    // const Eigen::RowVector3d &nhat = info_map_->rot_mat.col(0).transpose();
    // const Eigen::Vector3d &mu_tgt = info_map_->mean;

    if (residuals) 
    {
      residuals[0] = scalar_ * nhat_.dot((rot * mu_ + t) - mu_tgt_);
      if (isnan(residuals[0])) 
      {
        printf("residual wrong\n");
        exit(0);
      }
    }

    if (jacobians) 
    {
      if (jacobians[0]) 
      {
        Eigen::Map<Eigen::RowVector3d> jacob(jacobians[0]);
        jacob = -nhat_.transpose() * rot.matrix() * hat(mu_);
        jacob *= scalar_;
      }

      if (jacobians[1]) 
      {
        Eigen::Map<Eigen::RowVector3d> jacob(jacobians[1]);
        jacob = nhat_.transpose() * scalar_;
      }
    }

    return true;
  }

public:
  const double scalar_ = 1.0;

  // const Gaussian3d *info_map_ = nullptr;
  const Eigen::Vector3d mu_tgt_;
  const Eigen::Vector3d nhat_;
  const Eigen::Vector3d mu_;
};

class EigenFactorRotAxis : public ceres::SizedCostFunction<1, 3> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  EigenFactorRotAxis(Eigen::Vector3d nhat, const Eigen::Vector3d &axis,const double s) : nhat_(nhat), axis_(axis), scalar_(s) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
  {
    Eigen::Map<const Eigen::Vector3d> rot_log(parameters[0]);
    const auto &rot = Exp(rot_log);

    // const Eigen::RowVector3d &nhat = info_map_->rot_mat.col(0).transpose();

    if (residuals) 
    {
      residuals[0] = nhat_.dot(rot * axis_) * scalar_;
      if (isnan(residuals[0])) 
      {
        printf("residual wrong\n");
        exit(0);
      }
    }

    if (jacobians) {
      if (jacobians[0]) {
        Eigen::Map<Eigen::RowVector3d> jacob(jacobians[0]);
        jacob = -scalar_ * (nhat_.transpose() * rot.matrix() * hat(axis_));
      }
    }

    return true;
  }

public:
  const double scalar_ = 1.0;
  // const Gaussian3d *info_map_ = nullptr;
  const Eigen::Vector3d nhat_;
  const Eigen::Vector3d axis_;
};


