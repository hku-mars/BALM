#pragma once

#include <ceres/ceres.h>
#include <ros/ros.h>
#include <Eigen/Eigenvalues>
#include "tools.hpp"

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

