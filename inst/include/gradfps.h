#ifndef GRADFPS_PKG_H
#define GRADFPS_PKG_H

#include <RcppEigen.h>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using MapMat = Eigen::Map<Matrix>;
using MapVec = Eigen::Map<Vector>;
using MapConstVec = Eigen::Map<const Vector>;
using RefVec = Eigen::Ref<Vector>;
using SpMat = Eigen::SparseMatrix<double>;
using MapSpMat = Eigen::Map<SpMat>;
using MapConstSpMat = Eigen::Map<const SpMat>;

#endif  // GRADFPS_PKG_H
