#ifndef FASTFPS_PKG_H
#define FASTFPS_PKG_H

#include <RcppEigen.h>

typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::Map<MatrixXd> MapMat;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Map<SpMat> MapSpMat;

#endif  // FASTFPS_PKG_H
