#ifndef GRADFPS_PROX_FANTOPE_H
#define GRADFPS_PROX_FANTOPE_H

#include "common.h"
#include "quadprog.h"

// min  -lambda'x + 0.5 * ||x||^2
// s.t. 0 <= xi <= 1
//      x[1] + ... + x[p] = d

inline void quadprog_sol(const double* lambda, int p, int d, double* sol)
{
    MatrixXd Dmat = MatrixXd::Identity(p, p);
    VectorXd dvec(p);
    std::copy(lambda, lambda + p, dvec.data());

    int q = 2 * p + 1;
    MatrixXd Amat = MatrixXd::Zero(p, q);
    for(int i = 0; i < p; i++)
    {
        Amat(i, 0) = 1.0;
        Amat(i, i + 1) = 1.0;
        Amat(i, p + i + 1) = -1.0;
    }

    VectorXd bvec = VectorXd::Zero(q);
    bvec[0] = double(d);
    for(int i = 0; i < p; i++)
    {
        bvec[p + i + 1] = -1.0;
    }

    int meq = 1;

    VectorXd lagr(q), work(2 * p + (p * (p + 5)) / 2 + 2 * q + 1);
    Eigen::VectorXi iact(q);
    double crval;
    int nact;
    int iter[2];
    int ierr = 1;

    F77_CALL(qpgen2)
        (Dmat.data(), dvec.data(), &p, &p,
         sol, lagr.data(), &crval,
         Amat.data(), bvec.data(), &p, &q,
         &meq, iact.data(), &nact, iter,
         work.data(), &ierr);
}


#endif  // GRADFPS_PROX_FANTOPE_H
