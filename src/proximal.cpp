#include "common.h"
#include "quadprog.h"
#include "inceig.h"

using Rcpp::IntegerVector;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::List;

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

// [[Rcpp::export]]
MatrixXd prox_fantope(MapMat v, double alpha, MapMat S, int d, int inc, int max_try)
{
    MatrixXd mat = v + alpha * S;
    VectorXd new_evals(inc * max_try);

    IncrementalEig inceig;
    inceig.init(mat, inc * max_try, inc);

    for(int i = 0; i < max_try; i++)
    {
        Rcpp::Rcout << i << std::endl;
        inceig.compute_next();
        const VectorXd& evals = inceig.eigenvalues();
        quadprog_sol(evals.data(), inceig.num_computed(), d, new_evals.data());
        if(std::abs(new_evals[inceig.num_computed() - 1]) < 1e-6)
            break;
    }

    int pos = d;
    const int end = inceig.num_computed();
    for(; pos < end; pos++)
    {
        if(std::abs(new_evals[pos]) < 1e-6)
            break;
    }

    MatrixXd res = inceig.eigenvectors().leftCols(pos) *
        new_evals.head(pos).asDiagonal() *
        inceig.eigenvectors().leftCols(pos).transpose();

    return res;
}

