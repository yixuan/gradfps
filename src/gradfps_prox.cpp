#include "common.h"
#include "prox_fantope.h"
#include "walltime.h"

using Rcpp::IntegerVector;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::List;

// res += soft_threshold(x, penalty)
inline void add_soft_threshold(const MatrixXd& x, double penalty, MatrixXd& res)
{
    const int n = x.rows();

    for(int j = 0; j < n; j++)
    {
        for(int i = 0; i < n; i++)
        {
            const double xij = x.coeff(i, j);
            if(xij > penalty)
            {
                res.coeffRef(i, j) += (xij - penalty);
            } else if(xij < -penalty) {
                res.coeffRef(i, j) += (xij + penalty);
            }
        }
    }
}

// [[Rcpp::export]]
List gradfps_prox(MapMat S, MapMat x0, int d, double lambda,
                  double lr, int maxiter, double eps_abs, double eps_rel,
                  bool verbose = true)
{
    // Dimension of the covariance matrix
    const int n = S.rows();
    const int p = S.cols();
    if(n != p)
        Rcpp::stop("S must be square");

    MatrixXd z1 = x0, z2 = x0, zdiff(p, p), newz1(p, p);
    SpMat l1(p, p);

    for(int i = 0; i < maxiter; i++)
    {
        if(i % 50 == 0)
            Rcpp::Rcout << "iter = " << i << std::endl;

        // zdiff <- (z2 - z1) / 2
        zdiff.noalias() = 0.5 * (z2 - z1);

        // z1 <- -zdiff + prox_fantope(z2)
        z2.noalias() += lr * S;
        MapConstMat z2m(z2.data(), z2.rows(), z2.cols());
        MapMat newz1m(newz1.data(), newz1.rows(), newz1.cols());
        prox_fantope_impl(z2m, d, 5 * d, 10, newz1m);
        newz1.noalias() -= zdiff;

        // l1 <- soft_threshold(z1, lr * lambda)
        // z2 <- zdiff + l1
        z2.swap(zdiff);
        add_soft_threshold(z1, lr * lambda, z2);

        z1.swap(newz1);
    }

    return List::create(
        Rcpp::Named("z1") = z1,
        Rcpp::Named("z2") = z2
    );
}
