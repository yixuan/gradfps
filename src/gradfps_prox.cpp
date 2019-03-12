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
                  double lr = 0.001, int maxiter = 500,
                  double eps_abs = 1e-3, double eps_rel = 1e-3,
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
        double dsum;
        prox_fantope_impl(z2m, d, 5 * d, 10, newz1m, dsum);
        newz1.noalias() -= zdiff;

        // l1 <- soft_threshold(z1, lr * lambda)
        // z2 <- zdiff + l1
        z2.swap(zdiff);
        add_soft_threshold(z1, lr * lambda, z2);

        z1.swap(newz1);
    }

    return List::create(
        Rcpp::Named("projection") = 0.5 * (z1 + z2),
        Rcpp::Named("z1") = z1,
        Rcpp::Named("z2") = z2
    );
}

// [[Rcpp::export]]
List gradfps_prox_benchmark(MapMat S, MapMat x0, MapMat Pi, int d, double lambda,
                  double lr = 0.001, int maxiter = 500,
                  double eps_abs = 1e-3, double eps_rel = 1e-3,
                  int verbose = 0)
{
    // Dimension of the covariance matrix
    const int n = S.rows();
    const int p = S.cols();
    if(n != p)
        Rcpp::stop("S must be square");

    MatrixXd z1 = x0, z2 = x0, zdiff(p, p), newz1(p, p);
    SpMat l1(p, p);

    // Metrics in each iteration
    std::vector<double> errs;
    std::vector<double> times;
    double t1, t2;
    int fandim = 2 * d;
    double dsum = d;
    bool lr_const = false;
    double step = lr;

    for(int i = 0; i < maxiter; i++)
    {
        if(!lr_const)
            step = lr / std::sqrt(i + 1.0);

        if(verbose > 1 || (verbose > 0 && i % 50 == 0))
            Rcpp::Rcout << "iter = " << i << std::endl;

        t1 = get_wall_time();

        // zdiff <- (z2 - z1) / 2
        zdiff.noalias() = 0.5 * (z2 - z1);

        // z1 <- -zdiff + prox_fantope(z2)
        z2.noalias() += step * S;
        MapConstMat z2m(z2.data(), z2.rows(), z2.cols());
        MapMat newz1m(newz1.data(), newz1.rows(), newz1.cols());
        double newdsum;
        fandim = prox_fantope_impl(z2m, d, fandim, 10, newz1m, newdsum,
                                   0.01 / std::sqrt(i + 1.0), verbose);

        if(newdsum > dsum)
        {
            lr_const = true;
            step = lr;
        }

        if(verbose > 1)
            Rcpp::Rcout << "fandim = " << fandim << std::endl;

        fandim = std::max(5 * d, int(1.5 * fandim));
        fandim = std::min(fandim, 50 * d);
        newz1.noalias() -= zdiff;

        // l1 <- soft_threshold(z1, lr * lambda)
        // z2 <- zdiff + l1
        z2.swap(zdiff);
        add_soft_threshold(z1, step * lambda, z2);

        z1.swap(newz1);

        t2 = get_wall_time();
        times.push_back(t2 - t1);
        errs.push_back((0.5 * (z1 + z2) - Pi).norm());
    }

    return List::create(
        Rcpp::Named("projection") = 0.5 * (z1 + z2),
        Rcpp::Named("errors") = errs,
        Rcpp::Named("times") = times,
        Rcpp::Named("z1") = z1,
        Rcpp::Named("z2") = z2
    );
}
