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

// For two orthogonal matrices U and V, U'U = V'V = I_d,
// ||UU' - VV'||^2 = 2 * d - 2 * ||U'V||^2
inline double projection_diff(const MatrixXd& u, const MatrixXd& v)
{
    const int d = u.cols();
    MatrixXd uv(d, d);
    uv.noalias() = u.transpose() * v;
    return 2.0 * d - 2.0 * uv.squaredNorm();
}

// [[Rcpp::export]]
List gradfps_prox_(MapMat S, MapMat x0, int d, double lambda,
                   double lr = 0.001, int maxiter = 500,
                   int fan_maxinc = 100, int fan_maxiter = 10,
                   double eps_abs = 1e-3, double eps_rel = 1e-3,
                   int verbose = 0)
{
    // Dimension of the covariance matrix
    const int n = S.rows();
    const int p = S.cols();
    if(n != p)
        Rcpp::stop("S must be square");

    MatrixXd z1 = x0, z2 = x0, newz1(p, p), newz2(p, p);
    MatrixXd evecs(p, d), newevecs(p, d);

    // Metrics in each iteration
    std::vector<double> err_v;
    std::vector<double> err_z1;
    std::vector<double> err_z2;

    int fan_inc = 2 * d;
    double step = lr;

    int i = 0;
    for(i = 0; i < maxiter; i++)
    {
        if(verbose > 0)
            Rcpp::Rcout << "iter = " << i << std::endl;

        // z1 <- -zdiff + prox_fantope(z2)
        newz2.noalias() = z2 + step * S;
        double newdsum;
        fan_inc = prox_fantope_impl(newz2, d, fan_inc, fan_maxiter, newz1, newdsum,
                                    0.001 / std::sqrt(i + 1.0), verbose);
        newz2.noalias() = 0.5 * (z2 - z1);
        newz1.noalias() -= newz2;

        if(verbose > 1)
            Rcpp::Rcout << "fan_inc = " << fan_inc << std::endl;
        fan_inc = std::max(5 * d, int(1.5 * fan_inc));
        fan_inc = std::min(fan_inc, fan_maxinc);
        fan_inc = std::min(fan_inc, int(p / 10));

        // l1 <- soft_threshold(z1, lr * lambda)
        // z2 <- zdiff + l1
        add_soft_threshold(z1, step * lambda, newz2);

        err_z1.push_back((newz1 - z1).norm());
        err_z2.push_back((newz2 - z2).norm());

        z1.swap(newz1);
        z2.swap(newz2);

        // Reuse the memory of newz1
        MatrixXd& x = newz1;
        x.noalias() = 0.5 * (z1 + z2);
        Spectra::DenseSymMatProd<double> op(x);
        Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, d, 3 * d);
        eigs.init();
        eigs.compute();
        newevecs.noalias() = eigs.eigenvectors();

        if(i > 0)
        {
            const double diff = projection_diff(evecs, newevecs);
            err_v.push_back(diff);

            if(verbose > 0)
                Rcpp::Rcout << "  [info] err_v = " << diff
                            << ", err_z1 = " << err_z1.back()
                            << ", err_z2 = " << err_z2.back()
                            << std::endl << std::endl;

            if(diff < eps_abs || diff < eps_rel * d)
                break;
        }

        evecs.swap(newevecs);
    }

    // Reuse the memory of newz1
    MatrixXd& x = newz1;
    // x.noalias() = 0.5 * (z1 + z2);
    x.setZero();
    add_soft_threshold(z1, step * lambda, x);

    Spectra::DenseSymMatProd<double> op(x);
    Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, d, 3 * d);
    eigs.init();
    eigs.compute();

    return List::create(
        Rcpp::Named("projection") = newz1,
        Rcpp::Named("evecs") = eigs.eigenvectors(),
        Rcpp::Named("err_v") = err_v,
        Rcpp::Named("err_z1") = err_z1,
        Rcpp::Named("err_z2") = err_z2,
        Rcpp::Named("niter") = i + 1,
        Rcpp::Named("z1") = z1,
        Rcpp::Named("z2") = z2
    );
}

// [[Rcpp::export]]
List gradfps_prox_benchmark_(MapMat S, MapMat Pi, MapMat x0, int d, double lambda,
                             double lr = 0.001, int maxiter = 500,
                             int fan_maxinc = 100, int fan_maxiter = 10,
                             double eps_abs = 1e-3, double eps_rel = 1e-3,
                             int verbose = 0)
{
    // Dimension of the covariance matrix
    const int n = S.rows();
    const int p = S.cols();
    if(n != p)
        Rcpp::stop("S must be square");

    MatrixXd z1 = x0, z2 = x0, zdiff(p, p), newz1(p, p);

    // Metrics in each iteration
    std::vector<double> err_x;  // directly use X_hat
    std::vector<double> err_v;  // top d eigenvectors of X_hat
    std::vector<double> times;

    double t1, t2;
    int fan_inc = 2 * d;
    // double dsum = d;
    // bool lr_const = false;
    double step = lr;

    for(int i = 0; i < maxiter; i++)
    {
        // if(!lr_const)
        //     step = lr / std::sqrt(i + 1.0);

        if(verbose > 1 || (verbose > 0 && i % 50 == 0))
            Rcpp::Rcout << "\niter = " << i << ", alpha = " << step << std::endl;

        t1 = get_wall_time();

        // zdiff <- (z2 - z1) / 2
        zdiff.noalias() = 0.5 * (z2 - z1);

        // z1 <- -zdiff + prox_fantope(z2)
        z2.noalias() += step * S;
        double newdsum;
        fan_inc = prox_fantope_impl(z2, d, fan_inc, fan_maxiter, newz1, newdsum,
                                    0.001 / std::sqrt(i + 1.0), verbose);
        newz1.noalias() -= zdiff;

        /* if(newdsum > dsum)
        {
            lr_const = true;
            step = lr;
        } */

        if(verbose > 1)
            Rcpp::Rcout << "fan_inc = " << fan_inc << std::endl;
        fan_inc = std::max(5 * d, int(1.5 * fan_inc));
        fan_inc = std::min(fan_inc, fan_maxinc);
        fan_inc = std::min(fan_inc, int(p / 10));

        // l1 <- soft_threshold(z1, lr * lambda)
        // z2 <- zdiff + l1
        z2.swap(zdiff);
        add_soft_threshold(z1, step * lambda, z2);

        z1.swap(newz1);
        // Reuse the memory of zdiff
        MatrixXd& x = zdiff;
        // x.noalias() = 0.5 * (z1 + z2);
        x.setZero();
        add_soft_threshold(z1, step * lambda, x);

        Spectra::DenseSymMatProd<double> op(x);
        Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, d, 3 * d);
        eigs.init();
        eigs.compute();
        MatrixXd evecs = eigs.eigenvectors();

        t2 = get_wall_time();
        times.push_back(t2 - t1);
        err_x.push_back((x - Pi).norm());
        err_v.push_back((evecs * evecs.transpose() - Pi).norm());
    }

    return List::create(
        Rcpp::Named("projection") = zdiff,
        Rcpp::Named("err_x") = err_x,
        Rcpp::Named("err_v") = err_v,
        Rcpp::Named("times") = times,
        Rcpp::Named("z1") = z1,
        Rcpp::Named("z2") = z2
    );
}
