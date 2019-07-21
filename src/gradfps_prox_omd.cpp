#include "common.h"
#include "prox_fantope.h"
#include "prox_lp.h"
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

// -tr(S*X) + lambda * ||X||_1 + delta * ||X||_r^2 / 2
// [[Rcpp::export]]
List gradfps_prox_omd(MapMat S, MapMat x0, int d, double lambda, double delta,
                      double lr = 0.001, int maxiter = 500, int maxinc = 100,
                      double eps_abs = 1e-3, double eps_rel = 1e-3,
                      int verbose = 0)
{
    // Dimension of the covariance matrix
    const int n = S.rows();
    const int p = S.cols();
    if(n != p)
        Rcpp::stop("S must be square");

    // L_r penalty
    const double r = 1.0 + 1.0 / (std::log(double(p)) - 1.0);

    MatrixXd z1 = x0, z2 = x0, z3 = x0, x(p, p), prox_in(p, p), prox_out(p, p);
    MatrixXd evecs(p, d), newevecs(p, d);

    // Metrics in each iteration
    std::vector<double> resid;

    int fandim = 2 * d;
    const double radius = std::sqrt(double(d));
    double step = lr;

    int i = 0;
    for(i = 0; i < maxiter; i++)
    {
        if(verbose > 1 || (verbose > 0 && i % 50 == 0))
            Rcpp::Rcout << "iter = " << i << std::endl;

        // x = (z1 + z2 + z3) / 3, projected to the L2 ball
        x.noalias() = (z1 + z2 + z3) / 3.0;
        const double xnorm = x.norm();
        if(xnorm > radius)
            x *= (radius / xnorm);

        // z1 <- z1 - x + prox_fantope(2 * x - z1)
        prox_in.noalias() = x + x - z1 + step * S;
        double newdsum;
        fandim = prox_fantope_impl(prox_in, d, fandim, 10, prox_out, newdsum,
                                   0.01 / std::sqrt(i + 1.0), verbose);
        z1.noalias() += (prox_out - x);

        if(verbose > 1)
            Rcpp::Rcout << "fandim = " << fandim << std::endl;
        fandim = std::max(5 * d, int(1.5 * fandim));
        fandim = std::min(fandim, maxinc);
        fandim = std::min(fandim, int(p / 10));

        // z3 <- z3 - x + prox_lp(2 * x - z3, lr * delta)
        prox_in.noalias() = x + x - z3;
        prox_lp_mat_impl(prox_in, r, step * delta, prox_out, 1e-6, 100);
        z3.noalias() += (prox_out - x);

        // l1 <- soft_threshold(2 * x - z2, lr * lambda)
        // z2 <- z2 - x + l1
        prox_in.noalias() = x + x - z2;
        prox_out.setZero();
        add_soft_threshold(prox_in, step * lambda, prox_out);
        z2.noalias() += (prox_out - x);

        // l1 is sparse, and also converges to the solution
        // l1 is now stored in prox_out

        Spectra::DenseSymMatProd<double> op(prox_out);
        Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, d, 3 * d);
        eigs.init();
        eigs.compute();
        newevecs.noalias() = eigs.eigenvectors();

        if(i > 0)
        {
            const double diff = projection_diff(evecs, newevecs);
            resid.push_back(diff);
            if(diff < eps_abs || diff < eps_rel * d)
                break;
        }

        evecs.swap(newevecs);
    }

    return List::create(
        Rcpp::Named("projection") = prox_out,
        Rcpp::Named("evecs") = evecs,
        Rcpp::Named("resid") = resid,
        Rcpp::Named("niter") = i + 1,
        Rcpp::Named("z1") = z1,
        Rcpp::Named("z2") = z2
    );
}
