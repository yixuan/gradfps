#include "common.h"
#include "prox_fantope.h"
#include "prox_eigs.h"
#include "walltime.h"

using Rcpp::IntegerVector;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::List;

// z1 <- (z1 - z2) / 2 + prox_fantope(z2)
// Returns ||newz1 - z1||_F
inline double prox_update_z1(RefMat z1, const RefConstMat& z2, const RefConstMat& proxf)
{
    const int p = z1.rows();
    const int len = p * p;
    double* z1p = z1.data();
    const double* z2p = z2.data();
    const double* pfp = proxf.data();
    double diff = 0.0;

    #pragma omp simd aligned(z1p, z2p, pfp: 32)
    for(int i = 0; i < len; i++)
    {
        const double newz1 = 0.5 * (z1p[i] - z2p[i]) + pfp[i];
        diff += std::pow(newz1 - z1p[i], 2);
        z1p[i] = newz1;
    }
    return std::sqrt(diff);
}

// z2 <- (z2 - z1) / 2 + soft_threshold(z1, penalty)
//     = { (z1 + z2) / 2 - penalty, if z1 > penalty
//       { (z1 + z2) / 2 + penalty, if z1 < -penalty
//       { (z2 - z1) / 2          , otherwise
// Returns ||newz2 - z2||_F
inline double prox_update_z2(const RefConstMat& z1, RefMat z2, double penalty)
{
    const int p = z1.rows();
    const int len = p * p;
    const double* z1p = z1.data();
    double* z2p = z2.data();
    double diff = 0.0;

    #pragma omp simd aligned(z1p, z2p: 32)
    for(int i = 0; i < len; i++)
    {
        const double z1i = z1p[i];
        const double z2i = z2p[i];
        const double x = 0.5 * (z1i + z2i);
        const double newz2 = (z1i > penalty) ?
                             (x - penalty) :
                             ( (z1i < -penalty) ? (x + penalty) : 0.5 * (z2i - z1i) );
        diff += std::pow(newz2 - z2i, 2);
        z2p[i] = newz2;
    }
    return std::sqrt(diff);
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

// res = soft_threshold(x, penalty)
inline void soft_threshold(const MatrixXd& x, double penalty, MatrixXd& res)
{
    const int p = x.rows();
    const int len = p * p;
    const double* xp = x.data();
    double* rp = res.data();

    #pragma omp simd aligned(xp, rp: 32)
    for(int i = 0; i < len; i++)
    {
        rp[i] = (xp[i] > penalty) ?
                (xp[i] - penalty) :
                ( xp[i] < -penalty ? xp[i] + penalty : 0.0 );
    }
}

// [[Rcpp::export]]
List gradfps_prox_(MapMat S, MapMat x0, int d, double lambda,
                   double lr, double mu, double r1, double r2,
                   int maxiter = 500, int fan_maxinc = 100, int fan_maxiter = 10,
                   double eps_abs = 1e-3, double eps_rel = 1e-3,
                   int verbose = 0)
{
    // Dimension of the covariance matrix
    const int n = S.rows();
    const int p = S.cols();
    if(n != p)
        Rcpp::stop("S must be square");

    MatrixXd z1 = x0, z2 = x0, prox_in_out(p, p);
    MatrixXd evecs(p, d), newevecs(p, d);

    // Metrics in each iteration
    std::vector<double> err_v;
    std::vector<double> err_z1;
    std::vector<double> err_z2;

    int fan_inc = 2 * d;
    const double l1 = lr * mu * r1;
    const double l2 = lr * mu / std::sqrt(double(p));
    double step = lr;

    int i = 0;
    for(i = 0; i < maxiter; i++)
    {
        if(verbose > 0)
            Rcpp::Rcout << "iter = " << i << std::endl;

        // z1 <- (z1 - z2) / 2 + prox_fantope(z2)
        prox_in_out.noalias() = z2 + step * S;
        fan_inc = prox_fantope_impl(prox_in_out, l1, l2, d, fan_inc, fan_maxiter, prox_in_out,
                                    0.001 / std::sqrt(i + 1.0), verbose);
        const double diffz1 = prox_update_z1(z1, z2, prox_in_out);

        if(verbose > 1)
            Rcpp::Rcout << "fan_dim = " << fan_inc << std::endl;
        fan_inc = std::max(5 * d, int(1.5 * fan_inc));
        fan_inc = std::min(fan_inc, fan_maxinc);
        fan_inc = std::min(fan_inc, int(p / 10));

        // z2 <- (z2 - z1) / 2 + soft_threshold(z1, penalty)
        const double diffz2 = prox_update_z2(z1, z2, step * lambda);

        err_z1.push_back(diffz1);
        err_z2.push_back(diffz2);

        // Reuse the memory of prox_in_out
        MatrixXd& x = prox_in_out;
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

    // Reuse the memory of prox_in_out
    MatrixXd& x = prox_in_out;
    // x.noalias() = 0.5 * (z1 + z2);
    soft_threshold(z1, step * lambda, x);

    Spectra::DenseSymMatProd<double> op(x);
    Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, d, 3 * d);
    eigs.init();
    eigs.compute();

    return List::create(
        Rcpp::Named("projection") = prox_in_out,
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
                             double lr, double mu, double r1, double r2,
                             int maxiter = 500, int fan_maxinc = 100, int fan_maxiter = 10,
                             double eps_abs = 1e-3, double eps_rel = 1e-3,
                             int verbose = 0)
{
    // Dimension of the covariance matrix
    const int n = S.rows();
    const int p = S.cols();
    if(n != p)
        Rcpp::stop("S must be square");

    MatrixXd z1 = x0, z2 = x0, prox_in_out(p, p);

    // Metrics in each iteration
    std::vector<double> err_x;  // directly use X_hat
    std::vector<double> err_v;  // top d eigenvectors of X_hat
    std::vector<double> times;

    double t1, t2;
    int fan_inc = 2 * d;
    const double l1 = lr * mu * r1;
    const double l2 = lr * mu / std::sqrt(double(p));
    double step = lr;

    for(int i = 0; i < maxiter; i++)
    {
        if(verbose > 1 || (verbose > 0 && i % 50 == 0))
            Rcpp::Rcout << "\niter = " << i << ", alpha = " << step << std::endl;

        t1 = get_wall_time();

        // z1 <- (z1 - z2) / 2 + prox_fantope(z2)
        prox_in_out.noalias() = z2 + step * S;
        fan_inc = prox_fantope_impl(prox_in_out, l1, l2, d, fan_inc, fan_maxiter, prox_in_out,
                                    0.001 / std::sqrt(i + 1.0), verbose);
        prox_update_z1(z1, z2, prox_in_out);

        if(verbose > 1)
            Rcpp::Rcout << "fan_dim = " << fan_inc << std::endl;
        fan_inc = std::max(5 * d, int(1.5 * fan_inc));
        fan_inc = std::min(fan_inc, fan_maxinc);
        fan_inc = std::min(fan_inc, int(p / 10));

        // z2 <- (z2 - z1) / 2 + soft_threshold(z1, penalty)
        prox_update_z2(z1, z2, step * lambda);

        // Reuse the memory of prox_in_out
        MatrixXd& x = prox_in_out;
        // x.noalias() = 0.5 * (z1 + z2);
        soft_threshold(z1, step * lambda, x);

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
        Rcpp::Named("projection") = prox_in_out,
        Rcpp::Named("err_x") = err_x,
        Rcpp::Named("err_v") = err_v,
        Rcpp::Named("times") = times,
        Rcpp::Named("z1") = z1,
        Rcpp::Named("z2") = z2
    );
}



// z1 <- z1 - x + prox_f1(2 * x - z1)
//     = z1 - x + soft_threshold(2 * x - z1, penalty)
//     = { x - penalty, if 2 * x - z1 > penalty
//       { x + penalty, if 2 * x - z1 < -penalty
//       { z1 - x     , otherwise
// Returns ||newz1 - z1||_F
inline double prox2_update_z1(const RefConstMat& x, RefMat z1, double penalty)
{
    const int p = x.rows();
    const double* xptr = x.data();
    const double* xend = xptr + p * p;
    double* zptr = z1.data();
    double diff = 0.0;
    for(; xptr < xend; xptr++, zptr++)
    {
        const double test = 2.0 * (*xptr) - (*zptr);
        const double newz1 = (test > penalty) ?
                             (*xptr - penalty) :
                             ( (test < -penalty) ? (*xptr + penalty) : (*zptr - *xptr) );
        diff += std::pow(newz1 - *zptr, 2);
        *zptr = newz1;
    }
    return std::sqrt(diff);
}

// newx <- P_X(zbar + alpha * S), zbar = (z1 + z2 + z3) / 3, z2 = x + shift * I
// Before calling this function, x contains the old value
inline void prox2_update_x(
    const RefConstMat& z1, double z2_shift, const RefConstMat& z3,
    double alpha, const RefConstMat& S, int d, RefMat x
)
{
    x.noalias() = (z1 + x + z3) / 3.0 + alpha * S;
    x.diagonal().array() += z2_shift / 3.0;
    const double xnorm = x.norm();
    const double sqrtd = std::sqrt(double(d));
    if(xnorm > sqrtd)
        x *= (sqrtd / xnorm);
}

// [[Rcpp::export]]
List gradfps_prox2_(
    MapMat S, MapMat x0, int d, double lambda,
    double lr, double mu, double r1, double r2,
    int maxiter = 500, int fan_maxinc = 100, int fan_maxiter = 10,
    double eps_abs = 1e-3, double eps_rel = 1e-3, int verbose = 0
)
{
    // Dimension of the covariance matrix
    const int n = S.rows();
    const int p = S.cols();
    if(n != p)
        Rcpp::stop("S must be square");

    MatrixXd z1 = x0, z3 = x0, x = x0, newz3(p, p);
    double z2_tr = x0.trace();
    MatrixXd evecs(p, d), newevecs(p, d);

    // Metrics in each iteration
    std::vector<double> err_v;
    std::vector<double> err_z1;
    std::vector<double> err_z3;

    int i = 0;
    for(i = 0; i < maxiter; i++)
    {
        if(verbose > 0)
            Rcpp::Rcout << "iter = " << i << std::endl;

        // z1 <- z1 - x + prox_f1(2 * x - z1)
        //     = z1 - x + soft_threshold(2 * x - z1, 3 * lr * lambda)
        const double diffz1 = prox2_update_z1(x, z1, 3.0 * lr * lambda);

        // z2 <- z2 - x + prof_f2(2 * x - z2)
        // prox_f2(u) = u + min(beta, 1) * s * I,
        // where beta = penalty / sqrt(p) / |s|, s = (d - tr(u)) / p,
        // penalty = 3 * lr * mu
        // z2 <- x + min(beta, 1) * s * I
        // z2 only stores the trace
        const double trx = x.trace();
        const double tru = 2.0 * trx - z2_tr;
        const double s = (d - tru) / double(p);
        const double beta = 3.0 * lr * mu / std::sqrt(double(p)) / std::abs(s);
        const double z2_shift = std::min(beta, 1.0) * s;
        z2_tr = trx + z2_shift * double(p);

        // z3 <- z3 - x + prox_f3(2 * x - z3)
        // prox_f3(u) = u + e, where e is a low-rank matrix
        // z3 <- x + e
        newz3.noalias() = 2.0 * x - z3;
        prox_eigs_delta_impl(newz3, 3.0 * lr * mu * r1, 3.0 * lr * mu * r2, newz3,
                             10, 10, 10, 10, fan_maxiter);
        newz3.noalias() += x;
        const double diffz3 = (newz3 - z3).norm();
        z3.swap(newz3);

        err_z1.push_back(diffz1);
        err_z3.push_back(diffz3);

        // Update x
        prox2_update_x(z1, z2_shift, z3, lr, S, d, x);

        // Compute eigenvectors
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
                            << ", err_z3 = " << err_z3.back()
                            << std::endl << std::endl;

            if(diff < eps_abs || diff < eps_rel * d)
                break;
        }

        evecs.swap(newevecs);
    }

    // x = prox_f1(2 * x - z1)
    // Reuse the memory of newz3
    newz3.noalias() = 2.0 * x - z1;
    soft_threshold(newz3, 3.0 * lr * lambda, x);

    Spectra::DenseSymMatProd<double> op(x);
    Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, d, 3 * d);
    eigs.init();
    eigs.compute();

    return List::create(
        Rcpp::Named("projection") = newz3,
        Rcpp::Named("evecs") = eigs.eigenvectors(),
        Rcpp::Named("err_v") = err_v,
        Rcpp::Named("err_z1") = err_z1,
        Rcpp::Named("err_z3") = err_z3,
        Rcpp::Named("niter") = i + 1,
        Rcpp::Named("z1") = z1,
        Rcpp::Named("z3") = z3
    );
}
