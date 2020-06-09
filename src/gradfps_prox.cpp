#include "common.h"
#include "prox_fantope.h"
#include "prox_eigs.h"
#include "walltime.h"
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>

using Rcpp::IntegerVector;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::List;

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
    const int len = x.size();
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

// Proximal-proximal-gradient optimizer
class PPGOptimizer
{
private:
    const int   m_p;      // Dimension of the matrix, p
    const int   m_pp;     // p^2
    const int   m_d;      // Number of eigenvectors to estimate
    RefConstMat m_S;      // Sample covariance matrix

    MatrixXd    m_z1;     // Auxiliary variable
    MatrixXd    m_z2;     // Auxiliary variable
    MatrixXd    m_work;   // Work space
    MatrixXd    m_evecs;  // Eigenvectors
    MatrixXd    m_ework;  // Work space for computing eigenvectors

    int m_fan_inc;        // Parameter for computing the Fantope proximal operator
    int m_fan_maxinc;     // Parameter for computing the Fantope proximal operator
    int m_fan_maxiter;    // Parameter for computing the Fantope proximal operator

public:
    PPGOptimizer(const RefConstMat& S, int d) :
        m_p(S.rows()), m_pp(m_p * m_p), m_d(d), m_S(S),
        m_z1(m_p, m_p), m_z2(m_p, m_p), m_work(m_p, m_p),
        m_evecs(m_p, m_d)
    {}

    inline void init(const RefConstMat& x0, int fan_maxinc, int fan_maxiter, bool compute_diff_evec = true)
    {
        m_z1.noalias() = x0;
        m_z2.noalias() = m_z1;
        m_evecs.setZero();

        m_fan_inc = 2 * m_d;
        m_fan_maxinc = fan_maxinc;
        m_fan_maxiter = fan_maxiter;

        if(compute_diff_evec)
            m_ework.resize(m_p, m_d);
    }

    // z1 <- (z1 - z2) / 2 + prox_fantope(z2 + alpha * S)
    // Returns ||newz1 - z1||_F
    inline double update_z1(double lr, double l1, double l2, double eps, int verbose)
    {
        // Compute prox_fantope(z2 + alpha * S)
        m_work.noalias() = m_z2 + lr * m_S;
        m_fan_inc = prox_fantope_impl(
            m_work, lr * l1, lr * l2, m_d, m_fan_inc, m_fan_maxiter, m_work, eps, verbose
        );

        // Update z1
        double* z1p = m_z1.data();
        const double* z2p = m_z2.data();
        const double* pfp = m_work.data();
        double diff = 0.0;

        #pragma omp simd aligned(z1p, z2p, pfp: 32)
        for(int i = 0; i < m_pp; i++)
        {
            const double newz1 = 0.5 * (z1p[i] - z2p[i]) + pfp[i];
            diff += std::pow(newz1 - z1p[i], 2);
            z1p[i] = newz1;
        }

        // Adjust algorithm parameter
        if(verbose > 1)
            Rcpp::Rcout << "fan_dim = " << m_fan_inc << std::endl;
        m_fan_inc = std::max(5 * m_d, int(1.5 * m_fan_inc));
        m_fan_inc = std::min(m_fan_inc, m_fan_maxinc);
        m_fan_inc = std::min(m_fan_inc, int(m_p / 10));

        return std::sqrt(diff);
    }

    // z2 <- (z2 - z1) / 2 + soft_threshold(z1, penalty)
    //     = { (z1 + z2) / 2 - penalty, if z1 > penalty
    //       { (z1 + z2) / 2 + penalty, if z1 < -penalty
    //       { (z2 - z1) / 2          , otherwise
    // Returns ||newz2 - z2||_F
    inline double update_z2(double lr, double lambda, int verbose)
    {
        const double penalty = lr * lambda;
        const double* z1p = m_z1.data();
        double* z2p = m_z2.data();
        double diff = 0.0;

        #pragma omp simd aligned(z1p, z2p: 32)
        for(int i = 0; i < m_pp; i++)
        {
            const double z1 = z1p[i];
            const double z2 = z2p[i];
            const double x = 0.5 * (z1 + z2);
            const double newz2 = (z1 > penalty) ?
                                 (x - penalty) :
                                 ( (z1 < -penalty) ? (x + penalty) : 0.5 * (z2 - z1) );
            diff += std::pow(newz2 - z2, 2);
            z2p[i] = newz2;
        }
        return std::sqrt(diff);
    }

    // Returns ||UU' - VV'||_F, U = new eigenvectors, V = old eigenvectors
    // m_ework needs to be initialized by init(compute_diff_evec = true)
    inline double update_evecs()
    {
        MatrixXd& x = m_work;
        x.noalias() = 0.5 * (m_z1 + m_z2);
        Spectra::DenseSymMatProd<double> op(x);
        Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> >
            eigs(&op, m_d, std::max(10, 3 * m_d));
        eigs.init();
        eigs.compute();

        m_ework.noalias() = eigs.eigenvectors();
        double diff = projection_diff(m_ework, m_evecs);
        m_ework.swap(m_evecs);
        return std::sqrt(diff);
    }

    // On convergence, x = soft_threshold(z1, lr * lambda)
    const MatrixXd& get_saprse_x(double lr, double lambda, bool update_ev = true)
    {
        MatrixXd& x = m_work;
        soft_threshold(m_z1, lr * lambda, x);

        if(update_ev)
        {
            Spectra::DenseSymMatProd<double> op(x);
            Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> >
                eigs(&op, m_d, std::max(10, 3 * m_d));
            eigs.init();
            eigs.compute();
            m_evecs.noalias() = eigs.eigenvectors();
        }

        return x;
    }

    const MatrixXd& get_z1() { return m_z1; }
    const MatrixXd& get_z2() { return m_z2; }
    const MatrixXd& get_evecs() { return m_evecs; }
};



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

    PPGOptimizer opt(S, d);
    opt.init(x0, fan_maxinc, fan_maxiter, true);

    // Metrics in each iteration
    std::vector<double> err_v;
    std::vector<double> err_z1;
    std::vector<double> err_z2;

    const double l1 = mu * r1;
    const double l2 = mu / std::sqrt(double(p));

    int i = 0;
    for(i = 0; i < maxiter; i++)
    {
        if(verbose > 0)
            Rcpp::Rcout << "iter = " << i << std::endl;

        double diffz1 = opt.update_z1(lr, l1, l2, 0.001 / std::sqrt(i + 1.0), verbose);
        err_z1.push_back(diffz1);

        double diffz2 = opt.update_z2(lr, lambda, verbose);
        err_z2.push_back(diffz2);

        double diffev = opt.update_evecs();

        if(i > 0)
        {
            err_v.push_back(diffev);

            if(verbose > 0)
                Rcpp::Rcout << "  [info] err_v = " << diffev
                            << ", err_z1 = " << err_z1.back()
                            << ", err_z2 = " << err_z2.back()
                            << std::endl << std::endl;

            if(diffev < eps_abs || diffev < eps_rel * d)
                break;
        }
    }

    const MatrixXd& x = opt.get_saprse_x(lr, lambda);

    return List::create(
        Rcpp::Named("projection") = x,
        Rcpp::Named("evecs") = opt.get_evecs(),
        Rcpp::Named("err_v") = err_v,
        Rcpp::Named("err_z1") = err_z1,
        Rcpp::Named("err_z2") = err_z2,
        Rcpp::Named("niter") = i + 1,
        Rcpp::Named("z1") = opt.get_z1(),
        Rcpp::Named("z2") = opt.get_z2()
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

    PPGOptimizer opt(S, d);
    opt.init(x0, fan_maxinc, fan_maxiter, false);

    // Metrics in each iteration
    std::vector<double> err_x;  // directly use X_hat
    std::vector<double> err_v;  // top d eigenvectors of X_hat
    std::vector<double> times;

    double t1, t2;
    const double l1 = mu * r1;
    const double l2 = mu / std::sqrt(double(p));

    for(int i = 0; i < maxiter; i++)
    {
        if(verbose > 1 || (verbose > 0 && i % 50 == 0))
            Rcpp::Rcout << "\niter = " << i << ", alpha = " << lr << std::endl;

        t1 = get_wall_time();

        opt.update_z1(lr, l1, l2, 0.001 / std::sqrt(i + 1.0), verbose);
        opt.update_z2(lr, lambda, verbose);
        const MatrixXd& x = opt.get_saprse_x(lr, lambda);
        const MatrixXd& evecs = opt.get_evecs();

        t2 = get_wall_time();
        times.push_back(t2 - t1);
        err_x.push_back((x - Pi).norm());
        err_v.push_back((evecs * evecs.transpose() - Pi).norm());
    }

    return List::create(
        Rcpp::Named("projection") = opt.get_saprse_x(lr, lambda, false),
        Rcpp::Named("err_x") = err_x,
        Rcpp::Named("err_v") = err_v,
        Rcpp::Named("times") = times,
        Rcpp::Named("z1") = opt.get_z1(),
        Rcpp::Named("z2") = opt.get_z2()
    );
}



/*********************** Another implementation ***********************

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

*********************** End of another implementation ***********************/
