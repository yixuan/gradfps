#include "common.h"
#include "symmat.h"
#include "sparsemat.h"
#include "eigenvalue.h"
#include "walltime.h"

using Rcpp::IntegerVector;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::List;

// Thresholding of eigenvalues
inline double lambda_max_thresh(double x, double thresh)
{
    return (x > 1.0 + thresh) ?
           (x - thresh) :
           ((x > 1.0) ? 1.0 : x);
}
inline double lambda_min_thresh(double x, double thresh)
{
    return (x > 0.0) ?
           (x) :
           ((x > -thresh) ? 0.0 : (x + thresh));
}

// Apply a rank-r update on a sparse matrix x.
// Only the lower triangular part is read and written
// x <- xsp + a1 * v1 * v1' + ... + ar * vr * vr'
template <int r>
void rank_r_update_sparse(SymMat& x, const dgCMatrix& xsp, const RefVec& a, const RefMat& v)
{
    const int xn = x.dim();
    if(xn != xsp.rows())
        throw std::invalid_argument("matrix sizes do not match");

    double vj[r];

    for(int j = 0; j < xn; j++)
    {
        for(int k = 0; k < r; k++)
        {
            vj[k] = a[k] * v.coeff(j, k);
        }
        for(int i = j; i < xn; i++)
        {
            double sum = 0.0;
            for(int k = 0; k < r; k++)
            {
                sum += vj[k] * v.coeff(i, k);
            }
            x.ref(i, j) = sum;
        }
    }

    // Add the sparse matrix
    xsp.add_to(x);
}

// FPS objective function: -<S, X> + lambda * ||X||_1
// Only the lower triangular part is read
inline double fps_objfn(const SymMat& smat, const SymMat& xmat, double lambda)
{
    const int sn = smat.dim();
    const int sldim = smat.lead_dim();
    const int xn = xmat.dim();
    const int xldim = xmat.lead_dim();

    if(sn != xn)
        throw std::invalid_argument("matrix sizes do not match");

    const double* x = xmat.data();
    const double* x_col_begin = x;
    const double* x_col_end   = x + xn;

    const double* s = smat.data();
    const double* s_col_begin = s;

    double diag1 = 0.0, diag2 = 0.0;
    double off_diag1 = 0.0, off_diag2 = 0.0;

    for(int j = 0; j < xn; j++)
    {
        x = x_col_begin + j;
        s = s_col_begin + j;

        diag1 += (*s) * (*x);
        diag2 += std::abs(*x);

        x = x + 1;
        s = s + 1;

        for(; x < x_col_end; x++, s++)
        {
            off_diag1 += (*s) * (*x);
            off_diag2 += std::abs(*x);
        }

        x_col_begin += xldim;
        x_col_end   += xldim;
        s_col_begin += sldim;
    }

    return -(diag1 + off_diag1 * 2) + lambda * (diag2 + off_diag2 * 2);
}

// [[Rcpp::export]]
List gradfps_subgrad_(
    MapMat S, MapMat x0, int d, double lambda,
    double lr, double mu, double r1, double r2,
    int maxiter = 500, double eps_abs = 1e-3, double eps_rel = 1e-3, int verbose = 0
)
{
    // Dimension of S
    const int p = S.rows();

    // Projection matrices
    SymMat x(x0), xold(p), Smat(S);
    dgCMatrix xsp(p);

    // Objective function values
    std::vector<double> fn_obj, fn_feas, fn_feas1, fn_feas2, times;

    // Eigenvalue computation
    // Number of eigenvalue pairs to compute, i.e.,
    // the largest N and smallest N eigenvalues
    const int N = 1;
    // Eigenvalues and eigenvectors
    VectorXd evals(2 * N), evals_delta(2 * N);
    MatrixXd evecs(p, 2 * N);

    const double alpha0 = lr;
    double alpha = alpha0;
    double time1, time2;
    int i;
    for(i = 0; i < maxiter; i++)
    {
        if(verbose)
        {
            Rcpp::Rcout << "=";
            if(i % 50 == 49)
            {
                Rcpp::Rcout << " " << i + 1 << " iterations"<< std::endl;
                Rcpp::Rcout << "\neigenvalues = " << evals.transpose() << std::endl << std::endl;
            }
        }

        time1 = get_wall_time();
        alpha = alpha0 / std::sqrt(i / 10 + 1.0);

        // L1 thresholding, xsp <- soft_thresh(x)
        xsp.soft_thresh(x, lambda * alpha);

        // Eigenvalue shrinkage
        eigs_sparse_both_ends_primme<N>(xsp, evals, evecs);
        for(int i = 0; i < N; i++)
        {
            evals_delta[i]     = lambda_max_thresh(evals[i],     alpha * mu * r1);
            evals_delta[N + i] = lambda_min_thresh(evals[N + i], alpha * mu * r2);
        }
        evals_delta.noalias() -= evals;
        // Save x to xold and update x
        x.swap(xold);
        rank_r_update_sparse<2 * N>(x, xsp, evals_delta, evecs);

        // Trace shrinkage
        const double tbar = x.trace() / p;
        const double tr_shift = double(d) / double(p) - tbar;
        const double beta = alpha * mu / std::sqrt(double(p)) / std::abs(tr_shift);
        // d' = d + s, where d is the original diagonal elements, and s is the shift
        // If beta >= 1, d <- d' = d + s
        // Otherwise, d <- (1 - beta) * d + beta * d' = d + beta * s
        // In a single formula, d <- d + min(beta, 1) * s
        x.diag_add(std::min(beta, 1.0) * tr_shift);

        // Compute (approximate) feasibility loss
        double feas1 = 0.0;
        for(int i = 0; i < N; i++)
        {
            feas1 += std::max(0.0, evals[i] - 1.0) * r1 + std::max(0.0, -evals[N + i]) * r2;
        }
        feas1 *= mu;
        const double feas2 = mu * std::sqrt(double(p)) * std::abs(tr_shift);
        fn_feas1.push_back(feas1);
        fn_feas2.push_back(feas2);
        fn_feas.push_back(feas1 + feas2);

        // Gradient descent with momentum term
        if(i >= 2)
        {
            // x += (double(i - 1.0) / double(i + 2.0)) * (x - xold) + alpha * Smat;
            // const double w = (double(i - 1.0) / double(i + 2.0));
            // x.add(w, -w, alpha, xold, Smat);
            x.add(alpha, Smat);
        } else {
            // x += alpha * Smat;
            x.add(alpha, Smat);
        }
        const double xnorm = x.norm();
        const double radius = std::sqrt(double(d));
        // Scale to an L2 ball if too large
        if(xnorm > radius)
            x.scale(radius / xnorm);

        // Record elapsed time and objective function values
        time2 = get_wall_time();
        fn_obj.push_back(fps_objfn(Smat, x, lambda));
        times.push_back(time2 - time1);

        // Convergence test, only after 10 iterations
        if(i < 10)
            continue;
        // Feasibility loss, using the average of the most recent 5 values
        const double feas_curr = std::accumulate(fn_feas.end() - 5, fn_feas.end(), 0.0) / 5.0;
        const double feas_prev = std::accumulate(fn_feas.end() - 10, fn_feas.end() - 5, 0.0) / 5.0;
        const double feas_diff = std::abs(feas_curr - feas_prev);
        const bool feas_conv = feas_diff < eps_abs || feas_diff < eps_rel * std::abs(feas_prev);
        // Objective function, using the average of the most recent 5 values
        const double obj_curr = std::accumulate(fn_obj.end() - 5, fn_obj.end(), 0.0) / 5.0;
        const double obj_prev = std::accumulate(fn_obj.end() - 10, fn_obj.end() - 5, 0.0) / 5.0;
        const double obj_diff = std::abs(obj_curr - obj_prev);
        const bool obj_conv = obj_diff < eps_abs || obj_diff < eps_rel * std::abs(obj_prev);

        if(feas_conv && obj_conv)
            break;
    }

    // To make the final solution sparse
    xsp.soft_thresh(x, lambda * alpha);

    return List::create(
        Rcpp::Named("projection") = xsp.to_spmat(),
        Rcpp::Named("objfn")      = fn_obj,
        Rcpp::Named("feasfn1")    = fn_feas1,
        Rcpp::Named("feasfn2")    = fn_feas2,
        Rcpp::Named("feasfn")     = fn_feas,
        Rcpp::Named("niter")      = std::min(i + 1, maxiter),
        Rcpp::Named("times")      = times
    );
}

// [[Rcpp::export]]
List gradfps_subgrad_benchmark_(
    MapMat S, MapMat Pi, MapMat x0, int d, double lambda,
    double lr, double mu, double r1, double r2,
    int maxiter = 500, double eps_abs = 1e-3, double eps_rel = 1e-3, int verbose = 0
)
{
    // Dimension of S
    const int p = S.rows();

    // Projection matrices
    SymMat x(x0), xold(p), Smat(S);
    dgCMatrix xsp(p);

    // Metrics in each iteration
    std::vector<double> err_x;  // directly use X_hat
    std::vector<double> err_v;  // top d eigenvectors of X_hat
    std::vector<double> times;

    // Eigenvalue computation
    // Number of eigenvalue pairs to compute, i.e.,
    // the largest N and smallest N eigenvalues
    const int N = 1;
    // Eigenvalues and eigenvectors
    VectorXd evals(2 * N), evals_delta(2 * N);
    MatrixXd evecs(p, 2 * N);

    const double alpha0 = lr;
    double alpha = alpha0;
    double time1, time2;
    for(int i = 0; i < maxiter; i++)
    {
        if(verbose)
        {
            Rcpp::Rcout << "=";
            if(i % 50 == 49)
            {
                Rcpp::Rcout << " " << i + 1 << " iterations"<< std::endl;
                Rcpp::Rcout << "\neigenvalues = " << evals.transpose() << std::endl << std::endl;
            }
        }

        time1 = get_wall_time();
        alpha = alpha0 / std::sqrt(i / 10 + 1.0);

        // L1 thresholding, xsp <- soft_thresh(x)
        xsp.soft_thresh(x, lambda * alpha);

        // Eigenvalue shrinkage
        eigs_sparse_both_ends_primme<N>(xsp, evals, evecs);
        for(int i = 0; i < N; i++)
        {
            evals_delta[i]     = lambda_max_thresh(evals[i],     alpha * mu * r1);
            evals_delta[N + i] = lambda_min_thresh(evals[N + i], alpha * mu * r2);
        }
        evals_delta.noalias() -= evals;
        // Save x to xold and update x
        x.swap(xold);
        rank_r_update_sparse<2 * N>(x, xsp, evals_delta, evecs);

        // Trace shrinkage
        const double tbar = x.trace() / p;
        const double tr_shift = double(d) / double(p) - tbar;
        const double beta = alpha * mu / std::sqrt(double(p)) / std::abs(tr_shift);
        // d' = d + s, where d is the original diagonal elements, and s is the shift
        // If beta >= 1, d <- d' = d + s
        // Otherwise, d <- (1 - beta) * d + beta * d' = d + beta * s
        // In a single formula, d <- d + min(beta, 1) * s
        x.diag_add(std::min(beta, 1.0) * tr_shift);

        // Gradient descent with momentum term
        if(i >= 2)
        {
            // x += (double(i - 1.0) / double(i + 2.0)) * (x - xold) + alpha * Smat;
            // const double w = (double(i - 1.0) / double(i + 2.0));
            // x.add(w, -w, alpha, xold, Smat);
            x.add(alpha, Smat);
        } else {
            // x += alpha * Smat;
            x.add(alpha, Smat);
        }
        const double xnorm = x.norm();
        const double radius = std::sqrt(double(d));
        // Scale to an L2 ball if too large
        if(xnorm > radius)
            x.scale(radius / xnorm);

        // Record elapsed time and objective function values
        time2 = get_wall_time();
        times.push_back(time2 - time1);

        err_x.push_back(x.distance(Pi));
        MatrixXd evecs = eigs_dense_largest_spectra(x, d, 1e-6);
        err_v.push_back((evecs * evecs.transpose() - Pi).norm());
    }

    // To make the final solution sparse
    xsp.soft_thresh(x, lambda * alpha);

    return List::create(
        Rcpp::Named("projection") = xsp.to_spmat(),
        Rcpp::Named("err_x") = err_x,
        Rcpp::Named("err_v") = err_v,
        Rcpp::Named("times") = times
    );
}
