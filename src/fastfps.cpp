#include "fastfps.h"
#include "active.h"
#include "walltime.h"

using Rcpp::NumericMatrix;
using Rcpp::List;

// [[Rcpp::export]]
List fastfps(NumericMatrix S, int d, double lambda,
             int maxiter, double eps_abs, double eps_rel,
             double alpha0, double mu, double r,
             bool exact_feas = false, bool verbose = true)
{
    const int n0 = S.nrow();
    const int p0 = S.ncol();
    if(n0 != p0)
        Rcpp::stop("S must be square");

    MapMat S0(S.begin(), p0, p0);

    // Compute active set
    std::vector<Triple> pattern;
    analyze_pattern(S0, pattern);
    std::vector<int> act;
    find_active(pattern, d, lambda, act);

    // Create submatrix if possible
    const int p = act.size();
    MatrixXd Sact;
    if(verbose)
        Rcpp::Rcout << "Reduced active set size to " << p << std::endl;
    if(p < p0)
        submatrix_act(S0, act, Sact);
    MapMat Smat((p < p0) ? (Sact.data()) : (S.begin()), p, p);

    MatrixXd x(p, p), xold(p, p);
    SpMat xsp(p, p);

    std::vector<double> fn_obj, fn_feas, fn_feas1, fn_feas2, time;

    // Initial guess -- using partial eigen decomposition
    initial_guess(Smat, d, x);

    // Eigenvalue computation
    VectorXd evals(2);
    MatrixXd evecs(p, 2);
    VectorXd diag(p);

    double alpha = 0.0;
    double time1, time2;
    for(int i = 0; i < maxiter; i++)
    {
        time1 = get_wall_time();
        alpha = alpha0 / (i + 1.0);

        // L1 thresholding
        soft_thresh_sparse(x, lambda * alpha, xsp);

        // Eigenvalue shrinkage
        eigs_sparse_both_ends_primme(xsp, evals, evecs);
        const double lmax_new = lambda_max_thresh(evals[0], alpha * mu * r);
        const double lmin_new = lambda_min_thresh(evals[1], alpha * mu * r);
        rank2_update_sparse(xsp, lmax_new - evals[0], evecs.col(0), lmin_new - evals[1], evecs.col(1), x);

        // Trace shrinkage
        const double tbar = x.diagonal().mean();
        const double tr_shift = double(d) / double(p) - tbar;
        diag.array() = x.diagonal().array() + tr_shift;
        const double beta = alpha * mu / std::sqrt(double(p)) / std::abs(tr_shift);
        if(beta >= 1.0)
        {
            x.diagonal().noalias() = diag;
        } else {
            x.diagonal().noalias() = (1.0 - beta) * x.diagonal() + beta * diag;
        }

        // Compute (approximate) feasibility loss
        const double feas1 = mu * r * (std::max(0.0, -evals[1]),
                                       std::max(0.0, evals[0] - 1.0));
        const double feas2 = mu * std::sqrt(double(p)) * std::abs(tr_shift);
        fn_feas1.push_back(feas1);
        fn_feas2.push_back(feas2);
        fn_feas.push_back(feas1 + feas2);
        if(verbose)
            Rcpp::Rcout << i << std::endl;

        // Gradient descent
        x.triangularView<Eigen::Upper>() = x.triangularView<Eigen::Lower>().transpose();
        if(i >= 2)
        {
            x.noalias() += (double(i - 1.0) / double(i + 2.0)) * (x - xold) + alpha * Smat;
        } else {
            x.noalias() += alpha * Smat;
        }
        const double xnorm = x.norm();
        const double radius = std::sqrt(double(d));
        if(xnorm > radius)
        {
            x /= (xnorm / radius);
        }
        xold.noalias() = x;

        // Record elapsed time and objective function values
        time2 = get_wall_time();
        fn_obj.push_back(-Smat.cwiseProduct(x).sum() + lambda * x.cwiseAbs().sum());
        time.push_back(time2 - time1);

        // Compute exact feasibility loss
        if(exact_feas)
        {
            eigs_dense_both_ends_spectra(x, evals);
            const double feas1 = mu * r * (std::max(0.0, -evals[1]),
                                           std::max(0.0, evals[0] - 1.0));
            const double tbar = x.diagonal().mean();
            const double tr_shift = double(d) / double(p) - tbar;
            const double feas2 = mu * std::sqrt(double(p)) * std::abs(tr_shift);
            fn_feas1.back() = feas1;
            fn_feas2.back() = feas2;
            fn_feas.back() = feas1 + feas2;
        }

        // Convergence test, only after 10 iterations
        if(i < 10)
            continue;
        // Feasibility loss, using the average of the most recent 5 values
        const double feas_curr = std::accumulate(fn_feas.end() - 5, fn_feas.end(), 0.0) / 5.0;
        const double feas_prev = std::accumulate(fn_feas.end() - 6, fn_feas.end() - 1, 0.0) / 5.0;
        const double feas_diff = std::abs(feas_curr - feas_prev);
        const bool feas_conv = feas_diff < eps_abs || feas_diff < eps_rel * std::abs(feas_prev);
        // Objective function, using the average of the most recent 5 values
        const double obj_curr = std::accumulate(fn_obj.end() - 5, fn_obj.end(), 0.0) / 5.0;
        const double obj_prev = std::accumulate(fn_obj.end() - 6, fn_obj.end() - 1, 0.0) / 5.0;
        const double obj_diff = std::abs(obj_curr - obj_prev);
        const bool obj_conv = obj_diff < eps_abs || obj_diff < eps_rel * std::abs(obj_prev);

        if(feas_conv && obj_conv)
            break;
    }

    soft_thresh_sparse(x, lambda * alpha, xsp);
    xsp.makeCompressed();

    return List::create(
        Rcpp::Named("projection") = xsp,
        Rcpp::Named("objfn")      = fn_obj,
        Rcpp::Named("feasfn1")    = fn_feas1,
        Rcpp::Named("feasfn2")    = fn_feas2,
        Rcpp::Named("feasfn")     = fn_feas,
        Rcpp::Named("time")       = time
    );
}
