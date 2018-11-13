#include "fastfps.h"
#include "active.h"
#include "walltime.h"

using Rcpp::IntegerVector;
using Rcpp::NumericMatrix;
using Rcpp::List;

// [[Rcpp::export]]
List fastfps_path(
    NumericMatrix S, int d,
    double lambda_min, double lambda_max, int nlambda,
    int maxiter, double eps_abs, double eps_rel,
    double alpha0, double mu, double r,
    bool verbose = true)
{
    // Original dimension of the covariance matrix
    const int n0 = S.nrow();
    const int p0 = S.ncol();
    if(n0 != p0)
        Rcpp::stop("S must be square");

    MapMat S0(S.begin(), p0, p0);

    // Create the lambda sequence
    VectorXd lambda(nlambda);
    const double llmin = std::log(lambda_min), llmax = std::log(lambda_max);
    const double step = (llmax - llmin) / (nlambda - 1);
    for(int i = 0; i < nlambda; i++)
        lambda[i] = std::exp(llmin + step * i);

    // Compute active set based on the smallest lambda
    std::vector<Triple> pattern;
    std::vector<int> act;
    analyze_pattern(S0, pattern);
    find_active(pattern, d, lambda_min, act);

    // Create submatrix if possible
    const int p = act.size();
    IntegerVector act_ind(p);
    MatrixXd Sact;
    if(verbose)
        Rcpp::Rcout << "Reduced active set size to " << p << std::endl;
    if(p < p0)
    {
        submatrix_act(S0, act, Sact);
        for(int i = 0; i < p; i++)
            act_ind[i] = act[i] + 1;
    } else {
        for(int i = 0; i < p; i++)
            act_ind[i] = i + 1;
    }
    // Adjust mu based on the new active set size
    mu = std::min(mu, std::sqrt(double(p)));

    // Reference to the submatrix or the original matrix
    MapMat Smat((p < p0) ? (Sact.data()) : (S.begin()), p, p);
    // Projection matrices
    MatrixXd x(p, p), xold(p, p);
    dgCMatrix xsp(p);
    // Objective function values
    std::vector<double> fn_obj, fn_feas, fn_feas1, fn_feas2, time;

    // Eigenvalue computation
    // Number of eigenvalue pairs to compute, i.e.,
    // the largest N and smallest N eigenvalues
    const int N = 2;
    VectorXd evals(2 * N);
    VectorXd evals_new(2 * N);
    MatrixXd evecs(p, 2 * N);
    VectorXd diag(p);

    // Initial guess -- using partial eigen decomposition
    initial_guess(Smat, d, x);

    // Final result
    List res(nlambda);

    for(int l = 0; l < nlambda; l++)
    {
        const double curr_lambda = lambda[nlambda - l - 1];
        fn_obj.clear();
        fn_feas.clear();
        fn_feas1.clear();
        fn_feas2.clear();
        time.clear();

        if(verbose)
            Rcpp::Rcout << "lambda = " << curr_lambda << std::endl;

        double alpha = 0.0;
        double time1, time2;
        int i;
        for(i = 0; i < maxiter; i++)
        {
            if(verbose)
                Rcpp::Rcout << "Iter " << i << std::endl;

            time1 = get_wall_time();
            alpha = alpha0 / (l + 1.0) / (i + 1.0);

            // L1 thresholding, x -> xsp
            xsp.soft_thresh(x.data(), curr_lambda * alpha);

            // Eigenvalue shrinkage
            eigs_sparse_both_ends_primme<N>(xsp, evals, evecs);
            for(int i = 0; i < N; i++)
            {
                evals_new[i]     = lambda_max_thresh(evals[i],     alpha * mu * r);
                evals_new[N + i] = lambda_min_thresh(evals[N + i], alpha * mu * r);
            }
            evals_new.noalias() -= evals;
            // Save x to xold and update x
            x.swap(xold);
            rank_r_update_sparse<2 * N>(xsp, evals_new, evecs, x);

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
            double feas1 = 0.0;
            for(int i = 0; i < N; i++)
            {
                feas1 += std::max(0.0, evals[i] - 1.0) + std::max(0.0, -evals[N + i]);
            }
            feas1 *= (mu * r);
            const double feas2 = mu * std::sqrt(double(p)) * std::abs(tr_shift);
            fn_feas1.push_back(feas1);
            fn_feas2.push_back(feas2);
            fn_feas.push_back(feas1 + feas2);

            // Gradient descent with momentum term
            if(i >= 2)
            {
                const double w = (double(i - 1.0) / double(i + 2.0));
                sym_mat_update(p, x.data(), xold.data(), Smat.data(), w, -w, alpha);
            } else {
                sym_mat_update(p, x.data(), Smat.data(), alpha);
            }
            const double xnorm = x.norm();
            const double radius = std::sqrt(double(d));
            if(xnorm > radius)
            {
                x /= (xnorm / radius);
            }

            // Record elapsed time and objective function values
            time2 = get_wall_time();
            fn_obj.push_back(fps_objfn(p, x.data(), Smat.data(), curr_lambda));
            time.push_back(time2 - time1);

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

        // To make the final solution sparse
        xsp.soft_thresh(x.data(), curr_lambda * alpha);

        res[l] = List::create(
            Rcpp::Named("lambda")     = curr_lambda,
            Rcpp::Named("active")     = act_ind,
            Rcpp::Named("projection") = xsp.to_spmat(),
            Rcpp::Named("objfn")      = fn_obj,
            Rcpp::Named("feasfn1")    = fn_feas1,
            Rcpp::Named("feasfn2")    = fn_feas2,
            Rcpp::Named("feasfn")     = fn_feas,
            Rcpp::Named("niter")      = std::min(i + 1, maxiter),
            Rcpp::Named("time")       = time
        );
    }

    return res;
}



// [[Rcpp::export]]
List fastfps(NumericMatrix S, int d, double lambda,
             int maxiter, double eps_abs, double eps_rel,
             double alpha0, double mu, double r,
             bool exact_feas = false, bool verbose = true)
{
    // Original dimension of the covariance matrix
    const int n0 = S.nrow();
    const int p0 = S.ncol();
    if(n0 != p0)
        Rcpp::stop("S must be square");

    MapMat S0(S.begin(), p0, p0);

    // Compute active set
    std::vector<Triple> pattern;
    std::vector<int> act;
    analyze_pattern(S0, pattern);
    find_active(pattern, d, lambda, act);

    // Create submatrix if possible
    const int p = act.size();
    IntegerVector act_ind(p);
    MatrixXd Sact;
    if(verbose)
        Rcpp::Rcout << "Reduced active set size to " << p << std::endl;
    if(p < p0)
    {
        submatrix_act(S0, act, Sact);
        for(int i = 0; i < p; i++)
            act_ind[i] = act[i] + 1;
    } else {
        for(int i = 0; i < p; i++)
            act_ind[i] = i + 1;
    }
    // Adjust mu based on the new active set size
    mu = std::min(mu, std::sqrt(double(p)));

    // Reference to the submatrix or the original matrix
    MapMat Smat((p < p0) ? (Sact.data()) : (S.begin()), p, p);
    // Projection matrices
    MatrixXd x(p, p), xold(p, p);
    dgCMatrix xsp(p);
    // Objective function values
    std::vector<double> fn_obj, fn_feas, fn_feas1, fn_feas2, time;

    // Initial guess -- using partial eigen decomposition
    initial_guess(Smat, d, x);

    // Eigenvalue computation
    // Number of eigenvalue pairs to compute, i.e.,
    // the largest N and smallest N eigenvalues
    const int N = 2;
    VectorXd evals(2 * N);
    VectorXd evals_new(2 * N);
    MatrixXd evecs(p, 2 * N);
    VectorXd diag(p);

    double alpha = 0.0;
    double time1, time2;
    int i;
    for(i = 0; i < maxiter; i++)
    {
        if(verbose)
            Rcpp::Rcout << "Iter " << i << std::endl;

        time1 = get_wall_time();
        alpha = alpha0 / (i + 1.0);

        // L1 thresholding, xsp <- soft_thresh(x)
        xsp.soft_thresh(x.data(), lambda * alpha);

        // Eigenvalue shrinkage
        eigs_sparse_both_ends_primme<N>(xsp, evals, evecs);
        // const double lmax_new = lambda_max_thresh(evals[0], alpha * mu * r);
        // const double lmin_new = lambda_min_thresh(evals[1], alpha * mu * r);
        for(int i = 0; i < N; i++)
        {
            evals_new[i]     = lambda_max_thresh(evals[i],     alpha * mu * r);
            evals_new[N + i] = lambda_min_thresh(evals[N + i], alpha * mu * r);
        }
        evals_new.noalias() -= evals;
        // Save x to xold and update x
        x.swap(xold);
        // x <- xsp + (l1_new - l1) * v1 * v1' + (lp_new - lp) * vp * vp'
        // rank2_update_sparse(xsp, lmax_new - evals[0], evecs.col(0), lmin_new - evals[1], evecs.col(1), x);
        rank_r_update_sparse<2 * N>(xsp, evals_new, evecs, x);

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
        double feas1 = 0.0;
        for(int i = 0; i < N; i++)
        {
            feas1 += std::max(0.0, evals[i] - 1.0) + std::max(0.0, -evals[N + i]);
        }
        feas1 *= (mu * r);
        const double feas2 = mu * std::sqrt(double(p)) * std::abs(tr_shift);
        fn_feas1.push_back(feas1);
        fn_feas2.push_back(feas2);
        fn_feas.push_back(feas1 + feas2);

        // Gradient descent with momentum term
        if(i >= 2)
        {
            // x.noalias() += (double(i - 1.0) / double(i + 2.0)) * (x - xold) + alpha * Smat;
            const double w = (double(i - 1.0) / double(i + 2.0));
            sym_mat_update(p, x.data(), xold.data(), Smat.data(), w, -w, alpha);
        } else {
            // x.noalias() += alpha * Smat;
            sym_mat_update(p, x.data(), Smat.data(), alpha);
        }
        const double xnorm = sym_mat_norm(p, x.data());
        const double radius = std::sqrt(double(d));
        if(xnorm > radius)
        {
            x /= (xnorm / radius);
        }

        // Record elapsed time and objective function values
        time2 = get_wall_time();
        // fn_obj.push_back(-Smat.cwiseProduct(x).sum() + lambda * x.cwiseAbs().sum());
        fn_obj.push_back(fps_objfn(p, x.data(), Smat.data(), lambda));
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

    // To make the final solution sparse
    xsp.soft_thresh(x.data(), lambda * alpha);

    return List::create(
        Rcpp::Named("active")     = act_ind,
        Rcpp::Named("projection") = xsp.to_spmat(),
        Rcpp::Named("objfn")      = fn_obj,
        Rcpp::Named("feasfn1")    = fn_feas1,
        Rcpp::Named("feasfn2")    = fn_feas2,
        Rcpp::Named("feasfn")     = fn_feas,
        Rcpp::Named("niter")      = std::min(i + 1, maxiter),
        Rcpp::Named("time")       = time
    );
}
