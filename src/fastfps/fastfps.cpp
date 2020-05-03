#include "fastfps.h"
#include "active.h"
#include "symmat.h"
#include "walltime.h"

using Rcpp::IntegerVector;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::List;

// [[Rcpp::export]]
List fastfps_internal(
    MapMat S, int d,
    double lambda_min, double lambda_max, int nlambda,
    int maxiter, double eps_abs, double eps_rel,
    double alpha0, double mu0, double r,
    bool verbose = true)
{
    // Original dimension of the covariance matrix
    const int n0 = S.rows();
    const int p0 = S.cols();
    if(n0 != p0)
        Rcpp::stop("S must be square");

    // Create the lambda sequence
    VectorXd lambdas(nlambda);
    lambdas[0] = lambda_max;
    lambdas[nlambda - 1] = lambda_min;
    const double llmin = std::log(lambda_min), llmax = std::log(lambda_max);
    // nlambda has to be >=3 in order to enter the for-loop,
    // so we don't worry about the 0/0 issue
    const double step = (llmax - llmin) / (nlambda - 1);
    for(int i = 1; i < nlambda - 1; i++)
        lambdas[i] = std::exp(llmax - step * i);

    // Compute active set
    ActiveSet act_set(S);
    act_set.analyze_pattern();
    act_set.find_active(d, lambdas);
    const std::vector< std::vector<int> >& inc_act = act_set.incremental_active_set();

    // Submatrix
    SymMat Smat;
    std::vector<int> flat_act = act_set.compute_submatrix(Smat);
    const int pmax = Smat.dim();
    IntegerVector act_ind(pmax);
    for(int i = 0; i < pmax; i++)
        act_ind[i] = flat_act[i] + 1;

    // Projection matrices
    SymMat x(pmax), xold(pmax);
    dgCMatrix xsp(pmax);

    // Objective function values
    std::vector<double> fn_obj, fn_feas, fn_feas1, fn_feas2, time;

    // Initial guess -- using partial eigen decomposition
    // Set initial dimension
    Smat.set_dim(inc_act[0].size());
    x.set_dim(inc_act[0].size());
    initial_guess(Smat, d, x);

    // Eigenvalue computation
    // Number of eigenvalue pairs to compute, i.e.,
    // the largest N and smallest N eigenvalues
    const int N = 1;
    VectorXd evals(2 * N);
    VectorXd evals_new(2 * N);
    // Size of eigenvectors depend on the actual dimension
    // Here we allocate memory to the largest size
    MatrixXd evecs_data(pmax, 2 * N);

    // Final result
    List res(nlambda);

    // Size of active set for each lambda
    int p = 0;
    for(int l = 0; l < nlambda; l++)
    {
        const double curr_lambda = lambdas[l];
        fn_obj.clear();
        fn_feas.clear();
        fn_feas1.clear();
        fn_feas2.clear();
        time.clear();

        if(verbose)
            Rcpp::Rcout << "** lambda = " << curr_lambda << std::endl;

        // Compute the current active set size
        p += inc_act[l].size();
        Smat.set_dim(p);
        x.set_dim(p);
        xold.set_dim(p);
        xsp.resize(p);

        if(verbose)
            Rcpp::Rcout << "** active set size: " << p << std::endl;

        // Adjust mu based on the new active set size
        const double mu = std::min(mu0, std::sqrt(double(p)));

        double alpha = 0.0;
        double time1, time2;
        int i;
        for(i = 0; i < maxiter; i++)
        {
            if(verbose)
            {
                Rcpp::Rcout << "=";
                if(i % 50 == 49)
                    Rcpp::Rcout << " " << i + 1 << " iterations"<< std::endl;
            }

            time1 = get_wall_time();
            alpha = alpha0 / (l + 1.0) / std::sqrt(i + 1.0);

            // L1 thresholding, xsp <- soft_thresh(x)
            xsp.soft_thresh(x, curr_lambda * alpha);

            // Eigenvalue shrinkage
            MapMat evecs(evecs_data.data(), p, 2 * N);
            eigs_sparse_both_ends_primme<N>(xsp, evals, evecs);
            for(int i = 0; i < N; i++)
            {
                evals_new[i]     = lambda_max_thresh(evals[i],     alpha * mu * r);
                evals_new[N + i] = lambda_min_thresh(evals[N + i], alpha * mu * r);
            }
            evals_new.noalias() -= evals;
            // Save x to xold and update x
            x.swap(xold);
            rank_r_update_sparse<2 * N>(x, xsp, evals_new, evecs);

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
            fn_obj.push_back(fps_objfn(Smat, x, curr_lambda));
            time.push_back(time2 - time1);

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
        xsp.soft_thresh(x, curr_lambda * alpha);

        res[l] = List::create(
            Rcpp::Named("lambda")     = curr_lambda,
            Rcpp::Named("act_size")   = p,
            Rcpp::Named("projection") = xsp.to_spmat(),
            Rcpp::Named("objfn")      = fn_obj,
            Rcpp::Named("feasfn1")    = fn_feas1,
            Rcpp::Named("feasfn2")    = fn_feas2,
            Rcpp::Named("feasfn")     = fn_feas,
            Rcpp::Named("niter")      = std::min(i + 1, maxiter),
            Rcpp::Named("time")       = time
        );

        if(verbose)
            Rcpp::Rcout << std::endl << std::endl;
    }

    return List::create(
        Rcpp::Named("dim")      = p0,
        Rcpp::Named("rank")     = d,
        Rcpp::Named("active")   = act_ind,
        Rcpp::Named("lambdas")  = lambdas,
        Rcpp::Named("solution") = res
    );
}


// Expand a sparse projection matrix to dimension p and reorder it according
// to its active set index
// [[Rcpp::export]]
SpMat reorder_projection(int p, MapSpMat proj, IntegerVector act_id)
{
    // Check dimension
    const int p_act = proj.rows();
    if(p_act > p || p_act != proj.cols() || p_act != act_id.length())
        Rcpp::stop("matrix dimensions do not match");

    // Result matrix
    SpMat res(p, p);

    // Iterate over elements of proj and insert them to res
    // Note that only the lower triangular part of proj is referenced
    // Also note that act_id is one-based
    for(int j = 0; j < p_act; j++)
    {
        for(MapSpMat::InnerIterator it(proj, j); it; ++it)
        {
            const int proj_row = it.row();
            // Ignore upper triangular part
            if(proj_row < j)
                continue;

            const int res_row = act_id[proj_row] - 1;
            const int res_col = act_id[j] - 1;
            const double res_val = it.value();
            res.insert(res_row, res_col) = res_val;
            // Symmetry
            if(res_row != res_col)
                res.insert(res_col, res_row) = res_val;
        }
    }

    res.makeCompressed();

    return res;
}
