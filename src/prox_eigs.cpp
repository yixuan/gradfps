#include "prox_eigs.h"

using Rcpp::NumericMatrix;

// Proximal operator of alpha * f(x), f(x) = max(0, eigmax(x) - 1)
// [[Rcpp::export]]
NumericMatrix prox_eigmax(
    NumericMatrix x, double penalty, int init_neval = 10, int inc = 10, int max_try = 10
)
{
    const int n = x.nrow();
    Eigen::Map<const MatrixXd> mat(x.begin(), n, n);

    // Compute initial eigenvalues
    IncrementalEig inceig;
    inceig.init(mat, init_neval + inc * max_try, init_neval, 0, 0);
    // Test stopping criterion
    const VectorXd& evals = inceig.largest_eigenvalues();
    int neval = inceig.num_computed_largest();
    double shrink;
    int loc = shrink_max(evals.data(), neval, penalty, shrink);
    // If loc < 0, then all eigenvalues of x are <= 1, no shrinkage is needed
    if(loc < 0)
        return x;

    // Extend the search range if the shrinkage does not happen in computed eigenvalues
    if(loc >= neval - 1)
    {
        for(int i = 0; i < max_try; i++)
        {
            inceig.compute_next_largest(inc);
            const VectorXd& evals = inceig.largest_eigenvalues();
            neval = inceig.num_computed_largest();
            loc = shrink_max(evals.data(), neval, penalty, shrink);
            if(loc <= neval - 2)
                break;
        }
    }

    // Compute the final result
    NumericMatrix res = Rcpp::clone(x);
    Eigen::Map<MatrixXd> res_(res.begin(), n, n);
    // Difference of eigenvalues
    VectorXd evals_delta = shrink - inceig.largest_eigenvalues().head(loc + 1).array();
    // Eigenvectors
    inceig.compute_eigenvectors(loc + 1, 0);
    const MatrixXd& evecs = inceig.largest_eigenvectors();
    res_.noalias() += evecs.leftCols(loc + 1) * evals_delta.asDiagonal() * evecs.leftCols(loc + 1).transpose();
    return res;
}

// Proximal operator of alpha * f(x), f(x) = max(0, -eigmin(x))
// [[Rcpp::export]]
NumericMatrix prox_eigmin(
    NumericMatrix x, double penalty, int init_neval = 10, int inc = 10, int max_try = 10
)
{
    const int n = x.nrow();
    Eigen::Map<const MatrixXd> mat(x.begin(), n, n);

    // Compute initial eigenvalues
    IncrementalEig inceig;
    inceig.init(mat, 0, 0, init_neval + inc * max_try, init_neval);
    // Test stopping criterion
    const VectorXd& evals = inceig.smallest_eigenvalues();
    int neval = inceig.num_computed_smallest();
    double shrink;
    int loc = shrink_min(evals.data(), neval, penalty, shrink);
    // If loc < 0, then all eigenvalues of x are >= 0, no shrinkage is needed
    if(loc < 0)
        return x;

    // Extend the search range if the shrinkage does not happen in computed eigenvalues
    if(loc >= neval - 1)
    {
        for(int i = 0; i < max_try; i++)
        {
            inceig.compute_next_smallest(inc);
            const VectorXd& evals = inceig.smallest_eigenvalues();
            neval = inceig.num_computed_smallest();
            loc = shrink_min(evals.data(), neval, penalty, shrink);
            if(loc <= neval - 2)
                break;
        }
    }

    // Compute the final result
    NumericMatrix res = Rcpp::clone(x);
    Eigen::Map<MatrixXd> res_(res.begin(), n, n);
    // Difference of eigenvalues
    VectorXd evals_delta = shrink - inceig.smallest_eigenvalues().head(loc + 1).array();
    // Eigenvectors
    inceig.compute_eigenvectors(0, loc + 1);
    const MatrixXd& evecs = inceig.smallest_eigenvectors();
    res_.noalias() += evecs.leftCols(loc + 1) * evals_delta.asDiagonal() * evecs.leftCols(loc + 1).transpose();
    return res;
}

// Proximal operator of alpha1 * f(x) + alpha2 * f2(x)
// f1(x) = max(0, eigmax(x) - 1), f2(x) = max(0, -eigmin(x))
//
// min  alpha1 * f1(X) + alpha2 * f2(X) + 0.5 * ||X - A||_F^2
// [[Rcpp::export]]
NumericMatrix prox_eigminmax(
    NumericMatrix A, double alpha_lg, double alpha_sm,
    int init_neval_lg = 10, int inc_lg = 10,
    int init_neval_sm = 10, int inc_sm = 10, int max_try = 10
)
{
    const int n = A.nrow();
    Eigen::Map<const MatrixXd> A_(A.begin(), n, n);
    NumericMatrix res = Rcpp::clone(A);
    Eigen::Map<MatrixXd> res_(res.begin(), n, n);
    prox_eigs_impl(A_, alpha_lg, alpha_sm, res_,
                   init_neval_lg, inc_lg, init_neval_sm, inc_sm, max_try);
    return res;
}


/*
 set.seed(123)
 n = 100
 evals = sort(runif(n, -2, 2), decreasing = TRUE)
 print(evals)

 evecs = qr.Q(qr(matrix(rnorm(n^2), n)))
 x = evecs %*% diag(evals) %*% t(evecs)
 penalty = 1.0

 res1 = gradfps:::prox_eigmax(x, penalty)
 head(evals1 <- eigen(res1, symmetric = TRUE, only.values = TRUE)$values, 20)
 res2 = gradfps:::prox_eigmin(x, penalty)
 tail(evals2 <- eigen(res2, symmetric = TRUE, only.values = TRUE)$values, 20)
 sum(evals1 - evals)
 sum(evals2 - evals)

 thresh1 = gradfps:::thresh_eigmax(x, penalty)
 thresh2 = gradfps:::thresh_eigmin(x, penalty)
 max(abs(res1 - (x + thresh1)))
 max(abs(res2 - (x + thresh2)))

 penalty_sm = 1.234
 penalty_lg = 1.567
 res1 = gradfps:::prox_eigmin(x, penalty_sm)
 res1 = gradfps:::prox_eigmax(res1, penalty_lg)
 res2 = gradfps:::prox_eigmax(x, penalty_lg)
 res2 = gradfps:::prox_eigmin(res2, penalty_sm)
 res3 = gradfps:::prox_eigminmax(x, penalty_lg, penalty_sm)
 max(abs(res1 - res3))
 max(abs(res2 - res3))
 */
