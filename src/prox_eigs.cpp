#include "inceig_tridiag.h"

using Rcpp::NumericMatrix;

// Assume that x = (x[0], x[1], ..., x[n-1]) is in decreasing order
// Shrink x to x' = (s, s, ..., s, x[i+1], ..., x[n-1]), such that
// s >= 1, s >= x[i+1], and the total shrinkage amount equals `penalty`, i.e.,
// sum((x[0], ..., x[i]) - s) == penalty
//
// If i == -1, then all elements of x are <= 1
// If i == n-1, then we fail to satisfy s >= x[i+1]
inline int shrink_max(const double* x, int n, double penalty, double& shrink)
{
    int i = -1;
    if(x[0] <= 1.0)
        return i;

    double cumsum = 0.0;
    for(i = 0; i < n - 1; i++)
    {
        cumsum += x[i];
        shrink = std::max(1.0, (cumsum - penalty) / double(i + 1));
        if(shrink >= x[i + 1])
            return i;
    }

    // We fail to satisfy s >= x[i+1], use the best approximation
    shrink = (cumsum + x[n - 1]) / double(n);
    return n - 1;
}

// Assume that x = (x[0], x[1], ..., x[n-1]) is in increasing order
// Shrink x to x' = (s, s, ..., s, x[i+1], ..., x[n-1]), such that
// s <= 0, s <= x[i+1], and the total shrinkage amount equals `penalty`, i.e.,
// sum(s - (x[0], ..., x[i])) == penalty
//
// If i == 1, then all elements of x are >= 0
// If i == n-1, then we fail to satisfy s <= x[i+1]
inline int shrink_min(const double* x, int n, double penalty, double& shrink)
{
    int i = -1;
    if(x[0] >= 0.0)
        return i;

    double cumsum = 0.0;
    for(i = 0; i < n - 1; i++)
    {
        cumsum += x[i];
        shrink = std::min(0.0, (cumsum + penalty) / double(i + 1));
        if(shrink <= x[i + 1])
            return i;
    }

    // We fail to satisfy s <= x[i+1], use the best approximation
    shrink = (cumsum + x[n - 1]) / double(n);
    return n - 1;
}

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
// [[Rcpp::export]]
NumericMatrix prox_eigminmax(
    NumericMatrix x, double penalty_sm, double penalty_lg,
    int init_neval_sm = 10, int inc_sm = 10,
    int init_neval_lg = 10, int inc_lg = 10, int max_try = 10
)
{
    const int n = x.nrow();
    Eigen::Map<const MatrixXd> mat(x.begin(), n, n);

    // Compute initial eigenvalues
    IncrementalEig inceig;
    inceig.init(mat, init_neval_lg + inc_lg * max_try, init_neval_lg,
                     init_neval_sm + inc_sm * max_try, init_neval_sm);

    // Test stopping criterion for largest eigenvalues
    const VectorXd& evals_lg = inceig.largest_eigenvalues();
    int neval_lg = inceig.num_computed_largest();
    double shrink_lg;
    int loc_lg = shrink_max(evals_lg.data(), neval_lg, penalty_lg, shrink_lg);

    // Extend the search range if the shrinkage does not happen in computed eigenvalues
    if(loc_lg >= neval_lg - 1)
    {
        for(int i = 0; i < max_try; i++)
        {
            inceig.compute_next_largest(inc_lg);
            const VectorXd& evals_lg = inceig.largest_eigenvalues();
            neval_lg = inceig.num_computed_largest();
            loc_lg = shrink_max(evals_lg.data(), neval_lg, penalty_lg, shrink_lg);
            if(loc_lg <= neval_lg - 2)
                break;
        }
    }

    // Test stopping criterion for smallest eigenvalues
    const VectorXd& evals_sm = inceig.smallest_eigenvalues();
    int neval_sm = inceig.num_computed_smallest();
    double shrink_sm;
    int loc_sm = shrink_min(evals_sm.data(), neval_sm, penalty_sm, shrink_sm);

    // Extend the search range if the shrinkage does not happen in computed eigenvalues
    if(loc_sm >= neval_sm - 1)
    {
        for(int i = 0; i < max_try; i++)
        {
            inceig.compute_next_smallest(inc_sm);
            const VectorXd& evals_sm = inceig.smallest_eigenvalues();
            neval_sm = inceig.num_computed_smallest();
            loc_sm = shrink_min(evals_sm.data(), neval_sm, penalty_sm, shrink_sm);
            if(loc_sm <= neval_sm - 2)
                break;
        }
    }

    // Compute the final result
    NumericMatrix res = Rcpp::clone(x);
    Eigen::Map<MatrixXd> res_(res.begin(), n, n);
    inceig.compute_eigenvectors(loc_lg + 1, loc_sm + 1);
    // Only shrink largest eigenvalues if loc_lg >= 0
    if(loc_lg >= 0)
    {
        // Difference of eigenvalues
        VectorXd evals_delta = shrink_lg - inceig.largest_eigenvalues().head(loc_lg + 1).array();
        // Eigenvectors
        const MatrixXd& evecs = inceig.largest_eigenvectors();
        res_.noalias() += evecs.leftCols(loc_lg + 1) * evals_delta.asDiagonal() * evecs.leftCols(loc_lg + 1).transpose();

    }
    // Only shrink smallest eigenvalues if loc_sm >= 0
    if(loc_sm >= 0)
    {
        // Difference of eigenvalues
        VectorXd evals_delta = shrink_sm - inceig.smallest_eigenvalues().head(loc_sm + 1).array();
        // Eigenvectors
        const MatrixXd& evecs = inceig.smallest_eigenvectors();
        res_.noalias() += evecs.leftCols(loc_sm + 1) * evals_delta.asDiagonal() * evecs.leftCols(loc_sm + 1).transpose();
    }

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
 res3 = gradfps:::prox_eigminmax(x, penalty_sm, penalty_lg)
 max(abs(res1 - res3))
 max(abs(res2 - res3))
 */
