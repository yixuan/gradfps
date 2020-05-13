#ifndef GRADFPS_PROX_EIGS_H
#define GRADFPS_PROX_EIGS_H

#include "common.h"
#include "inceig_tridiag.h"

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

// Proximal operator of alpha1 * f(x) + alpha2 * f2(x)
// f1(x) = max(0, eigmax(x) - 1), f2(x) = max(0, -eigmin(x))
//
// min  alpha1 * f1(X) + alpha2 * f2(X) + 0.5 * ||X - A||_F^2
inline void prox_eigs_delta_impl(
    RefConstMat A, double penalty_lg, double penalty_sm, RefMat res,
    int init_neval_lg = 10, int inc_lg = 10,
    int init_neval_sm = 10, int inc_sm = 10, int max_try = 10
)
{
    // Compute initial eigenvalues
    IncrementalEig inceig(A.rows());
    inceig.init(A, init_neval_lg + inc_lg * max_try, init_neval_lg,
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
    res.setZero();
    inceig.compute_eigenvectors(loc_lg + 1, loc_sm + 1);
    // Only shrink largest eigenvalues if loc_lg >= 0
    if(loc_lg >= 0)
    {
        // Difference of eigenvalues
        VectorXd evals_delta = shrink_lg - inceig.largest_eigenvalues().head(loc_lg + 1).array();
        // Eigenvectors
        const MatrixXd& evecs = inceig.largest_eigenvectors();
        res.noalias() += evecs.leftCols(loc_lg + 1) * evals_delta.asDiagonal() * evecs.leftCols(loc_lg + 1).transpose();

    }
    // Only shrink smallest eigenvalues if loc_sm >= 0
    if(loc_sm >= 0)
    {
        // Difference of eigenvalues
        VectorXd evals_delta = shrink_sm - inceig.smallest_eigenvalues().head(loc_sm + 1).array();
        // Eigenvectors
        const MatrixXd& evecs = inceig.smallest_eigenvectors();
        res.noalias() += evecs.leftCols(loc_sm + 1) * evals_delta.asDiagonal() * evecs.leftCols(loc_sm + 1).transpose();
    }
}


#endif  // GRADFPS_PROX_EIGS_H
