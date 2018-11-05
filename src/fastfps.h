#ifndef FASTFPS_FASTFPS_H
#define FASTFPS_FASTFPS_H

#include "common.h"

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

// Apply the soft-thresholding operator on a symmetrix matrix x,
// and return a sparse matrix. Only the lower triangular part of x
// is referenced.
//
// NOTE: the returned sparse matrix is not compressed, and only the lower
//       triangular part contains values.
template <typename MatType>
inline void soft_thresh_sparse(
    const MatType& x, double lambda, SpMat& res
)
{
    const int p = x.rows();
    res.resize(p, p);
    res.setZero();

    for(int j = 0; j < p; j++)
    {
        for(int i = j; i < p; i++)
        {
            const double xij = x.coeff(i, j);
            if(xij > lambda)
            {
                res.insert(i, j) = xij - lambda;
            } else if(xij < -lambda)
            {
                res.insert(i, j) = xij + lambda;
            }
        }
    }
}

// Apply a rank-2 update on a sparse matrix x.
// Only the lower triangular part is referenced.
// res <- x + a1 * v1 * v1' + a2 * v2 * v2'
//
// NOTE: only the lower triagular part is written.
template <typename MatType>
inline void rank2_update_sparse(
    const SpMat& x, double a1, const RefVec& v1, double a2, const RefVec& v2, MatType& res
)
{
    const int p = x.rows();
    res.resize(p, p);

    const double a1_abs = std::abs(a1);
    const double a2_abs = std::abs(a2);
    const double* v1p = v1.data();
    const double* v2p = v2.data();

    const double eps = 1e-6;

    // If both a1 and a2 are zero, simply add x to a zero matrix
    if(a1_abs <= eps && a2_abs <= eps)
    {
        res.setZero();
    }
    // a1 == 0, a2 != 0
    if(a1_abs <= eps && a2_abs > eps)
    {
        for(int j = 0; j < p; j++)
        {
            const double v2j = a2 * v2p[j];
            for(int i = j; i < p; i++)
            {
                res.coeffRef(i, j) = v2j * v2p[i];
            }
        }
    }
    // a1 != 0, a2 == 0
    if(a1_abs > eps && a2_abs <= eps)
    {
        for(int j = 0; j < p; j++)
        {
            const double v1j = a1 * v1p[j];
            for(int i = j; i < p; i++)
            {
                res.coeffRef(i, j) = v1j * v1p[i];
            }
        }
    }
    // a1 != 0, a2 != 0
    if(a1_abs > eps && a2_abs > eps)
    {
        for(int j = 0; j < p; j++)
        {
            const double v1j = a1 * v1p[j];
            const double v2j = a2 * v2p[j];
            for(int i = j; i < p; i++)
            {
                res.coeffRef(i, j) = v1j * v1p[i] + v2j * v2p[i];
            }
        }
    }

    // Add the sparse matrix
    const int jx = x.outerSize();
    for(int j = 0; j < jx; j++)
    {
        for(SpMat::InnerIterator it(x, j); it; ++it)
        {
            res.coeffRef(it.row(), j) += it.value();
        }
    }
}


#endif  // FASTFPS_FASTFPS_H
