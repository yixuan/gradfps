#ifndef FASTFPS_FASTFPS_H
#define FASTFPS_FASTFPS_H

#include "common.h"
#include "sparsemat.h"
#include "eigenvalue.h"

// Initial guess using partial eigen decomposition
inline void initial_guess(const MapMat& S, int d, MatrixXd& x)
{
    MatrixXd evecs = eigs_dense_largest_spectra(S, d);
    x.noalias() = evecs * evecs.transpose();
}

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

// Apply a rank-2 update on a sparse matrix x.
// Only the lower triangular part is read and written
// res <- x + a1 * v1 * v1' + a2 * v2 * v2'
inline void rank2_update_sparse(
    const dgCMatrix& x, double a1, const RefVec& v1, double a2, const RefVec& v2, MatrixXd& res
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
    } else if(a1_abs <= eps && a2_abs > eps) {          // a1 == 0, a2 != 0
        for(int j = 0; j < p; j++)
        {
            const double v2j = a2 * v2p[j];
            for(int i = j; i < p; i++)
            {
                res.coeffRef(i, j) = v2j * v2p[i];
            }
        }
    } else if(a1_abs > eps && a2_abs <= eps) {          // a1 != 0, a2 == 0
        for(int j = 0; j < p; j++)
        {
            const double v1j = a1 * v1p[j];
            for(int i = j; i < p; i++)
            {
                res.coeffRef(i, j) = v1j * v1p[i];
            }
        }
    } else {                                            // a1 != 0, a2 != 0
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
    x.add_to(res.data());
}


#endif  // FASTFPS_FASTFPS_H
