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

// x += alpha*x + beta*y + gamma*z
// Only the lower triangular part is read and written
inline void sym_mat_update(
    int p, double* x, const double* y, const double* z,
    double alpha, double beta, double gamma
)
{
    double*       x_col_begin = x;
    const double* x_col_end   = x + p;

    const double* y_col_begin = y;
    const double* y_col_end   = y + p;

    const double* z_col_begin = z;
    const double* z_col_end   = z + p;

    for(int j = 0; j < p; j++)
    {
        x = x_col_begin + j;
        y = y_col_begin + j;
        z = z_col_begin + j;

        for(; x < x_col_end; x++, y++, z++)
            (*x) += alpha * (*x) + beta * (*y) + gamma * (*z);

        x_col_begin += p;
        x_col_end   += p;

        y_col_begin += p;
        y_col_end   += p;

        z_col_begin += p;
        z_col_end   += p;
    }
}

// x += gamma*z
// Only the lower triangular part is read and written
inline void sym_mat_update(
    int p, double* x, const double* z, double gamma
)
{
    double*       x_col_begin = x;
    const double* x_col_end   = x + p;

    const double* z_col_begin = z;
    const double* z_col_end   = z + p;

    for(int j = 0; j < p; j++)
    {
        x = x_col_begin + j;
        z = z_col_begin + j;

        for(; x < x_col_end; x++, z++)
            (*x) += gamma * (*z);

        x_col_begin += p;
        x_col_end   += p;

        z_col_begin += p;
        z_col_end   += p;
    }
}

// Frobenius norm of a symmetric matrix
// Only the lower triangular part is read
inline double sym_mat_norm(int p, const double* x)
{
    const double* col_begin = x;
    const double* col_end   = x + p;
    double diag = 0.0;
    double off_diag = 0.0;

    for(int j = 0; j < p; j++, col_begin += p, col_end += p)
    {
        x = col_begin + j;
        diag += (*x) * (*x);
        x = x + 1;

        for(; x < col_end; x++)
            off_diag += (*x) * (*x);
    }

    return std::sqrt(diag + 2 * off_diag);
}

// FPS objective function: -<S, X> + lambda * ||X||_1
// Only the lower triangular part is read
inline double fps_objfn(int p, const double* x, const double* s, double lambda)
{
    const double* x_col_begin = x;
    const double* x_col_end   = x + p;

    const double* s_col_begin = s;
    const double* s_col_end   = s + p;

    double diag1 = 0.0, diag2 = 0.0;
    double off_diag1 = 0.0, off_diag2 = 0.0;

    for(int j = 0; j < p; j++)
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

        x_col_begin += p;
        x_col_end   += p;

        s_col_begin += p;
        s_col_end   += p;
    }

    return -(diag1 + off_diag1 * 2) + lambda * (diag2 + off_diag2 * 2);
}


#endif  // FASTFPS_FASTFPS_H
