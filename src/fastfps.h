#ifndef FASTFPS_FASTFPS_H
#define FASTFPS_FASTFPS_H

#include "common.h"
#include "sparsemat.h"
#include "symmat.h"
#include "eigenvalue.h"

// Initial guess using partial eigen decomposition
inline void initial_guess(const SymMat& S, int d, SymMat& x)
{
    MatrixXd evecs = eigs_dense_largest_spectra(S, d);
    MatrixXd proj(S.dim(), S.dim());
    proj.noalias() = evecs * evecs.transpose();
    x.swap(proj);
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


#endif  // FASTFPS_FASTFPS_H
