#ifndef FASTFPS_SYM_MAT_H
#define FASTFPS_SYM_MAT_H

#include "common.h"
#include "sparsemat.h"

// Define a structure that represents a symmetric matrix with the following properties:
// (a) Only read and write the lower triangular part
// (b) Dimension is variable by referencing the topleft corner of the matrix, so that
//     memory does not need to be reallocated
class SymMat
{
private:
    typedef Eigen::MatrixXd Matrix;

    const int m_max_n;
    Matrix    m_data;
    int       m_n;

public:
    SymMat(int max_n, bool allocate = false) :
        m_max_n(max_n), m_n(max_n)
    {
        if(allocate)
            m_data.resize(m_max_n, m_max_n);
    }

    // Take over an existing matrix
    inline void swap(Matrix& other)
    {
        if(other.rows() != m_max_n || other.cols() != m_max_n)
            throw std::invalid_argument("matrix size does not match");
    }

    // Dimensions
    inline int max_rows() const { return m_max_n; }
    inline int rows() const { return m_n; }
    inline void set_dim(int n)
    {
        if(n > m_max_n)
            throw std::invalid_argument("n exceeds the maximum size");

        m_n = n;
    }

    // Reference to the (i, j) element
    inline double& ref(int i, int j)
    {
        return m_data.data()[j * m_max_n + i];
    }

    // Addition - version 1
    // x += gamma*z
    inline void add(double gamma, const SymMat& zmat)
    {
        if(m_n != zmat.m_n)
            throw std::invalid_argument("matrix sizes do not match");

        double*       x = m_data.data();
        double*       x_col_begin = x;
        const double* x_col_end   = x + m_n;

        const double* z = zmat.m_data.data();
        const double* z_col_begin = z;

        for(int j = 0; j < m_n; j++)
        {
            x = x_col_begin + j;
            z = z_col_begin + j;

            for(; x < x_col_end; x++, z++)
                (*x) += gamma * (*z);

            x_col_begin += m_max_n;
            x_col_end   += m_max_n;
            z_col_begin += zmat.m_max_n;
        }
    }

    // Addition - version 2
    // x += alpha*x + beta*y + gamma*z
    inline void add(double alpha, double beta, double gamma, const SymMat& ymat, const SymMat& zmat)
    {
        if(m_n != ymat.m_n || m_n != zmat.m_n)
            throw std::invalid_argument("matrix sizes do not match");

        double*       x = m_data.data();
        double*       x_col_begin = x;
        const double* x_col_end   = x + m_n;

        const double* y = ymat.m_data.data();
        const double* y_col_begin = y;

        const double* z = zmat.m_data.data();
        const double* z_col_begin = z;

        for(int j = 0; j < m_n; j++)
        {
            x = x_col_begin + j;
            y = y_col_begin + j;
            z = z_col_begin + j;

            for(; x < x_col_end; x++, y++, z++)
                (*x) += alpha * (*x) + beta * (*y) + gamma * (*z);

            x_col_begin += m_max_n;
            x_col_end   += m_max_n;
            y_col_begin += ymat.m_max_n;
            z_col_begin += zmat.m_max_n;
        }
    }

    // Apply a rank-r update on a sparse matrix x.
    // Only the lower triangular part is read and written
    // x <- xsp + a1 * v1 * v1' + ... + ar * vr * vr'
    template <int r>
    void rank_r_update_sparse(const dgCMatrix& xsp, const RefVec& a, const RefMat& v)
    {
        if(m_n != xsp.rows())
            throw std::invalid_argument("matrix sizes do not match");

        double vj[r];

        for(int j = 0; j < m_n; j++)
        {
            for(int k = 0; k < r; k++)
            {
                vj[k] = a[k] * v.coeff(j, k);
            }
            for(int i = j; i < m_n; i++)
            {
                double sum = 0.0;
                for(int k = 0; k < r; k++)
                {
                    sum += vj[k] * v.coeff(i, k);
                }
                ref(i, j) = sum;
            }
        }

        // Add the sparse matrix
        xsp.add_to(m_data.data(), m_max_n);
    }

    // Frobenius norm
    inline double norm() const
    {
        const double* x = m_data.data();
        const double* col_begin = x;
        const double* col_end = x + m_n;
        double diag = 0.0;
        double off_diag = 0.0;

        for(int j = 0; j < m_n; j++, col_begin += m_max_n, col_end += m_max_n)
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
    inline double fps_objfn(const SymMat& smat, double lambda) const
    {
        if(m_n != smat.m_n)
            throw std::invalid_argument("matrix sizes do not match");

        const double* x = m_data.data();
        const double* x_col_begin = x;
        const double* x_col_end   = x + m_n;

        const double* s = smat.m_data.data();
        const double* s_col_begin = s;

        double diag1 = 0.0, diag2 = 0.0;
        double off_diag1 = 0.0, off_diag2 = 0.0;

        for(int j = 0; j < m_n; j++)
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

            x_col_begin += m_max_n;
            x_col_end   += m_max_n;
            s_col_begin += smat.m_max_n;
        }

        return -(diag1 + off_diag1 * 2) + lambda * (diag2 + off_diag2 * 2);
    }
};

// Compute matrix-vector multiplication
/*class SymMatOp
{
private:
    const int m_n;
    MapConstSpMat m_mat;

public:
    SymMatOp(const SymMat& mat) :
        m_n(mat.rows()), m_mat(mat.to_spmat())
    {}

    inline int rows() const { return m_n; }
    inline int cols() const { return m_n; }
    inline void perform_op(const double* x_in, double* y_out) const
    {
        MapConstVec x(x_in,  m_n);
        MapVec      y(y_out, m_n);
        y.noalias() = m_mat.selfadjointView<Eigen::Lower>() * x;
    }
};*/


#endif  // FASTFPS_SYM_MAT_H
