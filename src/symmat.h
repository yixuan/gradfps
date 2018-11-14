#ifndef FASTFPS_SYM_MAT_H
#define FASTFPS_SYM_MAT_H

#include "common.h"

// Define a structure that represents a symmetric matrix with the following properties:
// (a) Only read and write the lower triangular part
// (b) Dimension is variable by referencing the topleft corner of the matrix, so that
//     memory does not need to be reallocated
class SymMat
{
private:
    typedef Eigen::VectorXd Vector;
    typedef Eigen::MatrixXd Matrix;
    typedef Eigen::Map<Vector> MapVec;
    typedef Eigen::Map<const Vector> MapConstVec;

    int     m_max_n;
    Matrix  m_data;
    int     m_n;

public:
    SymMat() :
        m_max_n(0), m_n(0)
    {}

    SymMat(int max_n, bool allocate = false) :
        m_max_n(max_n), m_n(max_n)
    {
        if(allocate)
            m_data.resize(m_max_n, m_max_n);
    }

    // For debugging
    inline void print() const
    {
        const int n = std::min(m_n, 10);
        Rcpp::Rcout << m_data.topLeftCorner(n, n) << std::endl;
    }

    // Take over an existing matrix
    inline void swap(Matrix& other)
    {
        if(other.rows() != m_max_n || other.cols() != m_max_n)
            throw std::invalid_argument("matrix sizes do not match");

        m_data.swap(other);
    }

    // Swap with another SymMat
    inline void swap(SymMat& other)
    {
        if(other.dim() != m_n || other.max_dim() != m_max_n)
            throw std::invalid_argument("matrix sizes do not match");

        m_data.swap(other.m_data);
    }

    // Access to data
    inline double* data() { return m_data.data(); }
    inline const double* data() const { return m_data.data(); }
    inline Matrix& storage() { return m_data; }
    inline const Matrix& storage() const { return m_data; }

    // Reference to the (i, j) element
    inline double& ref(int i, int j)
    {
        return m_data.data()[j * m_max_n + i];
    }

    // Dimensions
    inline int max_dim() const { return m_max_n; }
    inline int dim() const { return m_n; }

    // Resize
    // Set max size
    inline void set_max_dim(int max_n, bool resize = false)
    {
        m_max_n = max_n;
        if(resize)
            m_data.resize(m_max_n, m_max_n);
    }
    // Set actual size
    inline void set_dim(int n)
    {
        if(n > m_max_n)
            throw std::invalid_argument("n exceeds the maximum size");

        m_n = n;
    }

    // Scaling
    // x *= s
    inline void scale(double s)
    {
        m_data.topLeftCorner(m_n, m_n).triangularView<Eigen::Lower>() *= s;
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

    // Set diagonal elements
    // x.diag() = alpha * x.diag() + beta * v
    inline void diag(double alpha, double beta, const Vector& v)
    {
        for(int i = 0; i < m_n; i++)
            ref(i, i) = alpha * ref(i, i) + beta * v[i];
    }

    // Trace
    inline double trace() const
    {
        return m_data.diagonal().head(m_n).sum();
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

    // Eigen solver operators
    inline int rows() const { return m_n; }
    inline int cols() const { return m_n; }
    inline void perform_op(const double* x_in, double* y_out) const
    {
        MapConstVec x(x_in,  m_n);
        MapVec      y(y_out, m_n);
        y.noalias() = m_data.topLeftCorner(m_n, m_n).selfadjointView<Eigen::Lower>() * x;
    }
};


#endif  // FASTFPS_SYM_MAT_H
