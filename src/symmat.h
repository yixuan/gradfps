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

    int     m_lead_dim;  // leading dimension
    Matrix  m_data;      // storage
    int     m_n;         // actual dimension

public:
    SymMat() :
        m_lead_dim(0), m_n(0)
    {}

    SymMat(int max_n, bool allocate = false) :
        m_lead_dim(max_n), m_n(max_n)
    {
        if(allocate)
            m_data.resize(m_lead_dim, m_lead_dim);
    }

    // For debugging
    inline void print() const
    {
        const int n = std::min(m_n, 10);
        Rcpp::Rcout << m_data.topLeftCorner(n, n) << std::endl;
    }

    // Take over an existing matrix
    // Leading dimension is set to the new size
    // Actual dimension remains the same
    inline void swap(Matrix& other)
    {
        if(other.rows() < m_lead_dim || other.rows() != other.cols())
            throw std::invalid_argument("matrix sizes do not match");

        m_lead_dim = other.rows();
        m_data.swap(other);
    }

    // Swap with another SymMat
    // Leading and actual dimensions should match
    inline void swap(SymMat& other)
    {
        if(other.dim() != m_n || other.lead_dim() != m_lead_dim)
            throw std::invalid_argument("matrix sizes do not match");

        m_data.swap(other.m_data);
    }

    // Access to data
    inline double* data() { return m_data.data(); }
    inline const double* data() const { return m_data.data(); }

    // Reference to the (i, j) element
    inline double& ref(int i, int j)
    {
        return m_data.data()[j * m_lead_dim + i];
    }

    // Dimensions
    inline int lead_dim() const { return m_lead_dim; }
    inline int dim() const { return m_n; }

    // Resize
    // Set actual size
    inline void set_dim(int n)
    {
        if(n > m_lead_dim)
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

            x_col_begin += m_lead_dim;
            x_col_end   += m_lead_dim;
            z_col_begin += zmat.m_lead_dim;
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

            x_col_begin += m_lead_dim;
            x_col_end   += m_lead_dim;
            y_col_begin += ymat.m_lead_dim;
            z_col_begin += zmat.m_lead_dim;
        }
    }

    // Shift diagonal elements by a constant
    // x.diag() += shift
    inline void diag_add(double shift)
    {
        for(int i = 0; i < m_n; i++)
            ref(i, i) += shift;
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

        for(int j = 0; j < m_n; j++, col_begin += m_lead_dim, col_end += m_lead_dim)
        {
            x = col_begin + j;
            diag += (*x) * (*x);
            x = x + 1;

            for(; x < col_end; x++)
                off_diag += (*x) * (*x);
        }

        return std::sqrt(diag + 2 * off_diag);
    }

    // Eigen solver operator - computing matrix-vector multiplication
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
