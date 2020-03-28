#ifndef GRADFPS_SPARSE_MAT_H
#define GRADFPS_SPARSE_MAT_H

#include "common.h"
#include "symmat.h"

// Define a structure that mimics the dgCMatrix class in R
// Assume that the matrix is lower-triangular
class dgCMatrix
{
private:
    int m_n;
    std::vector<int> m_i;
    std::vector<int> m_p;
    std::vector<double> m_x;

public:
    dgCMatrix(int n, double reserve_nnz = 0.01) :
        m_n(n)
    {
        m_i.reserve(int(n * n * reserve_nnz));
        m_p.reserve(n + 1);
        m_x.reserve(int(n * n * reserve_nnz));
    }

    inline int rows() const { return m_n; }
    inline int cols() const { return m_n; }

    // Resize matrix
    // If new size is larger, we only need to append zeros to m_p, and existing
    // data can be preseved. Otherwise we destroy the existing data
    inline void resize(int n)
    {
        m_n = n;
        m_p.resize(n + 1);

        if(n < m_n)
        {
            m_i.clear();
            m_x.clear();
            std::fill(m_p.begin(), m_p.end(), 0);
        }
    }

    // Construct the matrix by soft-thresholding a symmetrix matrix x
    inline void soft_thresh(const SymMat& x, double lambda)
    {
        if(x.dim() != m_n)
            throw std::invalid_argument("matrix sizes do not match");

        m_i.clear();
        m_p.clear();
        m_p.push_back(0);
        m_x.clear();

        const double* col_begin = x.data();
        const double* col_end = col_begin + m_n;
        const int xldim = x.lead_dim();
        for(int j = 0; j < m_n; j++, col_begin += xldim, col_end += xldim)
        {
            int col_nnz = 0;
            for(const double* xptr = col_begin + j; xptr < col_end; xptr++)
            {
                if(*xptr > lambda)
                {
                    m_i.push_back(xptr - col_begin);
                    m_x.push_back(*xptr - lambda);
                    col_nnz++;
                } else if(*xptr < -lambda) {
                    m_i.push_back(xptr - col_begin);
                    m_x.push_back(*xptr + lambda);
                    col_nnz++;
                }
            }
            m_p.push_back(m_p.back() + col_nnz);
        }
    }

    // Return a mapped sparse matrix
    inline MapSpMat to_spmat()
    {
        return MapSpMat(m_n, m_n, m_p[m_n], &m_p[0], &m_i[0], &m_x[0]);
    }
    inline MapConstSpMat to_spmat() const
    {
        return MapConstSpMat(m_n, m_n, m_p[m_n], &m_p[0], &m_i[0], &m_x[0]);
    }

    // Add the sparse matrix to a symmetric matrix
    inline void add_to(SymMat& x) const
    {
        if(x.dim() != m_n)
            throw std::invalid_argument("matrix sizes do not match");

        double* col_begin = x.data();
        const int xldim = x.lead_dim();
        const double* dat_ptr = &m_x[0];
        const int* row_ptr = &m_i[0];
        for(int j = 0; j < m_n; j++, col_begin += xldim)
        {
            const int col_nnz = m_p[j + 1] - m_p[j];
            for(int k = 0; k < col_nnz; k++, dat_ptr++, row_ptr++)
            {
                col_begin[*row_ptr] += *dat_ptr;
            }
        }
    }

    // Eigen solver operator - computing matrix-vector multiplication
    inline void perform_op(const double* x_in, double* y_out) const
    {
        MapConstVec x(x_in,  m_n);
        MapVec      y(y_out, m_n);
        y.noalias() = to_spmat().selfadjointView<Eigen::Lower>() * x;
    }
};


#endif  // GRADFPS_SPARSE_MAT_H
