#ifndef GRADFPS_SPARSE_MAT_H
#define GRADFPS_SPARSE_MAT_H

#include <xsimd/xsimd.hpp>
#include "common.h"
#include "symmat.h"

// Create a SIMD vector v = x[ind[0]], x[ind[1]], x[ind[2]], ...
template <int simd_size>
inline xsimd::batch<double, simd_size> gather(const double* x, const int* ind)
{
    alignas(simd_size * sizeof(double)) double data[simd_size];
    for(int i = 0; i < simd_size; i++)
        data[i] = x[ind[i]];
    return xsimd::load_aligned(data);
}

// Add a SIMD vector v to x[ind[0]], x[ind[1]], x[ind[2]], ...
template <int simd_size>
inline void scatter_add(const xsimd::batch<double, simd_size>& v, double* x, const int* ind)
{
    alignas(simd_size * sizeof(double)) double data[simd_size];
    v.store_aligned(data);
    for(int i = 0; i < simd_size; i++)
        x[ind[i]] += data[i];
}

// Specialization for 256-bit vector
#ifdef __AVX__

template <>
inline xsimd::batch<double, 4> gather<4>(const double* x, const int* ind)
{
    return xsimd::batch<double, 4>(x[ind[0]], x[ind[1]], x[ind[2]], x[ind[3]]);
}

template <>
inline void scatter_add<4>(const xsimd::batch<double, 4>& v, double* x, const int* ind)
{
    x[ind[0]] += v[0];
    x[ind[1]] += v[1];
    x[ind[2]] += v[2];
    x[ind[3]] += v[3];
}

// Specialization for 128-bit vector
#elif defined(__SSE__)

template <>
inline xsimd::batch<double, 2> gather<2>(const double* x, const int* ind)
{
    return xsimd::batch<double, 2>(x[ind[0]], x[ind[1]]);
}

template <>
inline void scatter_add<2>(const xsimd::batch<double, 2>& v, double* x, const int* ind)
{
    x[ind[0]] += v[0];
    x[ind[1]] += v[1];
}

#endif



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
        // MapConstVec x(x_in,  m_n);
        // MapVec      y(y_out, m_n);
        // y.noalias() = to_spmat().selfadjointView<Eigen::Lower>() * x;

        typedef xsimd::batch<double, xsimd::simd_type<double>::size> vec;
        const int simd_size = xsimd::simd_type<double>::size;

        // Zero out y vector
        std::fill(y_out, y_out + m_n, 0.0);

        const double* elem_ptr = &m_x[0];
        const int* row_id_ptr = &m_i[0];
        const int* p_ptr = &m_p[0];
        // For each nonzero value, get the (row, col, value) triplet
        for(int col_id = 0; col_id < m_n; col_id++)
        {
            const int col_nnz = p_ptr[col_id + 1] - p_ptr[col_id];
            const double* col_end = elem_ptr + col_nnz;
            const double x_in_col = x_in[col_id];

            // We know that row_id >= col_id if the matrix is constructed
            // from soft_thresh(), so we can first test the presence of
            // diagonal elements, and then process off-diagonal elements

            // Diagonal elements
            if(*row_id_ptr == col_id)
            {
                // Add value to the y vector
                y_out[col_id] += (*elem_ptr) * x_in_col;
                elem_ptr++;
                row_id_ptr++;
            }

            // Off-diagonal elements
            const int len = col_end - elem_ptr;
            const int vec_size = len - len % simd_size;
            const double* simd_end = elem_ptr + vec_size;
            vec x_in_col_simd = xsimd::set_simd(x_in_col);
            double y_out_col = 0.0;
            for( ; elem_ptr < simd_end; elem_ptr += simd_size, row_id_ptr += simd_size)
            {
                vec elem_simd = xsimd::load_unaligned(elem_ptr);
                vec prod = elem_simd * x_in_col_simd;
                scatter_add<simd_size>(prod, y_out, row_id_ptr);
                vec x_in_row_simd = gather<simd_size>(x_in, row_id_ptr);
                y_out_col += xsimd::hadd(elem_simd * x_in_row_simd);
            }
            for( ; elem_ptr < col_end; elem_ptr++, row_id_ptr++)
            {
                // Add value to the y vector
                y_out[*row_id_ptr] += (*elem_ptr) * x_in_col;
                // For off-diagonal elements, do a symmetric update
                y_out_col += (*elem_ptr) * x_in[*row_id_ptr];
            }
            y_out[col_id] += y_out_col;
        }
    }
};


#endif  // GRADFPS_SPARSE_MAT_H
