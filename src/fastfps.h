#ifndef FASTFPS_H
#define FASTFPS_H

#include <RcppEigen.h>

typedef Eigen::MatrixXd Matrix;
typedef Eigen::Map<Matrix> MapMat;
typedef Eigen::SparseMatrix<double> SpMat;

// Apply the soft-thresholding operator on a symmetrix matrix x,
// and return a sparse matrix.
// NOTE: the returned sparse matrix is not compressed.
inline void soft_thresh_sparse(
    const MapMat& x, double lambda, SpMat& res
)
{
    const int p = x.rows();
    res.resize(p, p);
    res.setZero();

    for(int j = 0; j < p; j++)
    {
        // Diagonal terms
        const double xii = x.coeff(j, j);
        if(xii > lambda)
        {
            res.insert(j, j) = xii - lambda;
        } else if(xii < -lambda)
        {
            res.insert(j, j) = xii + lambda;
        }

        for(int i = j + 1; i < p; i++)
        {
            const double xij = x.coeff(i, j);
            if(xij > lambda)
            {
                res.insert(i, j) = xij - lambda;
                res.insert(j, i) = xij - lambda;
            } else if(xij < -lambda)
            {
                res.insert(i, j) = xij + lambda;
                res.insert(j, i) = xij + lambda;
            }
        }
    }
}


#endif  // FASTFPS_H
