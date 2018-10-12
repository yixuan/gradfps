#include "fastfps.h"

using Rcpp::NumericMatrix;

// [[Rcpp::export]]
SpMat soft_thresh(NumericMatrix x, double lambda)
{
    const int n = x.nrow();
    const int p = x.ncol();
    if(n != p)
        Rcpp::stop("x must be square");

    MapMat xmat(x.begin(), n, p);
    SpMat res(p, p);
    soft_thresh_sparse(xmat, lambda, res);

    // We need to compress the sparse matrix in order to use it in R
    res.makeCompressed();

    return res;
}

/*
library(Matrix)
set.seed(123)
x = matrix(rnorm(100), 10)
x = x + t(x)
y = soft_thresh(x, 0.5)
y
*/

// [[Rcpp::export]]
NumericMatrix rank2_update(MapSpMat x, double a1, MapVec v1, double a2, MapVec v2)
{
    const int n = x.rows();
    const int p = x.cols();
    if(n != p)
        Rcpp::stop("x must be square");

    NumericMatrix xnew(p, p);
    MapMat res(xnew.begin(), p, p);
    rank2_update_sparse(x, a1, v1, a2, v2, res);

    // Also write upper triangular part
    for(int j = 1; j < p; j++)
    {
        for(int i = 0; i < j; i++)
        {
            res.coeffRef(i, j) = res.coeff(j, i);
        }
    }

    return xnew;
}

/*
library(Matrix)
set.seed(123)
x = matrix(rnorm(100), 10)
x = x + t(x)
y = soft_thresh(x, 0.5)
v1 = rnorm(10)
v2 = rnorm(10)
z = rank2_update(y, 0.1, v1, 0.2, v2)
z
(y + t(y) - diag(diag(y))) + 0.1 * tcrossprod(v1) + 0.2 * tcrossprod(v2)
*/
