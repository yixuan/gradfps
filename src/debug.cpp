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
