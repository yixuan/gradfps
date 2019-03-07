#include "prox_fantope.h"
#include "inceig.h"

using Rcpp::IntegerVector;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::List;

// [[Rcpp::export]]
MatrixXd prox_fantope(MapMat v, double alpha, MapMat S, int d, int inc, int max_try)
{
    MatrixXd mat = v + alpha * S;
    VectorXd new_evals(inc * max_try);

    IncrementalEig inceig;
    inceig.init(mat, inc * max_try, inc);

    for(int i = 0; i < max_try; i++)
    {
        if(i == max_try)
            Rcpp::Rcout << "=== max_try: " << i << " ===" << std::endl;
        inceig.compute_next();
        const VectorXd& evals = inceig.eigenvalues();
        quadprog_sol(evals.data(), inceig.num_computed(), d, new_evals.data());
        if(std::abs(new_evals[inceig.num_computed() - 1]) < 1e-6)
            break;
    }

    int pos = d;
    const int end = inceig.num_computed();
    for(; pos < end; pos++)
    {
        if(std::abs(new_evals[pos]) < 1e-6)
            break;
    }

    MatrixXd res = inceig.eigenvectors().leftCols(pos) *
        new_evals.head(pos).asDiagonal() *
        inceig.eigenvectors().leftCols(pos).transpose();

    return res;
}

