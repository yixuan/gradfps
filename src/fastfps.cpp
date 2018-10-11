#include "fastfps.h"
#include <Spectra/SymEigsSolver.h>

using Rcpp::NumericMatrix;
using Rcpp::List;

using namespace Spectra;

// Initial guess using partial eigen decomposition
inline void initial_guess(const MapMat& S, int d, MapMat& x)
{
    const int ncv = std::max(10, 2 * d + 1);
    DenseSymMatProd<double> op(S);
    SymEigsSolver< double, LARGEST_ALGE, DenseSymMatProd<double> > eigs(&op, d, ncv);
    eigs.init();
    eigs.compute(1000, 0.001);
    MatrixXd evecs = eigs.eigenvectors();
    x.noalias() = evecs * evecs.transpose();
}

// [[Rcpp::export]]
List fastfps(NumericMatrix S, int d, double lambda, int maxiter,
             double alpha0, double mu, double r)
{
    const int n = S.nrow();
    const int p = S.ncol();
    if(n != p)
        Rcpp::stop("S must be square");

    MapMat Smat(S.begin(), p, p);
    NumericMatrix res(p, p);
    MapMat x(res.begin(), p, p);

    // Initial guess -- using partial eigen decomposition
    initial_guess(Smat, d, x);

    std::vector<double> objfn;

    return List::create(
        Rcpp::Named("projection") = x,
        Rcpp::Named("objfn") = objfn
    );
}
