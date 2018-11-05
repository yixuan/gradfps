#ifndef FASTFPS_EIGENVALUE_H
#define FASTFPS_EIGENVALUE_H

#include "common.h"
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

// proj = V * V', where V contains the k eigenvectors associated with the largest eigenvalues
inline MatrixXd eigs_dense_largest_spectra(
    const MapMat& x, int k, double eps = 1e-3
)
{
    const int ncv = std::max(10, 2 * k + 1);
    Spectra::DenseSymMatProd<double> op(x);
    Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, k, ncv);
    eigs.init();
    eigs.compute(1000, eps);
    return eigs.eigenvectors();
}

// Largest and smallest eigenvalues of a dense matrix x
inline void eigs_dense_both_ends_spectra(
        const MatrixXd& x, VectorXd& evals, double eps = 1e-6
)
{
    Spectra::DenseSymMatProd<double> op(x);
    Spectra::SymEigsSolver< double, Spectra::BOTH_ENDS, Spectra::DenseSymMatProd<double> > eigs(&op, 2, 10);
    eigs.init();
    eigs.compute(1000, eps);
    evals.noalias() = eigs.eigenvalues();
}

// Largest and smallest eigenvalues of a sparse matrix xsp
inline void eigs_sparse_both_ends_spectra(
    const SpMat& xsp, VectorXd& evals, MatrixXd& evecs, double eps = 1e-6
)
{
    Spectra::SparseSymMatProd<double> op(xsp);
    Spectra::SymEigsSolver< double, Spectra::BOTH_ENDS, Spectra::SparseSymMatProd<double> > eigs(&op, 2, 10);
    eigs.init();
    eigs.compute(1000, eps);
    evals.noalias() = eigs.eigenvalues();
    evecs.noalias() = eigs.eigenvectors();
}

#endif // FASTFPS_EIGENVALUE_H
