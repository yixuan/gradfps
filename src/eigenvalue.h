#ifndef FASTFPS_EIGENVALUE_H
#define FASTFPS_EIGENVALUE_H

#include "common.h"
#include "sparsemat.h"
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <primme.h>

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
    const dgCMatrix& xsp, VectorXd& evals, MatrixXd& evecs, double eps = 1e-6
)
{
    dgCMatrixOp op(xsp);
    Spectra::SymEigsSolver<double, Spectra::BOTH_ENDS, dgCMatrixOp> eigs(&op, 2, 10);
    eigs.init();
    eigs.compute(1000, eps);
    evals.noalias() = eigs.eigenvalues();
    evecs.noalias() = eigs.eigenvectors();

    // Rcpp::Rcout << "nop = " << eigs.num_operations() << std::endl;
}





inline void sp_mat_vec(
    void* x, PRIMME_INT* ldx, void* y, PRIMME_INT* ldy,
    int* blockSize, primme_params* primme, int* err)
{
    dgCMatrixOp* op = (dgCMatrixOp*) primme->matrix;

    double* xvec;     /* pointer to i-th input vector x */
    double* yvec;     /* pointer to i-th output vector y */

    for(int i = 0; i < *blockSize; i++)
    {
        xvec = (double*) x + *ldx * i;
        yvec = (double*) y + *ldy * i;

        op->perform_op(xvec, yvec);
    }
    *err = 0;
}

// Largest and smallest eigenvalues of a sparse matrix xsp
template <int N>
inline void eigs_sparse_both_ends_primme(
    const dgCMatrix& xsp, VectorXd& evals, MatrixXd& evecs, double eps = 1e-6
)
{
    const int n = xsp.rows();
    dgCMatrixOp op(xsp);
    double resid[2 * N];

    primme_params primme;
    primme_initialize(&primme);

    primme.matrixMatvec = sp_mat_vec;
    primme.n = n;
    primme.numEvals = N;
    primme.eps = eps;
    primme.target = primme_largest;
    primme.iseed[0] = 0;
    primme.iseed[1] = 1;
    primme.iseed[2] = 2;
    primme.iseed[3] = 3;
    primme.matrix = &op;

    primme_set_method(PRIMME_DEFAULT_MIN_TIME, &primme);
    int ret = dprimme(&evals[0], evecs.data(), &resid[0], &primme);
    // int nops = primme.stats.numMatvecs;

    // Rcpp::Rcout << "nop1 = " << nops;

    primme.target = primme_smallest;
    ret = dprimme(&evals[N], evecs.data() + N * n, &resid[N], &primme);
    // nops = primme.stats.numMatvecs;

    // Rcpp::Rcout << ", nop2 = " << nops << std::endl;

    primme_free(&primme);
}


#endif // FASTFPS_EIGENVALUE_H
