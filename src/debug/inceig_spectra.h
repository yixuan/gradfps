#ifndef GRADFPS_INCEIG_SPECTRA_H
#define GRADFPS_INCEIG_SPECTRA_H

#include "common.h"
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include "walltime.h"

class IncrementalEig
{
private:
    typedef Eigen::Map<VectorXd> MapVec;
    typedef Eigen::Map<const VectorXd> MapConstVec;
    typedef Eigen::Map<const MatrixXd> MapConstMat;
    typedef Eigen::Ref<const MatrixXd> RefConstMat;

    int           m_n;
    MatrixXd      m_deflate;

    VectorXd      m_evals;
    MatrixXd      m_evecs;

    int           m_max_evals;
    int           m_init_evals;
    int           m_num_computed;

    double        m_shift;
    Eigen::LLT<MatrixXd> m_fac;

public:
    IncrementalEig() {}

    // 1. Set the size of the problem
    // 2. Compute initial `init_evals` eigenvalues
    // 3. Cholesky factorization for the shift-and-invert mode
    inline void init(const RefConstMat& mat, int max_evals, int init_evals)
    {
        // 1. Set the size of the problem
        m_max_evals = max_evals;
        m_init_evals = init_evals;
        m_num_computed = 0;

        m_n = mat.rows();
        m_deflate.resize(m_n, m_n);
        m_evals.resize(m_max_evals);
        m_evecs.resize(m_n, m_max_evals);

        // 2. Compute initial `init_evals` eigenvalues
        double t1 = get_wall_time();
        Spectra::DenseSymMatProd<double> op(mat);
        Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> >
            eigs(&op, init_evals, std::min(m_n, std::max(20, init_evals * 2 + 1)));
        eigs.init();
        eigs.compute(1000, 1e-6);

        // Store computed eigenvalues and eigenvectors
        m_evals.head(init_evals).noalias() = eigs.eigenvalues();
        m_evecs.leftCols(init_evals).noalias() = eigs.eigenvectors();
        m_num_computed += init_evals;
        double t2 = get_wall_time();

        // 3. Cholesky factorization for the shift-and-invert mode
        // lambda' = 1 / (shift - lambda)
        m_shift = m_evals[init_evals - 1];
        // For our problem, we stop when we get zero eigenvalues
        if(m_shift < 1e-6)
            return;

        // Deflate the input matrix
        m_deflate.noalias() = mat;
        m_deflate.noalias() -= m_evecs.leftCols(init_evals) * m_evals.head(init_evals).asDiagonal() * m_evecs.leftCols(init_evals).transpose();
        double t3 = get_wall_time();

        // Cholesky decomposition
        m_fac.compute(m_shift * MatrixXd::Identity(m_n, m_n) - m_deflate);
        double t4 = get_wall_time();

        // ::Rprintf("time1 = %f, time2 = %f, time3 = %f\n", t2 - t1, t3 - t2, t4 - t3);

        if(m_fac.info() != Eigen::Success)
            throw std::logic_error("IncrementalEig: factorization failed");
    }

    int rows() const { return m_n; }
    int cols() const { return m_n; }

    // X - Gk * Lk * Gk'
    inline void perform_op(const double* x_in, double* y_out) const
    {
        MapConstVec x(x_in, m_n);
        MapVec y(y_out, m_n);
        y.noalias() = m_fac.solve(x);
    }

    inline int compute_next(int inc_evals)
    {
        if(m_num_computed + inc_evals > m_max_evals)
            throw std::logic_error("maximum number of eigenvalues computed");

        double t1 = get_wall_time();
        Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, IncrementalEig>
            eigs(this, inc_evals, std::min(m_n, inc_evals * 2 + 1));
        eigs.init();
        eigs.compute(1000, 1e-6);

        // Store computed eigenvalues and eigenvectors
        m_evals.segment(m_num_computed, inc_evals).array() = m_shift - 1.0 / eigs.eigenvalues().array();
        m_evecs.block(0, m_num_computed, m_n, inc_evals).noalias() = eigs.eigenvectors();
        double t2 = get_wall_time();

        // Deflate the input matrix
        m_deflate.noalias() -= m_evecs.block(0, m_num_computed, m_n, inc_evals) *
            m_evals.segment(m_num_computed, inc_evals).asDiagonal() *
            m_evecs.block(0, m_num_computed, m_n, inc_evals).transpose();
        m_num_computed += inc_evals;
        double t3 = get_wall_time();

        // Cholesky decomposition
        m_shift = m_evals[m_num_computed - 1];
        // For our problem, we stop when we get zero eigenvalues
        if(m_shift < 1e-6)
            return eigs.num_operations();

        m_fac.compute(m_shift * MatrixXd::Identity(m_n, m_n) - m_deflate);
        double t4 = get_wall_time();

        // ::Rprintf("time1 = %f, time2 = %f, time3 = %f\n", t2 - t1, t3 - t2, t4 - t3);

        return eigs.num_operations();
    }

    const int num_computed() const { return m_num_computed; }
    const VectorXd& eigenvalues() const { return m_evals; }
    const MatrixXd& eigenvectors() const { return m_evecs; }
};


#endif // GRADFPS_INCEIG_SPECTRA_H
