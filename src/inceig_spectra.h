#ifndef GRADFPS_INCEIG_SPECTRA_H
#define GRADFPS_INCEIG_SPECTRA_H

#include "common.h"
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>

class IncrementalEig
{
private:
    typedef Eigen::Map<VectorXd> MapVec;
    typedef Eigen::Map<const VectorXd> MapConstVec;
    typedef Eigen::Map<const MatrixXd> MapConstMat;
    typedef Eigen::Ref<const MatrixXd> RefConstMat;

    int           m_n;

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
        m_evals.resize(m_max_evals);
        m_evecs.resize(m_n, m_max_evals);

        // 2. Compute initial `init_evals` eigenvalues
        Spectra::DenseSymMatProd<double> op(mat);
        Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> >
            eigs(&op, init_evals, std::min(m_n, std::max(20, init_evals * 2 + 1)));
        eigs.init();
        eigs.compute(1000, 1e-6);

        m_evals.head(init_evals).noalias() = eigs.eigenvalues();
        m_evecs.leftCols(init_evals).noalias() = eigs.eigenvectors();
        m_num_computed += init_evals;

        // 3. Cholesky factorization for the shift-and-invert mode
        // lambda' = 1 / (shift - lambda)
        m_shift = m_evals[init_evals - 1];
        if(m_shift < 1e-6)
            return;

        MatrixXd mat_fac(m_n, m_n);
        mat_fac.noalias() = m_evecs.leftCols(init_evals) * m_evals.head(init_evals).asDiagonal() * m_evecs.leftCols(init_evals).transpose();
        mat_fac.noalias() -= mat;
        mat_fac.diagonal().array() += m_shift;

        m_fac.compute(mat_fac);
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

        if(m_num_computed <= m_init_evals)
        {
            y.noalias() = m_fac.solve(x);
            return;
        }

        VectorXd Gkx = m_evecs.block(0, m_init_evals, m_n, m_num_computed - m_init_evals).transpose() * x;
        Gkx.array() /= (m_shift - m_evals.segment(m_init_evals, m_num_computed - m_init_evals).array());
        y.noalias() = m_fac.solve(x) - m_evecs.block(0, m_init_evals, m_n, m_num_computed - m_init_evals) * Gkx;
    }

    inline int compute_next(int inc_evals)
    {
        if(m_num_computed + inc_evals > m_max_evals)
            throw std::logic_error("maximum number of eigenvalues computed");

        Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, IncrementalEig>
            eigs(this, inc_evals, std::min(m_n, inc_evals * 2 + 1));
        eigs.init();
        eigs.compute(1000, 1e-6);

        m_evals.segment(m_num_computed, inc_evals).array() = m_shift - 1.0 / eigs.eigenvalues().array();
        m_evecs.block(0, m_num_computed, m_n, inc_evals).noalias() = eigs.eigenvectors();
        m_num_computed += inc_evals;

        return eigs.num_operations();
    }

    const int num_computed() const { return m_num_computed; }
    const VectorXd& eigenvalues() const { return m_evals; }
    const MatrixXd& eigenvectors() const { return m_evecs; }
};


#endif // GRADFPS_INCEIG_SPECTRA_H
