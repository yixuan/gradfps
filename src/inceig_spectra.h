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

    const double* m_mat_data;
    int           m_n;

    VectorXd      m_evals;
    MatrixXd      m_evecs;

    int           m_max_evals;
    int           m_inc_evals;
    int           m_num_computed;

public:
    IncrementalEig() {}

    inline void init(const RefConstMat& mat, int max_evals, int inc_evals = 1)
    {
        m_max_evals = max_evals;
        m_inc_evals = inc_evals;
        m_num_computed = 0;

        m_n = mat.rows();
        m_mat_data = mat.data();
        m_evals.resize(m_max_evals);
        m_evecs.resize(m_n, m_max_evals);
    }

    int rows() const { return m_n; }
    int cols() const { return m_n; }

    // X - Gk * Lk * Gk'
    inline void perform_op(const double* x_in, double* y_out) const
    {
        MapConstVec x(x_in, m_n);
        MapVec y(y_out, m_n);
        MapConstMat mat(m_mat_data, m_n, m_n);

        if(m_num_computed < 1)
        {
            y.noalias() = mat * x;
            return;
        }

        VectorXd Gkx = m_evecs.leftCols(m_num_computed).transpose() * x;
        Gkx.array() *= m_evals.head(m_num_computed).array();
        y.noalias() = mat * x - m_evecs.leftCols(m_num_computed) * Gkx;
    }

    inline void compute_next(int verbose = 0)
    {
        if(m_num_computed + m_inc_evals > m_max_evals)
            throw std::logic_error("maximum number of eigenvalues computed");

        Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, IncrementalEig>
            eigs(this, m_inc_evals, std::min(m_n, m_inc_evals * 2 + 1));
        eigs.init();
        eigs.compute(1000, 1e-6);

        if(verbose > 1)
            Rcpp::Rcout << "[inceig] nops = " << eigs.num_operations() << std::endl;

        m_evals.segment(m_num_computed, m_inc_evals).noalias() = eigs.eigenvalues();
        m_evecs.block(0, m_num_computed, m_n, m_inc_evals).noalias() = eigs.eigenvectors();
        m_num_computed += m_inc_evals;
    }

    const int num_computed() const { return m_num_computed; }
    const VectorXd& eigenvalues() const { return m_evals; }
    const MatrixXd& eigenvectors() const { return m_evecs; }
};


#endif // GRADFPS_INCEIG_SPECTRA_H
