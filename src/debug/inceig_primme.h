#ifndef GRADFPS_INCEIG_PRIMME_H
#define GRADFPS_INCEIG_PRIMME_H

#include "common.h"
#include <primme.h>

inline void mat_vec_prod(
    void* x, PRIMME_INT* ldx, void* y, PRIMME_INT* ldy,
    int* blockSize, primme_params* primme, int* err
);

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
    VectorXd      m_resid;

    int           m_max_evals;
    int           m_inc_evals;
    int           m_num_computed;

    primme_params m_primme;

public:
    IncrementalEig() {}

    inline void init(const RefConstMat& mat, int max_evals, int inc_evals = 1, primme_target target = primme_largest)
    {
        m_max_evals = max_evals;
        m_inc_evals = inc_evals;
        m_num_computed = 0;

        m_n = mat.rows();
        m_mat_data = mat.data();
        m_evals.resize(m_max_evals);
        m_evecs.resize(m_n, m_max_evals);
        m_resid.resize(m_n);

        primme_initialize(&m_primme);
        m_primme.matrixMatvec = mat_vec_prod;
        m_primme.n = m_n;
        m_primme.numEvals = inc_evals;
        m_primme.eps = 1e-6;
        m_primme.target = target;
        m_primme.iseed[0] = 0;
        m_primme.iseed[1] = 1;
        m_primme.iseed[2] = 2;
        m_primme.iseed[3] = 3;
        m_primme.matrix = reinterpret_cast<void*>(this);

        primme_set_method(PRIMME_DEFAULT_MIN_TIME, &m_primme);
    }

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

        int ret = dprimme(&m_evals[m_num_computed], &m_evecs(0, m_num_computed), m_resid.data(), &m_primme);

        if(verbose > 1)
            Rcpp::Rcout << "[inceig] nops = " << m_primme.stats.numMatvecs << std::endl;

        m_num_computed += m_inc_evals;
    }

    const int num_computed() const { return m_num_computed; }
    const VectorXd& eigenvalues() const { return m_evals; }
    const MatrixXd& eigenvectors() const { return m_evecs; }
};

inline void mat_vec_prod(
    void* x, PRIMME_INT* ldx, void* y, PRIMME_INT* ldy,
    int* blockSize, primme_params* primme, int* err
)
{
    IncrementalEig* mat = reinterpret_cast<IncrementalEig*>(primme->matrix);

    double* xvec;     /* pointer to i-th input vector x */
    double* yvec;     /* pointer to i-th output vector y */

    for(int i = 0; i < *blockSize; i++)
    {
        xvec = (double*) x + *ldx * i;
        yvec = (double*) y + *ldy * i;

        mat->perform_op(xvec, yvec);
    }
    *err = 0;
}


#endif // GRADFPS_INCEIG_PRIMME_H
