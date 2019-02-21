#include "common.h"
#include <primme.h>

const int MaxEvals = 10;

void mat_vec_prod(
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
    int           m_num_computed;

    primme_params m_primme;

public:
    IncrementalEig() :
        m_evals(MaxEvals)
    {}

    inline void init(const RefConstMat& mat, primme_target target = primme_largest)
    {
        m_n = mat.rows();
        m_mat_data = mat.data();
        m_evecs.resize(m_n, MaxEvals);
        m_resid.resize(m_n);
        m_num_computed = 0;

        primme_initialize(&m_primme);
        m_primme.matrixMatvec = mat_vec_prod;
        m_primme.n = m_n;
        m_primme.numEvals = 1;
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
    inline void perform_op(const double* x_in, double* y_out)
    {
        MapConstVec x(x_in, m_n);
        MapVec y(y_out, m_n);
        MapConstMat mat(m_mat_data, m_n, m_n);

        if(m_num_computed < 1)
            y.noalias() = mat * x;

        VectorXd Gkx = m_evecs.leftCols(m_num_computed).transpose() * x;
        Gkx.array() *= m_evals.head(m_num_computed).array();
        y.noalias() = mat * x - m_evecs.leftCols(m_num_computed) * Gkx;
    }

    inline void compute_next()
    {
        if(m_num_computed >= MaxEvals)
            throw std::logic_error("maximum number of eigenvalues computed");

        int ret = dprimme(&m_evals[m_num_computed], &m_evecs(0, m_num_computed), m_resid.data(), &m_primme);
        m_num_computed++;
    }

    const int num_computed() const { return m_num_computed; }
    const VectorXd& eigenvalues() const { return m_evals; }
    const MatrixXd& eigenvectors() const { return m_evecs; }
};

void mat_vec_prod(
    void* x, PRIMME_INT* ldx, void* y, PRIMME_INT* ldy,
    int* blockSize, primme_params* primme, int* err)
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

using Rcpp::NumericMatrix;

// [[Rcpp::export]]
NumericMatrix eigmax_thresh(NumericMatrix x, double penalty)
{
    const int n = x.nrow();
    Eigen::Map<const MatrixXd> mat(x.begin(), n, n);
    NumericMatrix res(n, n);

    IncrementalEig solver;
    solver.init(mat, primme_largest);
    solver.compute_next();
    const VectorXd& evals = solver.eigenvalues();

    double lambda = evals[0];
    if(lambda <= 1.0)
        return res;

    double target = lambda - penalty;
    int i = 1;
    for(i = 1; i < MaxEvals; i++)
    {
        target = std::max(1.0, (evals.head(i).sum() - penalty) / double(i));
        solver.compute_next();
        if(target >= evals[i])
            break;
    }

    target = std::max(1.0, (evals.head(i).sum() - penalty) / double(i));
    // Rcpp::Rcout << "eigmax: " << target << " " << i << std::endl;
    VectorXd delta_evals = VectorXd::Constant(i, target) - evals.head(i);
    MapMat delta_x(res.begin(), n, n);
    const MatrixXd& evecs = solver.eigenvectors();
    delta_x.noalias() = evecs.leftCols(i) * delta_evals.asDiagonal() * evecs.leftCols(i).transpose();

    return res;
}

// [[Rcpp::export]]
NumericMatrix eigmin_thresh(NumericMatrix x, double penalty)
{
    const int n = x.nrow();
    Eigen::Map<const MatrixXd> mat(x.begin(), n, n);
    NumericMatrix res(n, n);

    IncrementalEig solver;
    solver.init(mat, primme_smallest);
    solver.compute_next();
    const VectorXd& evals = solver.eigenvalues();

    double lambda = evals[0];
    if(lambda >= 0.0)
        return res;

    double target = lambda + penalty;
    int i = 1;
    for(i = 1; i < MaxEvals; i++)
    {
        target = std::min(0.0, (evals.head(i).sum() + penalty) / double(i));
        solver.compute_next();
        if(target <= evals[i])
            break;
    }

    target = std::min(0.0, (evals.head(i).sum() + penalty) / double(i));
    // Rcpp::Rcout << "eigmin: " << target << " " << i << std::endl;
    VectorXd delta_evals = VectorXd::Constant(i, target) - evals.head(i);
    MapMat delta_x(res.begin(), n, n);
    const MatrixXd& evecs = solver.eigenvectors();
    delta_x.noalias() = evecs.leftCols(i) * delta_evals.asDiagonal() * evecs.leftCols(i).transpose();

    return res;
}
