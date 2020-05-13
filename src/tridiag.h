#ifndef GRADFPS_TRIDIAG_H
#define GRADFPS_TRIDIAG_H

#include <cmath>
#include <cstdlib>

#ifdef EIGEN_USE_BLAS
#define F77_CALL(x)	x ## _
#define F77_NAME(x) F77_CALL(x)
#define La_extern extern
extern "C" {

La_extern void
F77_NAME(dgttrf)(const int* n, double* dl, double* d,
                 double* du, double* du2, int* ipiv, int* info);

La_extern void
F77_NAME(dgttrs)(const char* trans, const int* n, const int* nrhs,
                 double* dl, double* d, double* du, double* du2,
                 int* ipiv, double* b, const int* ldb, int* info FCLEN);

}
#else
#include <R_ext/Lapack.h>
#endif

// y = A * x
// Diagonal:    b[0], ..., b[n-1]
// Subdiagonal: c[0], ..., c[n-2]
inline void tridiag_prod(int n, const double* b, const double* c, const double* x, double* y)
{
    y[0] = b[0] * x[0] + c[0] * x[1];
    y[n - 1] = c[n - 2] * x[n - 2] + b[n - 1] * x[n - 1];
    for(int i = 1; i < n - 1; i++)
    {
        y[i] = c[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1];
    }
}

// Factorize A to solve (A - s * I) * x = d
// Diagonal:    b[0], ..., b[n-1]
// Subdiagonal: c[0], ..., c[n-2]
// Working space cmod[n-1], denom[n]
// Return true if successful, otherwise there is a divided-by-zero issue
// https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
inline bool tridiag_fac(int n, const double* b, const double* c, double s,
                        double* cmod, double* denom)
{
    // Tolerance for denominator
    constexpr double eps = 1e-10;

    // Compute the modified c and denominators
    double dtest = b[0] - s;
    if(std::abs(dtest) < eps)
        return false;
    denom[0] = 1.0 / dtest;
    cmod[0] = c[0] * denom[0];
    for(int i = 1; i < n - 1; i++)
    {
        dtest = b[i] - s - c[i - 1] * cmod[i - 1];
        if(std::abs(dtest) < eps)
            return false;
        denom[i] = 1.0 / dtest;
        cmod[i] = c[i] * denom[i];
    }
    dtest = b[n - 1] - s - c[n - 2] * cmod[n - 2];
    denom[n - 1] = 1.0 / dtest;
    return std::abs(dtest) >= eps;
}

// Solve (A - s * I) * x = d based on the factorization result
inline void tridiag_solve(int n, const double* c, const double* cmod, const double* denom,
                          const double* d, double* x)
{
    // Use x to hold the modified values of d
    x[0] = d[0] * denom[0];
    for(int i = 1; i < n; i++)
    {
        x[i] = (d[i] - c[i - 1] * x[i - 1]) * denom[i];
    }
    // Back substitution
    for(int i = n - 2; i >= 0; i--)
    {
        x[i] -= cmod[i] * x[i + 1];
    }
}

enum class OpMode { Prod, ShiftSolve };
enum class ShiftSolver { Fast, Lapack };

class SymTridiag
{
private:
    const int m_n;
    const double* m_diag;
    const double* m_subdiag;

    // Operation mode
    OpMode m_mode;
    // Solver
    ShiftSolver m_solver;

    // Working spaces
    double* m_dcache;
    double* m_lcache;
    double* m_ucache;
    double* m_u2cache;
    int* m_icache;

public:
    SymTridiag(int n, const double* diag, const double* subdiag) :
        m_n(n), m_diag(diag), m_subdiag(subdiag),
        m_mode(OpMode::Prod), m_solver(ShiftSolver::Fast)
    {
        m_dcache = new double[n];
        m_lcache = new double[n - 1];
        m_ucache = new double[n - 1];
        m_u2cache = new double[n - 2];
        m_icache = new int[n];
    }

    ~SymTridiag()
    {
        delete[] m_dcache;
        delete[] m_lcache;
        delete[] m_ucache;
        delete[] m_u2cache;
        delete[] m_icache;
    }

    inline int rows() const { return m_n; }
    inline int cols() const { return m_n; }
    inline void set_mode(OpMode mode) { m_mode = mode; }

    inline bool factorize(double shift)
    {
        // First use the fast solver
        // If not stable (divided-by-zero), use the LAPACK function
        bool success = tridiag_fac(m_n, m_diag, m_subdiag, shift, m_lcache, m_dcache);
        if(success)
        {
            m_solver = ShiftSolver::Fast;
            return true;
        }

        // Make copies of coefficients
        std::transform(m_diag, m_diag + m_n, m_dcache, [shift](double x){ return shift - x; });
        std::transform(m_subdiag, m_subdiag + m_n - 1, m_lcache, std::negate<double>());
        std::copy(m_lcache, m_lcache + m_n - 1, m_ucache);

        int info;
        F77_CALL(dgttrf)(&m_n, m_lcache, m_dcache, m_ucache, m_u2cache, m_icache, &info);
        m_solver = ShiftSolver::Lapack;
        return info == 0;
    }

    inline void perform_op(const double* x_in, double* y_out) const
    {
        // Computing product
        if(m_mode == OpMode::Prod)
        {
            tridiag_prod(m_n, m_diag, m_subdiag, x_in, y_out);
            return;
        }

        // Fast shift solver
        if(m_solver == ShiftSolver::Fast)
        {
            tridiag_solve(m_n, m_subdiag, m_lcache, m_dcache, x_in, y_out);
            // Negate y
            std::transform(y_out, y_out + m_n, y_out, std::negate<double>());
            return;
        }

        // LU-based shift solver
        constexpr char trans = 'N';
        constexpr int nrhs = 1;
        int info;
        std::copy(x_in, x_in + m_n, y_out);
        F77_CALL(dgttrs)(&trans, &m_n, &nrhs,
                         const_cast<double*>(m_lcache),
                         const_cast<double*>(m_dcache),
                         const_cast<double*>(m_ucache),
                         const_cast<double*>(m_u2cache),
                         const_cast<int*>(m_icache),
                         y_out, &m_n, &info);
    }
};


#endif // GRADFPS_TRIDIAG_H
