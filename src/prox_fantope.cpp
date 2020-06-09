#include "common.h"
#include "prox_fantope.h"
#include "quadprog.h"
#include "inceig_tridiag_spectra.h"
#include "inceig_tridiag_lapack.h"
#include "walltime.h"

// min  0.5 * ||x - lambda||^2 + l1 * max(0, max(x) - 1) + l2 * |sum(x) - d|
// s.t. x >= 0
//
// This can be solved using quadratic programming
//
// min  0.5 * ||x - lambda||^2 + l1 * v2 + l2 * v5
// s.t. 0 <= xi <= v1
//      v2 >= 0, v2 >= v1 - 1
//      v5 >= sum(x) - d
//      v5 >= d - sum(x)
//
// This is the old method to solve the problem, only kept for record
// Below we have a better algorithm implemented as prox_fantope_vec()
inline double prox_fantope_vec_quadprog(const double* lambda, int p, double l1, double l2, int d, double* sol)
{
    int n = p + 3;
    // Inverse of the D matrix by setting ierr = 1 in input
    Matrix Dmat = Matrix::Identity(n, n);
    Dmat(p, p) = 1e6;
    Dmat(p + 1, p + 1) = 1e6;
    Dmat(p + 2, p + 2) = 1e6;

    Vector dvec(n);
    std::copy(lambda, lambda + p, dvec.data());
    dvec[p] = 0.0;
    dvec[p + 1] = -l1;
    dvec[p + 2] = -l2;

    int m = 2 * p + 4;
    Matrix Amat = Matrix::Zero(n, m);
    Vector bvec = Vector::Zero(m);
    // 0 <= xi <= v1
    for(int j = 0; j < p; j++)
    {
        // xi >= 0
        Amat(j, j) = 1.0;
        // v1 - xi >= 0;
        Amat(j, p + j) = -1.0;
        Amat(p, p + j) = 1.0;
    }
    // v2 >= 0, v2 >= v1 - 1
    Amat(p + 1, 2 * p) = 1.0;
    Amat(p, 2 * p + 1) = -1.0;
    Amat(p + 1, 2 * p + 1) = 1.0;
    bvec[2 * p + 1] = -1.0;
    // v5 >= sum(x) - d
    for(int i = 0; i < p; i++)
        Amat(i, 2 * p + 2) = -1.0;
    Amat(p + 2, 2 * p + 2) = 1.0;
    bvec[2 * p + 2] = -d;
    // v5 >= d - sum(x)
    for(int i = 0; i < p; i++)
        Amat(i, 2 * p + 3) = 1.0;
    Amat(p + 2, 2 * p + 3) = 1.0;
    bvec[2 * p + 3] = d;

    int meq = 0;

    Vector y(n), lagr(m), work(2 * n + (n * (n + 5)) / 2 + 2 * m + 1);
    Eigen::VectorXi iact(m);
    double crval;
    int nact;
    int iter[2];
    int ierr = 1;

    F77_CALL(qpgen2)
        (Dmat.data(), dvec.data(), &n, &n,
         y.data(), lagr.data(), &crval,
         Amat.data(), bvec.data(), &n, &m,
         &meq, iact.data(), &nact, iter,
         work.data(), &ierr);

    std::copy(y.data(), y.data() + p, sol);
    return crval;
}



// Helper function for prox_fantope_vec()
//
// A modified version of the algorithm in
// Projection onto the probability simplex, Wang and Carreira-Perpinan (2013)
// https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf
//
// Given x[0] >= x[1] >= ... >= x[n-1], find c such that
// sum y[i] = sum max(0, x[i] - c) = d, 0 < d <= n
inline double find_c_max0(const double* x, int n, int d)
{
    // Each time we assume y has the form
    // x[0]-c, ..., x[i]-c, 0, ..., 0
    // and then compute c such that the sum is d
    // If x[i+1]-c <= 0, then we have found the solution
    double sumx = 0.0;
    for(int i = 0; i < n - 1; i++)
    {
        sumx += x[i];
        const double c = double(sumx - d) / double(i + 1);
        if(x[i + 1] - c <= 0.0)
            return c;
    }
    // If we don't find c in the loop, then all elements of y are >= 0
    // So we let c = (sumx - d) / n
    sumx += x[n - 1];
    return double(sumx - d) / double(n);
}

// Helper function for prox_fantope_vec()
//
// Given x[0] >= x[1] >= ... >= x[n-1], find c such that
// sum y[i] = sum min(1, x[i] - c) = d, 0 < d <= n
//
// sumx is cached since this function is typically called sequentially in a loop
inline double find_c_min1(const double* x, int n, double sumx, int d)
{
    // Sum of x
    // double sumx = std::accumulate(x, x + n, 0.0);

    // Each time we assume y has the form
    // 1, ..., 1, x[i]-c, ..., x[n-1]-c
    // and then compute c such that the sum is d
    // If 1 >= x[i]-c, then we have found the solution
    for(int i = 0; i < n; i++)
    {
        const double c = double(i + sumx - d) / double(n - i);
        if(x[i] - c <= 1.0)
            return c;
        sumx -= x[i];
    }
    // If we don't find c in the loop, then all elements of y are 1
    // So we let x[n-1] - c = 1
    return x[n - 1] - 1.0;
}

// Helper function for prox_fantope_vec()
//
// Given x[0] >= x[1] >= ... >= x[n-1], find z such that
// sum z[i] = sum max(0, min(1, x[i] - c)) = d, d <= n
// This function returns ||x - lambda||^2
inline double proj_cube_simplex(const double* lambda, int p, int d, double& c, double* sol)
{
    // Let y[i] = min(1, x[i] - c)
    // Each time we assume z has the form
    // y[0], y[1], ..., y[i-1], 0, 0, ..., 0
    // and then compute c such that the sum is d
    // If y[i] <= 0, then we have found the solution of c
    double suml = std::accumulate(lambda, lambda + d, 0.0);
    c = 0.0;
    for(int i = d; i <= p; i++)
    {
        c = find_c_min1(lambda, i, suml, d);
        if(i == p || lambda[i] - c <= 0.0)
            break;
        suml += lambda[i];
    }

    double obj = 0.0;
    for(int i = 0; i < p; i++)
    {
        sol[i] = std::min(1.0, std::max(0.0, lambda[i] - c));
        obj += (sol[i] - lambda[i]) * (sol[i] - lambda[i]);
    }

    return obj;
}

// Helper function for prox_fantope_vec()
//
// ||x - y||^2
inline double squared_dist(const double* x, const double* y, int n)
{
    double res = 0.0;
    for(int i = 0; i < n; i++)
        res += (x[i] - y[i]) * (x[i] - y[i]);
    return res;
}

// min  0.5 * ||x - lambda||^2 + l1 * max(0, max(x) - 1) + l2 * |sum(x) - d|
// s.t. x >= 0
//
// Solved by KKT conditions, assuming lambda[i] is decreasing
// x has the form x[i] = min(t, max(0, lambda[i] - c)), where t and c are quantities to be determined
//
// This function returns the objective function value
inline double prox_fantope_vec(const double* lambda, int p, double l1, double l2, int d, double* sol)
{
    // Case I: t = 1
    // x[i] = min(1, max(0, lambda[i] - c))
    // Define mu[i] = max(0, lambda[i] - c - 1)
    // c and mu satisfy conditions (a) -l2 <= c <= l2, (b) sum(mu) <= l1

    // Case I.1: sum(x) > d
    // In this case c = l2, so we compute x and test sum(x)>d and (b)
    double c = l2, sumx = 0.0, summu = 0.0;
    for(int i = 0; i < p; i++)
    {
        sol[i] = std::min(1.0, std::max(0.0, lambda[i] - c));
        const double mui = std::max(0.0, lambda[i] - c - 1.0);
        sumx += sol[i];
        summu += mui;
    }
    if(sumx >= d && summu <= l1)
        return 0.5 * squared_dist(lambda, sol, p) + l2 * std::abs(sumx - d);

    // Case I.2: sum(x) < d
    // In this case c = -l2, so we compute x and test sum(x)<d and (b)
    c = -l2; sumx = 0.0; summu = 0.0;
    for(int i = 0; i < p; i++)
    {
        sol[i] = std::min(1.0, std::max(0.0, lambda[i] - c));
        const double mui = std::max(0.0, lambda[i] - c - 1.0);
        sumx += sol[i];
        summu += mui;
    }
    if(sumx <= d && summu <= l1)
        return 0.5 * squared_dist(lambda, sol, p) + l2 * std::abs(sumx - d);

    // Case I.3: sum(x) = d
    // This is the case of proj_cube_simplex()
    double sqdist = proj_cube_simplex(lambda, p, d, c, sol);
    summu = 0.0;
    for(int i = 0; i < p; i++)
    {
        summu += std::max(0.0, lambda[i] - c - 1.0);
    }
    if(std::abs(c) <= l2 && summu <= l1)
        return 0.5 * sqdist;


    // Case II: t > 1
    // x has the form
    // x[0] = t, ..., x[I-1] = t, x[I] = max(0, lambda[I] - c), ..., x[p-1] = max(0, lambda[p-1] - c)
    // I is the smallest index such that (sum(lambda[0..I-1]) - l1) / I > lambda[I]
    int I;
    double suml = 0.0;
    for(I = 1 ; I < p; I++)
    {
        suml += lambda[I - 1];
        if(suml - l1 >= I * lambda[I])
            break;
    }
    if(I == p)
        suml += lambda[p - 1];
    // t + c = (sum(lambda[0..I-1]) - l1) / I
    const double tc = (suml - l1) / I;

    // Case II.1: sum(x) > d
    // In this case c = l2, so we compute t and test whether sum(x)>d and t>1
    c = l2;
    double t = tc - c;
    if(t >= 1.0)
    {
        sumx = 0.0;
        for(int i = 0; i < p; i++)
        {
            sol[i] = std::min(t, std::max(0.0, lambda[i] - c));
            sumx += sol[i];
        }
        if(sumx >= d)
            return 0.5 * squared_dist(lambda, sol, p) + l1 * (t - 1.0) + l2 * std::abs(sumx - d);
    }

    // Case II.2: sum(x) < d
    // In this case c = -l2, so we compute t and test whether sum(x)<d and t>1
    c = -l2;
    t = tc - c;
    if(t >= 1.0)
    {
        sumx = 0.0;
        for(int i = 0; i < p; i++)
        {
            sol[i] = std::min(t, std::max(0.0, lambda[i] - c));
            sumx += sol[i];
        }
        if(sumx <= d)
            return 0.5 * squared_dist(lambda, sol, p) + l1 * (t - 1.0) + l2 * std::abs(sumx - d);
    }

    // Case II.3: sum(x) = d
    // Let z[0] = tc, ..., z[I-1] = tc, z[I] = lambda[I], ..., z[p-1] = lambda[p-1]
    // Find c such that sum(max(0, z - c)) = d
    double* z = new double[p];
    std::fill(z, z + I, tc);
    std::copy(lambda + I, lambda + p, z + I);
    c = find_c_max0(z, p, d);
    delete [] z;
    t = tc - c;
    sqdist = 0.0;
    for(int i = 0; i < p; i++)
    {
        sol[i] = std::min(t, std::max(0.0, lambda[i] - c));
        sqdist += (sol[i] - lambda[i]) * (sol[i] - lambda[i]);
    }
    return 0.5 * sqdist + l1 * (t - 1.0);
}



// min  -tr(SX) + s1 * max(0, eigmax(X)-1) + s2 * |tr(X)-d| + (0.5/alpha) * ||X - U||_F^2
// s.t. X is p.s.d.
//
// Equivalent to
// min  alpha*s1 * max(0, eigmax(X)-1) + alpha*s2 * |tr(X)-d| + 0.5 * ||X - U - alpha*S||_F^2
// s.t. X is p.s.d.
//
// For simplicity we solve
// min  l1 * max(0, eigmax(X)-1) + l2 * |tr(X)-d| + 0.5 * ||X - A||_F^2
// s.t. X is p.s.d.
template <typename IncrementalEigType>
int prox_fantope_template(
    RefConstMat A, double l1, double l2, int d, int inc, int maxiter, RefMat res,
    double eps, int verbose
)
{
    Vector theta(inc * maxiter + d + 1);
    IncrementalEigType inceig(A.rows());

    double t1 = get_wall_time();
    inceig.init(A, inc * maxiter + d + 1, d + 1, 0, 0);
    double t2 = get_wall_time();

    const Vector& evals = inceig.largest_eigenvalues();
    double f = prox_fantope_vec(evals.data(), inceig.num_computed_largest(),
                                l1, l2, d, theta.data());
    double theta_last = theta[inceig.num_computed_largest() - 1];

    if(verbose > 1)
    {
        Rcpp::Rcout << "  [prox_fantope_impl] time_init = " << t2 - t1 << std::endl;
        Rcpp::Rcout << "  [prox_fantope_impl] inc = " << inc << ", maxiter = " << maxiter << std::endl;
    }

    for(int i = 0; i < maxiter; i++)
    {
        // If theta has reached zero eigenvalues
        if(std::abs(theta_last) <= eps)
            break;

        if(verbose > 1)
            Rcpp::Rcout << "  [prox_fantope_impl] iter = " << i << std::endl;
        if(verbose > 0 && i == maxiter - 1)
            Rcpp::Rcout << "  [prox_fantope_impl] maxiter = " << maxiter << " reached!" << std::endl;

        double t1 = get_wall_time();
        int nops = inceig.compute_next_largest(inc);
        const Vector& evals = inceig.largest_eigenvalues();
        double t2 = get_wall_time();

        double newf = prox_fantope_vec(evals.data(), inceig.num_computed_largest(),
                                       l1, l2, d, theta.data());
        theta_last = theta[inceig.num_computed_largest() - 1];

        if(verbose > 1)
            Rcpp::Rcout << "  [prox_fantope_impl] f = " << f << ", nops = " << nops
                        << ", time_eig = " << t2 - t1 << std::endl << std::endl;

        // If f does not significantly decrease
        if(std::abs(newf - f) <= eps * std::abs(f))
            break;

        f = newf;
    }

    int pos = d;
    const int end = inceig.num_computed_largest();
    for(; pos < end; pos++)
    {
        if(std::abs(theta[pos]) <= 1e-6)
            break;
    }

    if(verbose > 1)
    {
        const int nevals = inceig.num_computed_largest();
        if(nevals <= 5)
        {
            Rcpp::Rcout << "  [prox_fantope_impl] evals = " << inceig.largest_eigenvalues().head(nevals).transpose() << std::endl;
        } else {
            const int tail = std::min(5, nevals - 5);
            Rcpp::Rcout << "  [prox_fantope_impl] evals = " << inceig.largest_eigenvalues().head(5).transpose() << " ..." << std::endl;
            Rcpp::Rcout << "                              " << inceig.largest_eigenvalues().segment(nevals - tail, tail).transpose() << std::endl;
        }

        if(pos <= 5)
        {
            Rcpp::Rcout << "  [prox_fantope_impl] theta = " << theta.head(pos).transpose() << std::endl;
        } else {
            const int tail = std::min(5, pos - 5);
            Rcpp::Rcout << "  [prox_fantope_impl] theta = " << theta.head(5).transpose() << " ..." << std::endl;
            Rcpp::Rcout << "                              " << theta.segment(pos - tail, tail).transpose() << std::endl;
        }
    }

    t1 = get_wall_time();
    inceig.compute_eigenvectors(pos, 0);
    t2 = get_wall_time();
    res.noalias() = inceig.largest_eigenvectors().leftCols(pos) *
        theta.head(pos).asDiagonal() *
        inceig.largest_eigenvectors().leftCols(pos).transpose();
    double t3 = get_wall_time();

    if(verbose > 1)
    {
        Rcpp::Rcout << "  [prox_fantope_impl] time_post1 = " << t2 - t1
                    << ", time_post2 = " << t3 - t2 << std::endl;
    }

    return pos;
}

int prox_fantope_impl(
    RefConstMat A, double l1, double l2, int d, int inc, int maxiter, RefMat res,
    double eps, int verbose, EigMethod method
)
{
    if(method == EigMethod::Spectra)
        return prox_fantope_template<IncrementalEigSpectra>(
            A, l1, l2, d, inc, maxiter, res, eps, verbose
        );

    return prox_fantope_template<IncrementalEigLapack>(
        A, l1, l2, d, inc, maxiter, res, eps, verbose
    );
}



// The old implementation, in fact a hard projection
//
// min  -tr(AX) + 0.5 * ||X||_F^2
// s.t. X in Fantope
int prox_fantope_hard_impl(
    RefConstMat A, int d, int inc, int maxiter, RefMat res, double& dsum,
    double eps, int verbose
)
{
    Vector theta(inc * maxiter + d + 1);
    IncrementalEigLapack inceig(A.rows());

    double t1 = get_wall_time();
    inceig.init(A, inc * maxiter + d + 1, d + 1, 0, 0);
    double t2 = get_wall_time();

    const Vector& evals = inceig.largest_eigenvalues();
    double c;
    double f = proj_cube_simplex(evals.data(), inceig.num_computed_largest(), d, c, theta.data());
    double theta_last = theta[inceig.num_computed_largest() - 1];

    if(verbose > 1)
    {
        Rcpp::Rcout << "  [prox_fantope_impl] time_init = " << t2 - t1 << std::endl;
        Rcpp::Rcout << "  [prox_fantope_impl] inc = " << inc << ", maxiter = " << maxiter << std::endl;
    }

    for(int i = 0; i < maxiter; i++)
    {
        // If theta has reached zero eigenvalues
        if(std::abs(theta_last) <= eps)
            break;

        if(verbose > 1)
            Rcpp::Rcout << "  [prox_fantope_impl] iter = " << i << std::endl;
        if(verbose > 0 && i == maxiter - 1)
            Rcpp::Rcout << "  [prox_fantope_impl] maxiter = " << maxiter << " reached!" << std::endl;

        double t1 = get_wall_time();
        int nops = inceig.compute_next_largest(inc);
        const Vector& evals = inceig.largest_eigenvalues();
        double t2 = get_wall_time();

        double newf = proj_cube_simplex(evals.data(), inceig.num_computed_largest(), d, c, theta.data());
        theta_last = theta[inceig.num_computed_largest() - 1];

        if(verbose > 1)
            Rcpp::Rcout << "  [prox_fantope_impl] f = " << f << ", nops = " << nops
                        << ", time_eig = " << t2 - t1 << std::endl << std::endl;

        // If f does not significantly decrease
        if(std::abs(newf - f) <= eps * std::abs(f))
            break;

        f = newf;
    }

    int pos = d;
    const int end = inceig.num_computed_largest();
    for(; pos < end; pos++)
    {
        if(std::abs(theta[pos]) <= 1e-6)
            break;
    }

    if(verbose > 1)
    {
        const int nevals = inceig.num_computed_largest();
        if(nevals <= 5)
        {
            Rcpp::Rcout << "  [prox_fantope_impl] evals = " << inceig.largest_eigenvalues().head(nevals).transpose() << std::endl;
        } else {
            const int tail = std::min(5, nevals - 5);
            Rcpp::Rcout << "  [prox_fantope_impl] evals = " << inceig.largest_eigenvalues().head(5).transpose() << " ..." << std::endl;
            Rcpp::Rcout << "                              " << inceig.largest_eigenvalues().segment(nevals - tail, tail).transpose() << std::endl;
        }

        if(pos <= 5)
        {
            Rcpp::Rcout << "  [prox_fantope_impl] theta = " << theta.head(pos).transpose() << std::endl;
        } else {
            const int tail = std::min(5, pos - 5);
            Rcpp::Rcout << "  [prox_fantope_impl] theta = " << theta.head(5).transpose() << " ..." << std::endl;
            Rcpp::Rcout << "                              " << theta.segment(pos - tail, tail).transpose() << std::endl;
        }
    }

    t1 = get_wall_time();
    inceig.compute_eigenvectors(pos, 0);
    t2 = get_wall_time();
    res.noalias() = inceig.largest_eigenvectors().leftCols(pos) *
        theta.head(pos).asDiagonal() *
        inceig.largest_eigenvectors().leftCols(pos).transpose();
    double t3 = get_wall_time();

    if(verbose > 1)
    {
        Rcpp::Rcout << "  [prox_fantope_impl] time_post1 = " << t2 - t1
                    << ", time_post2 = " << t3 - t2 << std::endl;
    }

    return pos;
}



// min  -tr(AX) + (0.5 / alpha) * ||X - B||_F^2
// s.t. X in Fantope

//' Proximal operator on Fantope
//'
//' This function solves the optimization problem
//' \deqn{\min\quad-tr(AX) + \frac{1}{2\alpha}||X - B||_F^2}{min  -tr(AX) + (0.5 / \alpha) * ||X - B||_F^2}
//' \deqn{s.t.\quad X\in \mathcal{F}^d}{s.t. X in F^d}
//'
//' @param A       A symmetric matrix.
//' @param B       A symmetric matrix of the same dimension as \code{A}.
//' @param alpha   Proximal parameter.
//' @param d       Fantope parameter.
//' @param eps     Precision of the result.
//' @param inc     How many incremental eigenvalues to compute in each iteration.
//' @param maxiter Maximum number of iterations.
//' @param verbose Level of verbosity.
// [[Rcpp::export]]
Rcpp::NumericMatrix prox_fantope(MapMat A, MapMat B, double alpha, int d,
                                 double eps = 1e-5, int inc = 1, int maxiter = 10,
                                 int verbose = 0)
{
    const int n = A.rows();
    if(A.cols() != n)
        Rcpp::stop("A is not square");
    if(B.rows() != n || B.cols() != n)
        Rcpp::stop("dimensions of A and B do not change");

    Matrix mat = B + alpha * A;
    MapConstMat matm(mat.data(), n, n);

    Rcpp::NumericMatrix res(n, n);
    MapMat resm(res.begin(), n, n);
    double dsum;

    prox_fantope_hard_impl(matm, d, inc, maxiter, resm, dsum, eps, verbose);

    return res;
}
