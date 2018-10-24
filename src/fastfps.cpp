#include "fastfps.h"
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include "walltime.h"

using Rcpp::NumericMatrix;
using Rcpp::List;

using namespace Spectra;

// Initial guess using partial eigen decomposition
inline void initial_guess(const MapMat& S, int d, MatrixXd& x)
{
    const int ncv = std::max(10, 2 * d + 1);
    DenseSymMatProd<double> op(S);
    SymEigsSolver< double, LARGEST_ALGE, DenseSymMatProd<double> > eigs(&op, d, ncv);
    eigs.init();
    eigs.compute(1000, 0.001);
    MatrixXd evecs = eigs.eigenvectors();
    x.noalias() = evecs * evecs.transpose();
}

// Thresholding of eigenvalues
inline double lambda_max_thresh(double x, double thresh)
{
    return (x > 1.0 + thresh) ?
           (x - thresh) :
           ((x > 1.0) ? 1.0 : x);
}
inline double lambda_min_thresh(double x, double thresh)
{
    return (x > 0.0) ?
           (x) :
           ((x > -thresh) ? 0.0 : (x + thresh));
}

// [[Rcpp::export]]
List fastfps(NumericMatrix S, int d, double lambda, int maxiter,
             double alpha0, double mu, double r, bool feas = false)
{
    const int n = S.nrow();
    const int p = S.ncol();
    if(n != p)
        Rcpp::stop("S must be square");

    MapMat Smat(S.begin(), p, p);
    MatrixXd x(p, p), xold(p, p);
    SpMat xsp(p, p);

    std::vector<double> fn_obj, fn_feas, fn_feas1, fn_feas2, time;

    // Initial guess -- using partial eigen decomposition
    initial_guess(Smat, d, x);

    // Eigenvalue computation
    SparseSymMatProd<double> op(xsp);
    SymEigsSolver< double, BOTH_ENDS, SparseSymMatProd<double> > eigs(&op, 2, 10);
    VectorXd evals(2);
    MatrixXd evecs(p, 2);
    VectorXd diag(p);

    DenseSymMatProd<double> op_feas(x);
    SymEigsSolver< double, BOTH_ENDS, DenseSymMatProd<double> > eigs_feas(&op_feas, 2, 10);

    double alpha = 0.0;
    double time1, time2;
    for(int i = 0; i < maxiter; i++)
    {
        Rcpp::Rcout << i << std::endl;

        time1 = get_wall_time();
        alpha = alpha0 / std::pow(i + 1.0, 0.75);

        // L1 thresholding
        soft_thresh_sparse(x, lambda * alpha, xsp);

        // Eigenvalue shrinkage
        eigs.init();
        eigs.compute(1000, 1e-6);
        evals.noalias() = eigs.eigenvalues();
        evecs.noalias() = eigs.eigenvectors();

        const double lmax_new = lambda_max_thresh(evals[0], alpha * mu * r);
        const double lmin_new = lambda_min_thresh(evals[1], alpha * mu * r);
        rank2_update_sparse(xsp, lmax_new - evals[0], evecs.col(0), lmin_new - evals[1], evecs.col(1), x);

        // Trace shrinkage
        const double tbar = x.diagonal().mean();
        const double tr_shift = double(d) / double(p) - tbar;
        diag.array() = x.diagonal().array() + tr_shift;
        const double beta = alpha * mu / std::sqrt(double(p)) / std::abs(tr_shift);
        if(beta >= 1.0)
        {
            x.diagonal().noalias() = diag;
        } else {
            x.diagonal().noalias() = (1.0 - beta) * x.diagonal() + beta * diag;
        }

        // Gradient descent
        x.triangularView<Eigen::Upper>() = x.triangularView<Eigen::Lower>().transpose();
        if(i >= 2)
        {
            x.noalias() += (double(i - 1.0) / double(i + 2.0)) * (x - xold) + alpha * Smat;
        } else {
            x.noalias() += alpha * Smat;
        }
        const double xnorm = x.norm();
        const double radius = std::sqrt(double(d));
        if(xnorm > radius)
        {
            x /= (xnorm / radius);
        }
        xold.noalias() = x;

        time2 = get_wall_time();
        fn_obj.push_back(-Smat.cwiseProduct(x).sum() + lambda * x.cwiseAbs().sum());
        time.push_back(time2 - time1);

        if(feas)
        {
            eigs_feas.init();
            eigs_feas.compute(1000, 1e-6);
            evals.noalias() = eigs_feas.eigenvalues();
            const double feas1 = mu * r * (std::max(0.0, -evals[1]),
                                           std::max(0.0, evals[0] - 1.0));
            const double tbar = x.diagonal().mean();
            const double tr_shift = double(d) / double(p) - tbar;
            const double feas2 = mu * std::sqrt(double(p)) * std::abs(tr_shift);
            fn_feas1.push_back(feas1);
            fn_feas2.push_back(feas2);
            fn_feas.push_back(feas1 + feas2);
        }
    }

    soft_thresh_sparse(x, lambda * alpha, xsp);
    xsp.makeCompressed();

    return List::create(
        Rcpp::Named("projection") = xsp,
        Rcpp::Named("objfn")      = fn_obj,
        Rcpp::Named("feasfn1")    = fn_feas1,
        Rcpp::Named("feasfn2")    = fn_feas2,
        Rcpp::Named("feasfn")     = fn_feas,
        Rcpp::Named("time")       = time
    );
}
