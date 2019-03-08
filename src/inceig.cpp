#include "inceig_primme.h"

using Rcpp::NumericMatrix;

// [[Rcpp::export]]
NumericMatrix eigmax_thresh(NumericMatrix x, double penalty)
{
    const int MaxEvals = 10;

    const int n = x.nrow();
    Eigen::Map<const MatrixXd> mat(x.begin(), n, n);
    NumericMatrix res(n, n);

    IncrementalEig solver;
    solver.init(mat, MaxEvals, 1, primme_largest);
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
    const int MaxEvals = 10;

    const int n = x.nrow();
    Eigen::Map<const MatrixXd> mat(x.begin(), n, n);
    NumericMatrix res(n, n);

    IncrementalEig solver;
    solver.init(mat, MaxEvals, 1, primme_smallest);
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
