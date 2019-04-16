#include "prox_lp.h"

using Rcpp::NumericVector;
using Rcpp::NumericMatrix;

// [[Rcpp::export]]
NumericVector prox_lp(NumericVector v, double p, double alpha, int maxit = 100)
{
    const int n = v.length();
    MapConstVec vv(v.begin(), n);

    NumericVector res(n);
    MapVec x(res.begin(), n);

    prox_lp_impl(vv, p, alpha, x, maxit);

    return res;
}

// [[Rcpp::export]]
NumericMatrix prox_lp_mat(NumericMatrix v, double p, double alpha, int maxit = 100)
{
    const int n = v.nrow();
    MapConstMat vv(v.begin(), n, n);

    NumericMatrix res(n, n);
    MapMat x(res.begin(), n, n);

    prox_lp_mat_impl(vv, p, alpha, x, maxit);

    return res;
}

/* // [[Rcpp::export]]
NumericVector prox_lp_mat2(NumericMatrix vv, double p, double alpha, int maxit = 100)
{
    const int n = vv.nrow();
    const int offdiagn = (n * (n - 1)) / 2;

    NumericVector vdiag(n), voffdiag(offdiagn);

    // Copy v and set x = v / (1 + alpha)
    int offind = 0;
    for(int j = 0; j < n; j++)
    {
        vdiag[j] = vv(j, j);
        for(int i = j + 1; i < n; i++, offind++)
        {
            voffdiag[offind] = vv(i, j);
        }
    }
    NumericVector xdiag = vdiag / (1.0 + alpha);
    NumericVector xoffdiag = voffdiag / (1.0 + alpha);

    NumericVector xpdiag = Rcpp::pow(Rcpp::abs(xdiag), p);
    NumericVector xpoffdiag = Rcpp::pow(Rcpp::abs(xoffdiag), p);

    const double cp = 2.0 / p - 1.0;
    double c = 0.0;

    double psum = Rcpp::sum(xpoffdiag) * 2.0 + Rcpp::sum(xpdiag);
    double newc = alpha * std::pow(psum, cp);

    for(int it = 0; it < maxit; it++)
    {
        Rcpp::Rcout << "iter = " << it << std::endl;

        for(int j = 0; j < n; j++)
        {
            const double signv = (vdiag[j] > 0.0) - (vdiag[j] < 0.0);
            xdiag[j] = signv * solve_equation(newc, p, std::abs(vdiag[j]), std::abs(xdiag[j]));
            const double newxip = std::pow(std::abs(xdiag[j]), p);
            psum += (newxip - xpdiag[j]);
            newc = alpha * std::pow(psum, cp);
            xpdiag[j] = newxip;
        }

        for(int j = 0; j < offdiagn; j++)
        {
            const double signv = (voffdiag[j] > 0.0) - (voffdiag[j] < 0.0);
            xoffdiag[j] = signv * solve_equation(newc, p, std::abs(voffdiag[j]), std::abs(xoffdiag[j]));
            const double newxip = std::pow(std::abs(xoffdiag[j]), p);
            psum += 2 * (newxip - xpoffdiag[j]);
            newc = alpha * std::pow(psum, cp);
            xpoffdiag[j] = newxip;
        }

        // Rcpp::Rcout << newc << std::endl;
        // Rcpp::Rcout << "diff = " << std::abs(newc - c) << ", thresh = " << 1e-6 * std::max(1.0, c) << std::endl;

        if(std::abs(newc - c) < 1e-6 * std::max(1.0, c))
            break;

        c = newc;
    }

    return xdiag;
} */
