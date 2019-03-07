#include "prox_lp.h"

using Rcpp::NumericVector;
using Rcpp::NumericMatrix;

// [[Rcpp::export]]
NumericVector prox_lp(NumericVector vv, double p, double alpha, int maxit = 100)
{
    const int n = vv.length();
    const double* v = vv.begin();

    NumericVector res = vv / (1.0 + alpha);
    double* x = res.begin();
    const double cp = 2.0 / p - 1.0;
    double c = 0.0;

    NumericVector xp = Rcpp::pow(Rcpp::abs(res), p);
    double psum = Rcpp::sum(xp);
    double newc = alpha * std::pow(psum, cp);

    for(int it = 0; it < maxit; it++)
    {
        // Rcpp::Rcout << "iter = " << it << std::endl;

        for(int i = 0; i < n; i++)
        {
            const double signv = (v[i] > 0.0) - (v[i] < 0.0);
            x[i] = signv * solve_equation(newc, p, std::abs(v[i]));
            const double newxip = std::pow(std::abs(x[i]), p);
            psum += (newxip - xp[i]);
            newc = alpha * std::pow(psum, cp);

            xp[i] = newxip;
        }

        // Rcpp::Rcout << newc << std::endl;
        // Rcpp::Rcout << "diff = " << std::abs(newc - c) << ", thresh = " << 1e-6 * std::max(1.0, c) << std::endl;

        if(std::abs(newc - c) < 1e-6 * std::max(1.0, c))
            break;

        c = newc;
    }

    return res;
}

// [[Rcpp::export]]
NumericMatrix prox_lp_mat(NumericMatrix vv, double p, double alpha, int maxit = 100)
{
    const int n = vv.nrow();
    const double* v = vv.begin();

    NumericMatrix res = vv / (1.0 + alpha);
    double* x = res.begin();
    const double cp = 2.0 / p - 1.0;
    double c = 0.0;

    NumericMatrix xp(n, n);
    for(int j = 0; j < n; j++)
    {
        for(int i = j; i < n; i++)
        {
            xp(i, j) = std::pow(std::abs(res(i, j)), p);
            xp(j, i) = xp(i, j);
        }
    }
    double psum = Rcpp::sum(xp);
    double newc = alpha * std::pow(psum, cp);

    for(int it = 0; it < maxit; it++)
    {
        // Rcpp::Rcout << "iter = " << it << std::endl;

        for(int j = 0; j < n; j++)
        {
            const int ind = j * (n + 1);
            const double signv = (v[ind] > 0.0) - (v[ind] < 0.0);
            x[ind] = signv * solve_equation(newc, p, std::abs(v[ind]));
            const double newxip = std::pow(std::abs(x[ind]), p);
            psum += (newxip - xp[ind]);
            newc = alpha * std::pow(psum, cp);
            xp[ind] = newxip;

            for(int i = j + 1; i < n; i++)
            {
                const int ind = j * n + i;
                const int indt = i * n + j;
                const double signv = (v[ind] > 0.0) - (v[ind] < 0.0);
                x[ind] = signv * solve_equation(newc, p, std::abs(v[ind]));
                x[indt] = x[ind];
                const double newxip = std::pow(std::abs(x[ind]), p);
                psum += 2 * (newxip - xp[ind]);
                newc = alpha * std::pow(psum, cp);

                xp[ind] = newxip;
                xp[indt] = newxip;
            }
        }

        // Rcpp::Rcout << newc << std::endl;
        // Rcpp::Rcout << "diff = " << std::abs(newc - c) << ", thresh = " << 1e-6 * std::max(1.0, c) << std::endl;

        if(std::abs(newc - c) < 1e-6 * std::max(1.0, c))
            break;

        c = newc;
    }

    return res;
}
