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
            x[i] = signv * solve_equation(newc, p, std::abs(v[i]), std::abs(x[i]));
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
            x[ind] = signv * solve_equation(newc, p, std::abs(v[ind]), std::abs(x[ind]));
            const double newxip = std::pow(std::abs(x[ind]), p);
            psum += (newxip - xp[ind]);
            newc = alpha * std::pow(psum, cp);
            xp[ind] = newxip;

            for(int i = j + 1; i < n; i++)
            {
                const int ind = j * n + i;
                const int indt = i * n + j;
                const double signv = (v[ind] > 0.0) - (v[ind] < 0.0);
                x[ind] = signv * solve_equation(newc, p, std::abs(v[ind]), std::abs(x[ind]));
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
