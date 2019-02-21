#include "common.h"
#include <boost/math/tools/roots.hpp>

using Rcpp::NumericVector;
using Rcpp::NumericMatrix;

// Solve f(x) = c * x^(p - 1) + x = v, c > 0, 1 < p < 2, x > 0, v > 0
// f'(x) = (p - 1) * c * x^(p - 2) + 1
class sub_equation1
{
private:
    const double m_c;
    const double m_pm1;
    const double m_v;

public:
    sub_equation1(double c, double p, double v) :
        m_c(c), m_pm1(p - 1.0), m_v(v)
    {}

    std::pair<double, double> operator()(const double& x) const
    {
        const double xpm1 = std::pow(x, m_pm1);
        const double fx = m_c * xpm1 + x - m_v;
        const double dx = m_pm1 * m_c * xpm1 / x + 1.0;
        // Rcpp::Rcout << "eq1: " << fx << std::endl;
        return std::make_pair(fx, dx);
    }
};

// Solve g(u) = u^r + c * u = v, c > 0, r > 1, u > 0, v > 0
// g'(u) = r * u^(r - 1) + c
class sub_equation2
{
private:
    const double m_c;
    const double m_r;
    const double m_v;

public:
    sub_equation2(double c, double r, double v) :
        m_c(c), m_r(r), m_v(v)
    {}

    std::pair<double, double> operator()(const double& u) const
    {
        const double urm1 = std::pow(u, m_r - 1.0);
        const double gu = urm1 * u + m_c * u - m_v;
        const double du = m_r * urm1 + m_c;
        // Rcpp::Rcout << "eq2: u = " << u << ", gu = " << gu << std::endl;
        return std::make_pair(gu, du);
    }
};

// Solve equation c * x^(p - 1) + x - v = 0, c > 0, 1 < p < 2, x > 0, v > 0
inline double solve_equation(double c, double p, double v)
{
    // f(x) = c * x^(p - 1) + x is increasing in x, f(1) = c + 1
    // If v >= f(1), solve f(x) = v, 1 < x < v
    // If v < f(1), u = x^(p - 1), solve g(u) = u^r + c * u = v, r = 1 / (p - 1), 0 < u < 1
    const double f1 = c + 1.0;
    if(v >= f1)
    {
        // Guess is based on f(x) ~= c + x
        const double guess = v - c;
        const double lb = 1.0;
        const double ub = v;
        const int digits = 30;
        boost::uintmax_t maxit = 30;
        const double res = boost::math::tools::newton_raphson_iterate(sub_equation1(c, p, v), guess, lb, ub, digits, maxit);
        return res;
    }

    // Let u0 be such that u0^r = 0.1 * c * u0
    // Guess is based on g(u) ~= c * u, u <= u0
    //                        ~= u^r + 0.9 * c * u0, u0 < u <= 1
    const double r = 1.0 / (p - 1.0);
    const double u0 = std::pow(0.1 * c, 1.0 / (r - 1.0));
    const double cu0 = c * u0;
    const double g0 = 1.1 * cu0;
    double guess = (v <= g0) ? (v / c) : (std::pow(v - 0.9 * cu0, 1.0 / r));
    guess = std::min(guess, 1.0);
    const double lb = 0.0;
    const double ub = 1.0;
    const int digits = 30;
    boost::uintmax_t maxit = 30;
    const double res = boost::math::tools::newton_raphson_iterate(sub_equation2(c, r, v), guess, lb, ub, digits, maxit);
    return std::pow(res, r);
}

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
