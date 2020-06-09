#ifndef GRADFPS_PROX_LP_H
#define GRADFPS_PROX_LP_H

#include "common.h"
#include <boost/math/tools/roots.hpp>
#include <boost/fusion/tuple.hpp>

// Solve f(x) = c * x^(p - 1) + x = v, c > 0, 1 < p < 2, x > 0, v > 0
// f'(x) = (p - 1) * c * x^(p - 2) + 1
// f''(x) = (p - 1) * (p - 2) * c * x^(p - 3)
class sub_equation1
{
private:
    const double m_c;
    const double m_pm1;
    const double m_pm1c;
    const double m_v;

public:
    sub_equation1(double c, double p, double v) :
        m_c(c), m_pm1(p - 1.0), m_pm1c(m_pm1 * c), m_v(v)
    {}

    std::pair<double, double> operator()(const double& x) const
    {
        const double xpm1 = std::pow(x, m_pm1);
        const double xpm2 = xpm1 / x;
        const double fx = m_c * xpm1 + x - m_v;
        const double dx = m_pm1c * xpm2 + 1.0;
        // Rcpp::Rcout << "eq1: " << fx << std::endl;
        return std::make_pair(fx, dx);
    }
};

// Solve g(u) = u^r + c * u = v, c > 0, r > 1, u > 0, v > 0
// g'(u) = r * u^(r - 1) + c
// g''(u) = r * (r - 1) * u^(r - 2)
class sub_equation2
{
private:
    const double m_c;
    const double m_r;
    const double m_rr1;
    const double m_v;

public:
    sub_equation2(double c, double r, double v) :
        m_c(c), m_r(r), m_rr1(r * (r - 1.0)), m_v(v)
    {}

    boost::fusion::tuple<double, double, double> operator()(const double& u) const
    {
        const double urm1 = std::pow(u, m_r - 1.0);
        const double gu = urm1 * u + m_c * u - m_v;
        const double du = m_r * urm1 + m_c;
        const double d2u = m_rr1 * urm1 / u;
        // Rcpp::Rcout << "eq2: u = " << u << ", gu = " << gu << std::endl;
        return boost::fusion::make_tuple(gu, du, d2u);
    }
};

// Solve equation c * x^(p - 1) + x - v = 0, c > 0, 1 < p < 2, x > 0, v > 0
inline double solve_equation(double c, double p, double v, double x0)
{
    // Edge case: if v is small, return 0
    if(v < 1e-8)
        return 0.0;

    // f(x) = c * x^(p - 1) + x is increasing in x, f(1) = c + 1
    // If v >= f(1), solve f(x) = v, 1 < x < v
    // If v < f(1), u = x^(p - 1), solve g(u) = u^r + c * u = v, r = 1 / (p - 1), 0 < u < 1
    const double f1 = c + 1.0;
    if(v >= f1)
    {
        // Guess is based on f(x) ~= c + x
        const double guess = (x0 >= 1.0 && x0 <= v) ? x0: (v - c);
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
    double guess = 0.5;
    if(x0 < 1.0)
    {
        guess = std::pow(x0, p - 1.0);
    } else {
        guess = (v <= g0) ? (v / c) : (std::pow(v - 0.9 * cu0, p - 1.0));
        guess = std::min(guess, 1.0);
    }
    const double lb = 0.0;
    const double ub = 1.0;
    const int digits = 30;
    boost::uintmax_t maxit = 30;
    const double res = boost::math::tools::schroder_iterate(sub_equation2(c, r, v), guess, lb, ub, digits, maxit);
    return std::pow(res, r);
}



// Proximal operator of 0.5 * ||x||_p^2
inline void prox_lp_impl(RefConstVec vv, double p, double alpha, RefVec res,
                         double eps = 1e-6, int maxiter = 100, int verbose = 0)
{
    const int n = vv.size();
    const double* v = vv.data();

    res.noalias() = vv / (1.0 + alpha);
    double* x = res.data();
    const double cp = 2.0 / p - 1.0;
    double c = 0.0;

    Vector xp(n);
    xp.array() = res.array().abs().pow(p);
    double psum = xp.sum();
    double newc = alpha * std::pow(psum, cp);

    for(int it = 0; it < maxiter; it++)
    {
        if(verbose > 0)
            Rcpp::Rcout << "iter = " << it;

        for(int i = 0; i < n; i++)
        {
            const double signv = (v[i] > 0.0) - (v[i] < 0.0);
            x[i] = signv * solve_equation(newc, p, std::abs(v[i]), std::abs(x[i]));
            const double newxip = std::pow(std::abs(x[i]), p);
            psum += (newxip - xp[i]);
            newc = alpha * std::pow(psum, cp);

            xp[i] = newxip;
        }

        if(verbose > 0)
            Rcpp::Rcout << ", diff = " << std::abs(newc - c) << ", thresh = " << eps * std::max(1.0, c) << std::endl;

        if(std::abs(newc - c) < eps * std::max(1.0, c))
            break;

        c = newc;
    }
}

// Proximal operator of 0.5 * ||x||_p^2, applied to a symmetric matrix
inline void prox_lp_mat_impl(RefConstMat vv, double p, double alpha, RefMat res,
                             double eps = 1e-6, int maxiter = 100, int verbose = 0)
{
    const int n = vv.rows();
    const double* v = vv.data();

    res.noalias() = vv / (1.0 + alpha);
    double* x = res.data();
    const double cp = 2.0 / p - 1.0;
    double c = 0.0;

    Matrix xpow(n, n);
    double* xp = xpow.data();
    for(int j = 0; j < n; j++)
    {
        for(int i = j; i < n; i++)
        {
            xpow.coeffRef(i, j) = std::pow(std::abs(res.coeff(i, j)), p);
            xpow.coeffRef(j, i) = xpow.coeff(i, j);
        }
    }
    double psum = xpow.sum();
    double newc = alpha * std::pow(psum, cp);

    for(int it = 0; it < maxiter; it++)
    {
        if(verbose > 0)
            Rcpp::Rcout << "iter = " << it;

        for(int j = 0; j < n; j++)
        {
            // Diagonal elements
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

        if(verbose > 0)
            Rcpp::Rcout << ", diff = " << std::abs(newc - c) << ", thresh = " << eps * std::max(1.0, c) << std::endl;

        if(std::abs(newc - c) < eps * std::max(1.0, c))
            break;

        c = newc;
    }
}


#endif  // GRADFPS_PROX_LP_H
