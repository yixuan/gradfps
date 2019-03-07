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
    const double res = boost::math::tools::schroder_iterate(sub_equation2(c, r, v), guess, lb, ub, digits, maxit);
    return std::pow(res, r);
}


#endif  // GRADFPS_PROX_LP_H
