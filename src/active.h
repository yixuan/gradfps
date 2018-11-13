#ifndef FASTFPS_ACTIVE_H
#define FASTFPS_ACTIVE_H

#include "common.h"
#include <vector>

class Triple
{
public:
    int    m_ind;          // index of the variable
    double m_diag;         // diagonal element
    double m_max_off_diag; // max absolute value of off-diagonal elements

    Triple(int ind, double diag, double max_off_diag) :
        m_ind(ind), m_diag(diag), m_max_off_diag(max_off_diag)
    {}
};

// Comparator for std::sort(), in descending order
inline bool triple_comparator(const Triple& t1, const Triple& t2)
{
    return t1.m_diag > t2.m_diag;
}

class ActiveSet
{
private:
    typedef Eigen::VectorXd Vector;
    typedef const Eigen::Ref<const Eigen::MatrixXd> ConstRefMat;
    typedef Eigen::Map<const Eigen::MatrixXd> ConstMapMat;

    const int                        m_p;
    ConstMapMat                      m_mat;
    std::vector<Triple>              m_pattern;
    std::vector< std::vector<int> >  m_act_set;
    Eigen::MatrixXd                  m_sub_mat;

    // Find max absolute value in {x[0], ..., x[p-1]} \ {x[j]}
    inline double max_abs_excluding_j(const double* x, int j) const
    {
        // |x[0]|
        double maxabs = std::abs(x[0]);
        // x[1], ..., x[j-1]
        for(int i = 1; i < j; i++)
        {
            const double xi_abs = std::abs(x[i]);
            if(xi_abs > maxabs)
                maxabs = xi_abs;
        }
        // x[j+1], ..., x[p-1]
        for(int i = j + 1; i < m_p; i++)
        {
            const double xi_abs = std::abs(x[i]);
            if(xi_abs > maxabs)
                maxabs = xi_abs;
        }

        return maxabs;
    }

public:
    ActiveSet(ConstRefMat& x) :
        m_p(x.rows()), m_mat(x.data(), m_p, m_p)
    {
        m_pattern.reserve(m_p);
    }

    // Analyze the pattern of a covariance matrix
    inline void analyze_pattern()
    {
        m_pattern.clear();

        for(int j = 0; j < m_p; j++)
        {
            // For each column j, find the element x(i, j), i != j
            // that has the largest absolute value
            const double max_off_diag = max_abs_excluding_j(&m_mat.coeffRef(0, j), j);
            m_pattern.push_back(Triple(j, m_mat.coeff(j, j), max_off_diag));
        }

        std::sort(m_pattern.begin(), m_pattern.end(), triple_comparator);
    }

    // Assume that lambda is in decreasing order
    // We start from the largest lambda and compute its active set
    // When moving to the next lambda, we keep the previous active set in order,
    // and expand it by including more variables
    inline void find_active(int d, const Vector& lambda)
    {
        const int nlambda = lambda.size();
        m_act_set.resize(nlambda);

        // First compute the active set for the largest lambda
        // Selection criterion: (a) or (b)
        // (a) Variable j has Mj = max_{i != j} |x(i, j)| > lambda
        // (b) j < d
        // Here variables are already sorted according to their variances
        // (diagonal elements of the covariance matrix)
        // m_pattern[j].m_ind is the original index of variable j in the covariance matrix
        double curr_lambda = lambda[0];
        m_act_set[0].clear();
        for(int j = 0; j < m_p; j++)
        {
            const double Mj = m_pattern[j].m_max_off_diag;
            if((Mj > curr_lambda) || (j < d))
                m_act_set[0].push_back(m_pattern[j].m_ind);
        }

        // Then compute the incremental active sets
        // m_act_set[l] contains the difference set of active set l and active set (l-1)
        for(int l = 1; l < nlambda; l++)
        {
            // Variable j (j >= d) is included if lambda[l] < Mj <= lambda[l-1]
            // This is because:
            // (a) {j: j < d} has been included in m_act_set[0], so we can start from j = d
            // (b) For j >= d, if Mj > lambda[l-1], then it will be included in m_act_set[l-1]
            // (c) In order to include variable j, we must have Mj > lambda[l]
            curr_lambda = lambda[l];
            const double prev_lambda = lambda[l - 1];
            m_act_set[l].clear();
            for(int j = d; j < m_p; j++)
            {
                const double Mj = m_pattern[j].m_max_off_diag;
                if((Mj <= prev_lambda) && (Mj > curr_lambda))
                    m_act_set[l].push_back(m_pattern[j].m_ind);
            }
        }
    }

    // Return the incremental active sets to R, mostly for debugging
    inline Rcpp::List active_set_to_r() const
    {
        const int nlambda = m_act_set.size();
        Rcpp::List res(nlambda);
        for(int i = 0; i < nlambda; i++)
        {
            res[i] = Rcpp::wrap(m_act_set[i]);
        }

        return res;
    }
};

// Analyze the pattern of the matrix
inline void analyze_pattern(const MapMat& x, std::vector<Triple>& pattern)
{
    const int n = x.rows();
    pattern.clear();
    pattern.reserve(n);

    for(int j = 0; j < n; j++)
    {
        double max_off_diag = 0.0;
        for(int i = 0; i < n; i++)
        {
            const double xij_abs = std::abs(x.coeff(i, j));
            if((i != j) && (xij_abs > max_off_diag))
                max_off_diag = xij_abs;
        }
        pattern.push_back(Triple(j, x.coeff(j, j), max_off_diag));
    }

    std::sort(pattern.begin(), pattern.end(), triple_comparator);
}

// Find the active set
inline void find_active(
    const std::vector<Triple>& pattern, int d, double lambda,
    std::vector<int>& act
)
{
    const int n = pattern.size();
    act.clear();

    for(int i = 0; i < n; i++)
    {
        if((pattern[i].m_max_off_diag > lambda) || (i < d))
            act.push_back(pattern[i].m_ind);
    }
}

// Create a submatrix based on the active set
// Assume x is symmetric and only the lower triangular part is referenced
inline void submatrix_act(const MapMat& x, const std::vector<int>& act, MatrixXd& subx)
{
    const int p = act.size();
    subx.resize(p, p);

    for(int j = 0; j < p; j++)
    {
        for(int i = j; i < p; i++)
        {
            subx.coeffRef(i, j) = x.coeff(act[i], act[j]);
        }
    }

    // Copy to the upper triangular part
    subx.triangularView<Eigen::StrictlyUpper>() = subx.triangularView<Eigen::StrictlyLower>().transpose();
}


#endif // FASTFPS_ACTIVE_H
