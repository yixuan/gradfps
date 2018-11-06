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
