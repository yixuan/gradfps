#ifndef FASTFPS_ACTIVE_H
#define FASTFPS_ACTIVE_H

#include "common.h"
#include "symmat.h"
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

    // Flatten the incremental active sets into a single vector
    inline std::vector<int> flattened_active_set() const
    {
        std::vector<int> res;
        res.reserve(m_p);
        const int nlambda = m_act_set.size();

        for(int i = 0; i < nlambda; i++)
            res.insert(res.end(), m_act_set[i].begin(), m_act_set[i].end());

        return res;
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

    inline const std::vector< std::vector<int> >& incremental_active_set() const
    {
        return m_act_set;
    }

    // Return the incremental active sets to R, mostly for debugging
    inline Rcpp::List incremental_active_set_to_r() const
    {
        const int nlambda = m_act_set.size();
        Rcpp::List res(nlambda);
        for(int i = 0; i < nlambda; i++)
        {
            res[i] = Rcpp::wrap(m_act_set[i]);
        }

        return res;
    }
    inline Rcpp::IntegerVector flattened_active_set_to_r() const
    {
        return Rcpp::wrap(flattened_active_set());
    }

    // Compute the largest submatrix (the one associated with the smallest lambda)
    inline std::vector<int> compute_submatrix(SymMat& sub_mat)
    {
        std::vector<int> act = flattened_active_set();
        const int pa = act.size();
        MatrixXd sub_mat_data(pa, pa);

        // We ony write the lower triangular part
        for(int j = 0; j < pa; j++)
        {
            for(int i = j; i < pa; i++)
            {
                sub_mat_data.coeffRef(i, j) = m_mat.coeff(act[i], act[j]);
            }
        }

        // Transfer data to sub_mat
        sub_mat.swap(sub_mat_data);
        sub_mat.set_dim(pa);

        return act;
    }
};


#endif // FASTFPS_ACTIVE_H
