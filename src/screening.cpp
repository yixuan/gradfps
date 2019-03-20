#include "common.h"
#include "active.h"

using Rcpp::NumericVector;


// Comparator for std::sort(), in descending order
inline bool triple_comparator_maxoffdiag(const Triple& t1, const Triple& t2)
{
    return t1.m_max_off_diag > t2.m_max_off_diag;
}

// Get the range of lambda by selecting the size of active sets
// [[Rcpp::export]]
NumericVector lambda_range(MapMat S, int d, int act_size_min, int act_size_max)
{
    // Dimension of the covariance matrix
    const int n = S.rows();
    const int p = S.cols();
    if(n != p)
        Rcpp::stop("S must be square");

    act_size_min = std::max(act_size_min, d);
    act_size_max = std::min(act_size_max, p);

    // Analyze covariance pattern
    ActiveSet act_set(S);
    act_set.analyze_pattern();
    std::vector<Triple> pattern = act_set.pattern();
    std::sort(pattern.begin() + d, pattern.end(), triple_comparator_maxoffdiag);

    return NumericVector::create(
        pattern[act_size_max - 1].m_max_off_diag,
        pattern[act_size_min - 1].m_max_off_diag
    );
}
