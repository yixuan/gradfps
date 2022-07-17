# Code to reproduce Figure 8

library(gradfps)  # devtools::install_github("yixuan/gradfps")
library(RSpectra)
library(mvtnorm)
library(Matrix)
library(ggplot2)
library(showtext)
showtext_auto()

n = 50
p = 200
d1 = 20  # first group
d2 = 15  # second group

# Generate covariance matrices
# Simulate eigenvectors
set.seed(123)
ev = matrix(rnorm(p^2), p, p)
ev[, 1:2] = 0
ev[1:d1, 1] = runif(d1, 0.9 / sqrt(d1), 1.1 / sqrt(d1))
ev[(d1 + 1):(d1 + d2), 2] = runif(d2, 0.9 / sqrt(d2), 1.1 / sqrt(d2))
ev = qr.Q(qr(ev))
# Simulate eigenvalues
sigmas = c(12, 6, runif(p - 2, 0, 2))
# True covariance
Sigma = ev %*% diag(sigmas) %*% t(ev)

# Visualization
view_matrix(Sigma[1:200, 1:200], legend_title = "Covariance\nCoefficient")

# Eigenvalues
eigs_sym(Sigma, 10)$values

# Eigenvectors
d = 5
V = eigs_sym(Sigma, d)$vectors
Pi = tcrossprod(V[, 1:2])
view_evec(-V[1:200, ], asp = 0.4, bar_height = 8)

# Generate data
set.seed(123)
z = rmvnorm(n, sigma = Sigma)
# Smat = cov(z)
Smat = crossprod(z) / n

d = 2
lambda = 0.5 * sqrt(log(p) / n)
e = eigs_sym(Smat, d, which = "LA")
x0 = tcrossprod(e$vectors)
view_matrix(x0[1:200, 1:200])

L = norm(Smat, type = "F") + lambda * p
mu_max = (sqrt(2) + 1) * (L + 1) * sqrt(p / (d + 1))

# ratios = c(0.01, 0.02, 0.05, 0.07, 0.09, 0.1, 0.5, 1)
ratios = c(0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.1, 1)
mus = ratios * mu_max
projs = list()
for(mu in mus)
{
    res = gradfps_prox(Smat, d, lambda, x0, lr = 0.05, maxiter = 100,
                       control = list(mu = mu, verbose = 1))
    projs = c(projs, list(res$projection))
}
errs = sapply(projs, function(x) norm(x - projs[[length(projs)]], type = "F"))
plot(errs)

constants = ifelse(ratios == 1, "", as.character(ratios))
constants = factor(constants, levels = constants)
gdat = data.frame(constant = constants, err = errs)
ggplot(gdat, aes(x = constant, y = err)) +
    geom_bar(stat = "identity", width = 0.5) +
    xlab("Penalty Parameter μ") +
    ylab("Computational Error") +
    scale_x_discrete(labels = scales::math_format(expr = .x * mu["max"])) +
    theme_bw(base_size = 22)
# ggsave("constants.pdf", width = 13.5, height = 5)
