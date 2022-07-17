# Code to reproduce Figure 1

library(Matrix)
library(ggplot2)
library(mvtnorm)
library(RSpectra)
library(gradfps)  # devtools::install_github("yixuan/gradfps")
library(showtext)
font_add_google("Lato")
showtext_auto()

n = 25
p = 100
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

# Visualization of the true covariance matrix
pal = c("#d00003", "#f13a3c", "#f56566", "#f49c9c",
        "#facdcd", "#fae1e1", "#fcf1f1",
        "#ffffff",
        "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
        "#2171b5", "#08519c", "#08306b")
view_matrix(Sigma, legend_title = "True\nCovariance\nCoefficient\n") +
    guides(fill = guide_colorbar(barwidth = 1.8, barheight = 20)) +
    theme_bw(base_size = 22, base_family = "Lato") +
    theme(axis.title = element_blank(),
          legend.title = element_text(face = "bold"))
# ggsave("true_cov_p100.pdf", width = 9, height = 7)

# Eigenvectors
d = 5
V = eigs_sym(Sigma, d)$vectors
view_evec(-V, asp = 0.4, legend_title = "Factor Loading    ") +
    guides(fill = guide_colorbar(barwidth = 30, barheight = 1.8)) +
    theme_bw(base_size = 26, base_family = "Lato") +
    theme(axis.title = element_text(face = "bold"),
          legend.position = "top",
          legend.title = element_text(face = "bold"))
# ggsave("true_eigvec_p100.pdf", width = 12, height = 7)

# Generate data
set.seed(123)
z = rmvnorm(n, sigma = Sigma)
# Smat = cov(z)
Smat = crossprod(z) / n
view_matrix(Smat, legend_title = "Sample\nCovariance\nCoefficient")
