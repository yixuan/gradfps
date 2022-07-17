library(RSpectra)
library(gradfps)
library(mvtnorm)
library(Matrix)
library(ggplot2)
library(dplyr)
library(showtext)
font_add_google("Lato")

source("online_algorithms.R")

# Simulate the true covariance matrix
simulate_cov = function(p, s1, s2, d)
{
    # Simulate eigenvectors
    ev = matrix(rnorm(p^2), p, p)
    ev[, 1:2] = 0
    ev[1:s1, 1] = runif(s1, 0.9 / sqrt(s1), 1.1 / sqrt(s1))
    ev[(s1 + 1):(s1 + s2), 2] = runif(s2, 0.9 / sqrt(s2), 1.1 / sqrt(s2))
    ev = qr.Q(qr(ev))

    # Simulate eigenvalues
    sigmas = c(12, 6, runif(p - 2, 0, 2))

    # True covariance
    Sigma = ev %*% diag(sigmas) %*% t(ev)

    # True projection matrix
    V = eigs_sym(Sigma, d, which = "LA")$vectors
    Pi = tcrossprod(V)

    list(Sigma = Sigma, proj = Pi, evecs = V)
}

# Visualize the covariance matrix
visualize_cov = function(Sigma, Pi)
{
    # Visualize Sigma
    g1 = view_matrix(Sigma[1:200, 1:200], legend_title = "Covariance\nCoefficient")
    print(g1)

    # Visualize projection matrix
    g2 = view_matrix(Pi[1:200, 1:200], legend_title = "Covariance\nCoefficient")
    print(g2)

    # Visualize eigenvectors
    d = 5
    V = eigs_sym(Sigma, d, which = "LA")$vectors
    g3 = view_evec(-V[1:200, ], asp = 0.4, bar_height = 8)
    print(g3)

    # Print eigenvalues
    print(eigs_sym(Sigma, 10)$values)
}

# Simulate data set according to the covariance matrix
simulate_data = function(n, Sigma)
{
    rmvnorm(n, sigma = Sigma)
}

# Simulate the initial values for the online algorithms
simulate_init = function(n, d, Sigma)
{
    z = rmvnorm(n, sigma = Sigma)
    S = crossprod(z) / n
    e = eigs_sym(S, d, which = "LA")
    proj = tcrossprod(e$vectors)

    list(norm = norm(S, type = "F"), evals = e$values, evecs = e$vectors, proj = proj)
}

N = 5000  # number of total data points
p = 800   # dimension of the variables
s1 = 20   # size of the first signal group
s2 = 15   # size of the second signal group
d = 2     # number of principal components

# Simulate Sigma
set.seed(123)
sim_sigma = simulate_cov(p, s1, s2, d)
Sigma = sim_sigma$Sigma
Pi = sim_sigma$proj
visualize_cov(Sigma, Pi)

# Generate data
z = simulate_data(N, Sigma)

# Initialization
lambda = 0.1
lr = 0.1
dat_init = simulate_init(100, d, Sigma)
view_matrix(dat_init$proj[1:200, 1:200])

# The online learning process
lossfn = function(x, zt1, lambda, nu, d, feasible = FALSE)
{
    p = nrow(x)
    res = -sum(zt1 * (x %*% zt1)) + lambda * sum(abs(x))
    if(!feasible)
    {
        proj = prox_fantope(A = matrix(0, p, p), B = x, alpha = 1, d = d)
        res = res + nu * norm(x - proj, type = "F")
    }
    res
}

gradfps = GradFPS$new(dat_init, lambda = lambda, lr = lr)
nu = lambda * p + norm(Sigma, type = "F") + 1
regret_gradfps = c()

for(i in 1:(N - 1))
{
    cat("===== iter = ", i, " =====\n", sep = "")

    # Current data point
    zt = z[i, , drop = FALSE]
    St = crossprod(zt)
    # Historical data
    S1t = crossprod(z[1:i, , drop = FALSE]) / i
    # Next data for validation
    zt1 = z[i + 1, ]

    # Online GradFPS
    proj_gradfps = gradfps$step(iter = i, zt = zt, S1t = S1t)

    # Compute regret value
    reg = lossfn(proj_gradfps, zt1, lambda, nu, d) -
        lossfn(Pi, zt1, lambda, nu, d, feasible = TRUE)
    regret_gradfps = c(regret_gradfps, reg)
}

plot(regret_gradfps, type = "l")
plot(cumsum(regret_gradfps) / 1:(N - 1), type = "l")
plot(log(1:(N - 1)), log(cumsum(regret_gradfps) / 1:(N - 1)), type = "l")
abline(8.1, -0.5, col = "red")
view_matrix(proj_gradfps[1:200, 1:200])

save(regret_gradfps, proj_gradfps, file = "result/rates.RData")



load("result/rates.RData")
gdat = tibble(iter = 1:(N - 1), avgreg = cumsum(regret_gradfps) / iter,
              log_iter = log10(iter), log_avgreg = log10(avgreg))
ggplot(gdat, aes(x = log_iter, y = log_avgreg)) +
    geom_line() +
    geom_abline(slope = -0.5, intercept = 3.52, color = "red") +
    xlab(expression(plain("Log")[10](T))) +
    ylab(expression(plain("Log")[10](R * group("{", list(X[t], Z[t], T), "}")/T))) +
    theme_bw(base_size = 22, base_family = "Lato")
# showtext_auto()
# ggsave("rates.pdf", width = 10, height = 5)
