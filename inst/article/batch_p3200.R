library(gradfps)  # devtools::install_github("yixuan/gradfps")
library(fps)      # devtools::install_github("yixuan/fps")
library(RSpectra)
library(mvtnorm)
library(Matrix)
library(ggplot2)

n = 800
p = 3200
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

res = gradfps_subgrad(Smat, d, lambda, x0, lr = 0.01, maxiter = 100,
                      control = list(mu = 100, eps_abs = 1e-5, eps_rel = 1e-5, verbose = 1))
view_matrix(res$projection[1:200, 1:200])



# FPS
res_fps = fps:::fps_benchmark(
    Smat, d, lambda, x0, Pi, rho = -1, maxiter = 60, tolerance = 1e-3, verbose = 0
)
# plot(res_fps$errors, xlab = "Iteration", ylab = "Error")
# plot(cumsum(res_fps$times), res_fps$errors, xlab = "Time", ylab = "Error")
# view_matrix(res_fps$projection[1:100, 1:100])

# Gradient FPS
res_subgrad = gradfps_subgrad_benchmark(
    Smat, Pi, d, lambda, x0, lr = 0.01, maxiter = 30,
    control = list(eps_abs = 1e-5, eps_rel = 1e-5, verbose = 1)
)
# plot(res_subgrad$err_v, xlab = "Iteration", ylab = "Error")
# plot(cumsum(res_subgrad$times), res_subgrad$err_v, xlab = "Time", ylab = "Error")
# view_matrix(res_subgrad$projection[1:100, 1:100])

e = eigs_sym(res_subgrad$projection, d, which = "LA")
x0w = tcrossprod(e$vectors)

res_grad = gradfps_prox_benchmark(
    Smat, Pi, d, lambda, x0w, lr = 0.01, maxiter = 60,
    control = list(fan_maxinc = 10, verbose = 0)
)
# plot(res_grad$err_v, xlab = "Iteration", ylab = "Error")
# plot(cumsum(res_grad$times), res_grad$err_v, xlab = "Time", ylab = "Error")
# view_matrix(res_grad$projection[1:100, 1:100])

gradfps_times = c(res_subgrad$times, res_grad$times)
gradfps_errors = c(res_subgrad$err_v, res_grad$err_v)

gdat = data.frame(
    method = c(rep("ADMM-FPS", length(res_fps$times)), rep("GradFPS", length(gradfps_times))),
    time = c(cumsum(res_fps$times), cumsum(gradfps_times)),
    err = c(res_fps$errors, gradfps_errors),
    par = sprintf("n = %d, p = %d", n, p)
)
ggplot(gdat, aes(x = time, y = err)) +
    geom_line(aes(group = method, color = method, linetype = method), size = 1) +
    scale_linetype_manual("Method", values = c("32", "solid")) +
    scale_color_hue("Method") +
    guides(color = guide_legend(keyheight = 2, keywidth = 3),
           linetype = guide_legend(keyheight = 2, keywidth = 3)) +
    xlab("Elapsed Time (s)") + ylab("Estimation Error") +
    ggtitle(sprintf("n = %d, p = %d", n, p)) +
    theme_bw(base_size = 20) +
    theme(plot.title = element_text(hjust = 0.5))

write.csv(gdat, sprintf("result/n_%d_p_%d.csv", n, p))
