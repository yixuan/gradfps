## GradFPS

The `gradfps` R package implements the gradient-based Fantope projection and selection
algorithm, a convex formulation of sparse principal component analysis. The algorithm
is based on the paper
[Gradient-based Sparse Principal Component Analysis with Extensions to Online Learning](https://arxiv.org/abs/1911.08048)
by Yixuan Qiu, Jing Lei, and Kathryn Roeder.

### Installation

Currently `gradfps` has not been submitted to CRAN, but it can be installed just like
any other R package hosted on GibHub. For `devtools` users, the following command
should work on most platforms:

```r
library(devtools)
install_github("yixuan/gradfps")
```

Note that a C++ compiler that supports the C++11 standard is needed.
For best performance, it is **strongly suggested** linking your R to the
[OpenBLAS](https://www.openblas.net/) library for matrix computation.
You can achieve this with the help of the
[ropenblas](https://prdm0.github.io/ropenblas/) package.

### Example

Here we compare our implementation with the original ADMM-based algorithm
(via the package [fps](https://github.com/vqv/fps) developed by
[Vincent Vu](http://www.vince.vu/)).

Note that for a fair comparison, I added a new function `fps_benchmark()` to the
original **fps** package, and the example below requires the installation of my
[modified version](https://github.com/yixuan/fps):

```r
devtools::install_github("yixuan/fps")
```

And then run the example code:

```r
library(gradfps)
library(fps)  # The modified version
library(RSpectra)
library(mvtnorm)
library(Matrix)
library(ggplot2)

n = 200
p = 800
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
view_matrix(Sigma[1:100, 1:100], legend_title = "True\nCovariance\nMatrix")
```

<img src="https://i.imgur.com/jpvb5dp.png" alt="True covariance matrix" width="350px" />

```r
# Eigenvectors
d = 5
V = eigs_sym(Sigma, d)$vectors
Pi = tcrossprod(V[, 1:2])

# Visualization of the true eigenvectors
view_evec(-V[1:200, ], asp = 0.4, bar_height = 8)
```

<img src="https://i.imgur.com/EqON45E.png" alt="True eigenvectors" width="600px" />

```r
# Generate data
set.seed(123)
z = rmvnorm(n, sigma = Sigma)
Smat = crossprod(z) / n

d = 2
lambda = 0.5 * sqrt(log(p) / n)
e = eigs_sym(Smat, d, which = "LA")
# Initial value, should be noisy
x0 = tcrossprod(e$vectors)

# ADMM FPS
res_fps = fps::fps_benchmark(
    Smat, d, lambda, x0, Pi, rho = -1, maxiter = 60, tolerance = 1e-3, verbose = 0
)
# Verify the result
view_matrix(res_fps$projection[1:100, 1:100], legend_title = "ADMM-FPS\nProjection\nMatrix")
```

<img src="https://i.imgur.com/Ob2XHyj.png" alt="ADMM-FPS projection matrix" width="350px" />

```r
# Gradient FPS
res_grad = gradfps_prox_benchmark(
    Smat, Pi, d, lambda, x0, lr = 0.02, maxiter = 60,
    control = list(fan_maxinc = 10, verbose = 0)
)
# Verify the result
view_matrix(res_grad$projection[1:100, 1:100], legend_title = "GradFPS\nProjection\nMatrix")
```

<img src="https://i.imgur.com/F1XSDnH.png" alt="GradFPS projection matrix" width="350px" />

```r
# Compare computational efficiency
gdat = data.frame(
    method = c(rep("ADMM-FPS", length(res_fps$times)), rep("GradFPS", length(res_grad$times))),
    time = c(cumsum(res_fps$times), cumsum(res_grad$times)),
    err = c(res_fps$errors, res_grad$err_v),
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
```

<img src="https://i.imgur.com/SCQYopF.png" alt="Time comparison" width="500px" />

### Citation

Please consider to cite our work if you have used our algorithm or package in your research.

```
@article{qiu2019gradient,
    title={Gradient-based Sparse Principal Component Analysis with Extensions to Online Learning},
    author={Qiu, Yixuan and Lei, Jing and Roeder, Kathryn},
    journal={arXiv preprint arXiv:1911.08048},
    year={2019}
}
```
