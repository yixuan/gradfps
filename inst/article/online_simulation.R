# Code to reproduce Figure 4 and Figure 5

library(RSpectra)
library(gradfps)  # devtools::install_github("yixuan/gradfps")
library(mvtnorm)
library(Matrix)
library(ggplot2)
library(dplyr)
library(showtext)
font_add_google("Lato")
showtext_auto()

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

# Repetitions
simulation_one_run = function(N, p, s1, s2, d, lr = 0.01, save_evec = FALSE)
{
    # Simulate Sigma
    sim_sigma = simulate_cov(p, s1, s2, d)
    Sigma = sim_sigma$Sigma
    Pi = sim_sigma$proj

    # Generate data
    z = simulate_data(N, Sigma)

    # Initialization
    d = 2
    dat_init = simulate_init(10, d, Sigma)

    # The online learning process
    oja = Oja$new(dat_init, lr = lr)
    ipca = IPCA$new(dat_init)
    ccipca = CCIPCA$new(dat_init)
    gradfps = GradFPS$new(dat_init, lambda = sqrt(log(p) / N), lr = lr)

    err_oja = norm(Pi - dat_init$proj, type = "F")
    err_ipca = err_ccipca = err_gradfps = err_oja

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

        # Online PCA based on Oja's method
        proj_oja = oja$step(iter = i, zt = zt, S1t = S1t)
        err_oja = c(err_oja, norm(Pi - proj_oja, type = "F"))

        # IPCA
        proj_ipca = ipca$step(iter = i, zt = zt, S1t = S1t)
        err_ipca = c(err_ipca, norm(Pi - proj_ipca, type = "F"))

        # CCIPCA
        proj_ccipca = ccipca$step(iter = i, zt = zt, S1t = S1t)
        err_ccipca = c(err_ccipca, norm(Pi - proj_ccipca, type = "F"))

        # Online GradFPS
        proj_gradfps = gradfps$step(iter = i, zt = zt, S1t = S1t)
        err_gradfps = c(err_gradfps, norm(Pi - proj_gradfps, type = "F"))
    }

    res = list(err_oja = err_oja, err_ipca = err_ipca,
               err_ccipca = err_ccipca, err_gradfps = err_gradfps)
    if(save_evec)
        res = c(res, list(evec_truth = sim_sigma$evecs,
                          evec_oja = oja$eigenvectors(), evec_ipca = ipca$eigenvectors(),
                          evec_ccipca = ccipca$eigenvectors(), evec_gradfps = gradfps$eigenvectors(),
                          proj_truth = Pi,
                          proj_oja = proj_oja, proj_ipca = proj_ipca,
                          proj_ccipca = proj_ccipca, proj_gradfps = proj_gradfps))
    res
}

# Example
set.seed(123)
run1 = simulation_one_run(N = 200, p = 800, s1 = 20, s2 = 15, d = 2, lr = 0.1, save_evec = TRUE)
err_oja = run1$err_oja
err_ipca = run1$err_ipca
err_ccipca = run1$err_ccipca
err_gradfps = run1$err_gradfps
ylim = range(c(err_oja, err_ipca, err_ccipca, err_gradfps))
plot(err_gradfps, ylim = ylim, type = "l")
lines(err_oja, col = "red")
lines(err_ipca, col = "orange")
lines(err_ccipca, col = "blue")

view_matrix(run1$proj_oja[1:200, 1:200])
view_matrix(run1$proj_ipca[1:200, 1:200])
view_matrix(run1$proj_ccipca[1:200, 1:200])
view_matrix(run1$proj_gradfps[1:200, 1:200])
view_matrix(run1$proj_truth[1:200, 1:200])

view_evec(run1$evec_oja[1:200, ])
view_evec(run1$evec_ipca[1:200, ])
view_evec(run1$evec_ccipca[1:200, ])
view_evec(run1$evec_gradfps[1:200, ])
view_evec(run1$evec_truth[1:200, ])

# Simulation with p=800
set.seed(123)
nrun = 10
runs = vector("list", nrun)
for(i in 1:nrun)
{
    cat("Run ", i, "\n", sep = "")
    runs[[i]] = simulation_one_run(N = 200, p = 800, s1 = 20, s2 = 15, d = 2, lr = 0.1, save_evec = (i == 1))
}
save(runs, compress = "xz", file = "result/online_p800.RData")

# Simulation with p=1600
set.seed(123)
nrun = 10
runs = vector("list", nrun)
for(i in 1:nrun)
{
    cat("Run ", i, "\n", sep = "")
    runs[[i]] = simulation_one_run(N = 200, p = 1600, s1 = 20, s2 = 15, d = 2, lr = 0.1, save_evec = (i == 1))
}
save(runs, compress = "xz", file = "result/online_p1600.RData")



# Summarize experiment results
summarize_err = function(runs)
{
    methods = c("Oja's", "Incremental PCA", "Candid", "Proposed")
    nsim = length(runs)
    N = length(runs[[1]]$err_gradfps)
    gdat = vector("list", nsim)
    for(i in 1:nsim)
    {
        gdat[[i]] = tibble(
            iter   = rep(1:N, 4),
            err    = c(runs[[i]]$err_oja,
                       runs[[i]]$err_ipca,
                       runs[[i]]$err_ccipca,
                       runs[[i]]$err_gradfps),
            type   = rep(methods, times = rep(N, 4)),
            dataid = i
        )
    }
    gdat = do.call(rbind, gdat)
    gdat$type = factor(gdat$type, levels = methods)
    gdat
}

load("result/online_p800.RData")
gdat800 = summarize_err(runs)
gdat800$dim = "p = 800"

load("result/online_p1600.RData")
gdat1600 = summarize_err(runs)
gdat1600$dim = "p = 1600"

gdat = rbind(gdat800, gdat1600)
gdat$dim = factor(gdat$dim, levels = c("p = 800", "p = 1600"))
# gdat_mean = gdat %>% group_by(iter, type) %>% summarize(err = mean(err))
ggplot(gdat, aes(x = iter, y = err)) +
    facet_grid(rows = vars(dim), cols = vars(type)) +
    geom_line(aes(color = type,
                  # linetype = type,
                  group = interaction(type, dataid)),
              size = 0.8, alpha = 0.6) +
    scale_color_hue("Method") +
    # scale_linetype_manual("Method", values = c("33", "solid")) +
    # guides(color = guide_legend(keyheight = 2, keywidth = 3,
    #                             override.aes = list(alpha = 1))) +
    guides(color = "none") +
    xlab("Iteration") + ylab("Estimation Error") +
    theme_bw(base_size = 22, base_family = "Lato") +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.title = element_text(face = "bold"),
          legend.title = element_text(face = "bold"),
          legend.text = element_text(size = 18))
# ggsave("online_err.pdf", width = 16, height = 8)



summarize_evec = function(run, size = 100)
{
    methods = c("Oja's", "Incremental PCA", "Candid", "Proposed", "Ground Truth")

    evec_oja = run$evec_oja[1:size, ]
    evec_ipca = run$evec_ipca[1:size, ]
    evec_ccipca = run$evec_ccipca[1:size, ]
    evec_gradfps = run$evec_gradfps[1:size, ]
    evec_truth = run$evec_truth[1:size, ]

    row_id = row(evec_oja)
    col_id = col(evec_oja)
    gdat = tibble(
        row_id = rep(as.integer(row_id), 5),
        col_id = rep(as.integer(col_id), 5),
        evec   = c(as.numeric(evec_oja),
                   as.numeric(evec_ipca),
                   as.numeric(evec_ccipca),
                   as.numeric(evec_gradfps),
                   as.numeric(evec_truth)),
        type   = rep(methods, times = rep(length(evec_oja), 5))
    )
    gdat$type = factor(gdat$type, levels = methods)
    gdat
}

load("result/online_p800.RData")
gdat800 = summarize_evec(runs[[1]])
gdat800$dim = "p = 800"

load("result/online_p1600.RData")
gdat1600 = summarize_evec(runs[[1]])
gdat1600$dim = "p = 1600"

gdat = rbind(gdat800, gdat1600)
gdat$dim = factor(gdat$dim, levels = c("p = 800", "p = 1600"))

# Compute palatte
ngrid = 1001
pal = c("#67000d", "#a50f15", "#cb181d", "#ef3b2c",
        "#fb6a4a", "#fc9272", "#fcbba1", "#fee0d2",
        "#ffffff",
        "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
        "#4292c6", "#2171b5", "#08519c", "#08306b")
col_pal = colorRampPalette(pal)(ngrid)
lo = min(gdat$evec)
hi = max(gdat$evec)
r = max(abs(c(lo, hi)))
col_val = seq(-r, r, length.out = ngrid)
lo_ind = findInterval(lo, col_val)
hi_ind = findInterval(hi, col_val)
colors = col_pal[lo_ind:hi_ind]

ggplot(gdat, aes(x = row_id, y = col_id, fill = evec)) +
    facet_grid(rows = vars(dim), cols = vars(type)) +
    geom_tile() +
    scale_x_continuous("Variable Index", breaks = c(20, 40, 60, 80), expand = c(0, 0)) +
    scale_y_reverse("Principal Component", breaks = c(1, 2), expand = c(0, 0)) +
    scale_fill_gradientn("Factor\nLoading", colors = colors) +
    guides(fill = guide_colorbar(barwidth = 2, barheight = 12)) +
    theme_bw(base_size = 22, base_family = "Lato") +
    theme(axis.title = element_text(face = "bold"),
          panel.grid = element_blank(),
          legend.title = element_text(face = "bold"),
          legend.text = element_text(size = 18))
# ggsave("online_evec.pdf", width = 16, height = 6)



summarize_proj = function(run, size = 200)
{
    methods = c("Oja's", "Incremental PCA", "Candid", "Proposed", "Ground Truth")

    proj_oja = run$proj_oja[1:size, 1:size]
    proj_ipca = run$proj_ipca[1:size, 1:size]
    proj_ccipca = run$proj_ccipca[1:size, 1:size]
    proj_gradfps = run$proj_gradfps[1:size, 1:size]
    proj_truth = run$proj_truth[1:size, 1:size]

    proj_oja = proj_oja / max(abs(range(proj_oja)))
    proj_ipca = proj_ipca / max(abs(range(proj_ipca)))
    proj_ccipca = proj_ccipca / max(abs(range(proj_ccipca)))
    proj_gradfps = proj_gradfps / max(abs(range(proj_gradfps)))
    proj_truth = proj_truth / max(abs(range(proj_truth)))

    row_id = row(proj_oja)
    col_id = col(proj_oja)
    gdat = tibble(
        row_id = rep(as.integer(row_id), 5),
        col_id = rep(as.integer(col_id), 5),
        proj   = c(as.numeric(proj_oja),
                   as.numeric(proj_ipca),
                   as.numeric(proj_ccipca),
                   as.numeric(proj_gradfps),
                   as.numeric(proj_truth)),
        type   = rep(methods, times = rep(size^2, 5))
    )
    gdat$type = factor(gdat$type, levels = methods)
    gdat
}

load("result/online_p800.RData")
gdat800 = summarize_proj(runs[[1]])
gdat800$dim = "p = 800"

load("result/online_p1600.RData")
gdat1600 = summarize_proj(runs[[1]])
gdat1600$dim = "p = 1600"

gdat = rbind(gdat800, gdat1600)
gdat$dim = factor(gdat$dim, levels = c("p = 800", "p = 1600"))

# Compute palatte
ngrid = 1001
pal = c("#67000d", "#a50f15", "#cb181d", "#ef3b2c",
        "#fb6a4a", "#fc9272", "#fcbba1", "#fee0d2",
        "#ffffff",
        "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
        "#4292c6", "#2171b5", "#08519c", "#08306b")
col_pal = colorRampPalette(pal)(ngrid)
lo = min(gdat$proj)
hi = max(gdat$proj)
r = max(abs(c(lo, hi)))
col_val = seq(-r, r, length.out = ngrid)
lo_ind = findInterval(lo, col_val)
hi_ind = findInterval(hi, col_val)
colors = col_pal[lo_ind:hi_ind]

ggplot(gdat, aes(x = col_id, y = row_id, fill = proj)) +
    facet_grid(rows = vars(dim), cols = vars(type)) +
    geom_tile() +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_reverse(expand = c(0, 0)) +
    scale_fill_gradientn("Coefficient", colors = colors) +
    guides(fill = "none") +
    coord_equal() +
    theme_bw(base_size = 22, base_family = "Lato") +
    theme(axis.ticks = element_blank(),
          axis.text = element_blank(),
          axis.title = element_blank(),
          panel.grid = element_blank())
# ggsave("online_proj.pdf", width = 16, height = 7)
