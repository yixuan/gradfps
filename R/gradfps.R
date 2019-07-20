gradfps_prox = function(
    S, d, lambda, x0 = NULL, lr = 0.001, maxiter = 100L, control = list()
)
{
    # Initial value
    if(is.null(x0))
    {
        e = RSpectra::eigs_sym(S, d, which = "LA")
        x0 = tcrossprod(e$vectors)
    }

    # Default control parameters
    opts = list(
        eps_abs = 1e-3,
        eps_rel = 1e-3,
        fan_maxiter = 10,
        fan_maxinc = 100,
        verbose = 1
    )
    opts[names(control)] = control

    gradfps_prox_(S, x0, d, lambda, lr, maxiter,
                  opts$fan_maxinc, opts$fan_maxiter,
                  opts$eps_abs, opts$eps_rel,
                  opts$verbose)
}
