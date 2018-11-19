fastfps = function(
    S, d, lambda_min, lambda_max, nlambda,
    maxiter, eps_abs, eps_rel, opts
)
{
    p = nrow(S)

    # Default parameter values
    control_opts = list(lr = 0.01, mu = sqrt(p), r = min(10, p), verbose = TRUE)
    # Update parameters from 'opts' argument
    control_opts[names(opts)] = opts

    res = fastfps_internal(S, d, lambda_min, lambda_max, nlambda,
                           maxiter, eps_abs, eps_rel,
                           alpha0 = control_opts$lr, mu0 = control_opts$mu, r = control_opts$r,
                           verbose = control_opts$verbose)
    res
}
