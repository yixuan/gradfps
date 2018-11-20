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
    class(res) = "fastfps"
    res
}

coef.fastfps = function(object, lambda, ...)
{
    lambda_id = which.min(abs(lambda - object$lambdas))
    sol = object$solution[[lambda_id]]

    p = object$dim
    act_id = object$active[seq_len(sol$act_size)]

    proj = Matrix(0, p, p)
    proj[act_id, act_id] = forceSymmetric(sol$projection, "L")

    evecs = matrix(0, p, object$rank)
    evecs_act = eigs_sym(sol$projection, object$rank, which = "LA")$vectors
    evecs[act_id, ] = round(evecs_act, 6)

    list(proj = proj, evecs = evecs)
}
