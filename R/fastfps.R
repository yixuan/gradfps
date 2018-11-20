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
    # Check the range of lambda
    lambda = lambda[1]
    nlambda = length(object$lambdas)
    lrange = range(object$lambdas)
    if(lambda < lrange[1] || lambda > lrange[2])
        stop(sprintf("lambda out of range [%f, %f]", lrange[1], lrange[2]))

    # Find two lambdas l1 >= l2 such that l1 >= lambda >= l2
    id = findInterval(lambda, sort(object$lambdas))
    l2_id = nlambda - id + 1
    l1_id = max(l2_id - 1, 1)
    l1 = object$lambdas[l1_id]
    l2 = object$lambdas[l2_id]

    # Interpolate between two projection matrices
    # Since l1 >= l2, proj2 will have a larger dimension, so we first enlarge proj1
    act_size = object$solution[[l2_id]]$act_size
    proj1 = object$solution[[l1_id]]$projection
    proj1 = sparseMatrix(i = proj1@i, p = proj1@p, x = proj1@x,
                         dims = c(act_size, act_size), index1 = FALSE)
    proj2 = object$solution[[l2_id]]$projection
    w = ifelse(l1 == l2, 1.0, (lambda - l2) / (l1 - l2))
    proj_act = w * proj1 + (1 - w) * proj2

    # Expand and reorder the projection matrix
    p = object$dim
    act_id = object$active[seq_len(act_size)]
    proj = reorder_projection(p, proj_act, act_id)

    # Eigenvectors
    evecs = matrix(0, p, object$rank)
    evecs_act = eigs_sym(proj_act, object$rank, which = "LA")$vectors
    evecs[act_id, ] = round(evecs_act, 6)

    list(projection = proj, eigenvectors = evecs)
}
