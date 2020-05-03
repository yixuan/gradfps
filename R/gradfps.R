##' The GradFPS algorithm for sparse principal component analysis
##'
##' The GradFPS algorithm for sparse PCA using proximal-proximal gradient method.
##' It solves the optimization problem
##' \deqn{\min\quad-tr(SX) + \lambda ||X||_{1,1},}{min  -tr(AX) + \lambda * ||X||_{1,1},}
##' \deqn{s.t.\quad X\in \mathcal{F}^d,}{s.t. X in F^d,}
##' where \eqn{\mathcal{F}^d}{F^d} is the Fantope.
##'
##' @param S       The sample covariance matrix.
##' @param d       The number of sparse principal components to seek.
##' @param lambda  Sparsity parameter.
##' @param x0      Initial value for the projection matrix. If \code{NULL}, a
##'                default value based on the regular PCA will be used.
##' @param maxiter Maximum number of iterations.
##' @param control Additional parameters to control the optimization process.
##'
##' @details The \code{control} argument is a list that can supply any of the
##' following parameters:
##'
##' \describe{
##' \item{\code{mu, r1, r2}}{Penalty parameters.}
##' \item{\code{eps_abs}}{Absolute precision parameter. Default is \code{1e-3}.}
##' \item{\code{eps_rel}}{Relative precision parameter. Default is \code{1e-3}.}
##' \item{\code{fan_maxinc}}{Maximum number of incremental eigenvalues to compute
##'                          in each iteration of the Fantope proximal operator.
##'                          Default is 100.}
##' \item{\code{fan_maxiter}}{Maximum number of iterations in the Fantope proximal operator.
##'                           Default is 10.}
##' \item{\code{verbose}}{Level of verbosity. Default is 1.}
##' }
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
    p = nrow(S)
    opts = list(
        mu      = 100,
        r1      = sqrt(d * (d + 1)),
        r2      = sqrt(p * (d + 1)),
        eps_abs = 1e-3,
        eps_rel = 1e-3,
        fan_maxiter = 10,
        fan_maxinc = 100,
        verbose = 0
    )
    opts[names(control)] = control

    gradfps_prox_(S, x0, d, lambda, lr, opts$mu, opts$r1, opts$r2,
                  maxiter, opts$fan_maxinc, opts$fan_maxiter,
                  opts$eps_abs, opts$eps_rel,
                  opts$verbose)

}

##' @rdname gradfps_prox
##'
##' @param Pi The true projection matrix.
gradfps_prox_benchmark = function(
    S, Pi, d, lambda, x0 = NULL, lr = 0.001, maxiter = 100L, control = list()
)
{
    # Initial value
    if(is.null(x0))
    {
        e = RSpectra::eigs_sym(S, d, which = "LA")
        x0 = tcrossprod(e$vectors)
    }

    # Default control parameters
    p = nrow(S)
    opts = list(
        mu      = 100,
        r1      = sqrt(d * (d + 1)),
        r2      = sqrt(p * (d + 1)),
        eps_abs = 1e-3,
        eps_rel = 1e-3,
        fan_maxiter = 10,
        fan_maxinc = 100,
        verbose = 0
    )
    opts[names(control)] = control

    gradfps_prox_benchmark_(S, Pi, x0, d, lambda, lr, opts$mu, opts$r1, opts$r2,
                            maxiter, opts$fan_maxinc, opts$fan_maxiter,
                            opts$eps_abs, opts$eps_rel,
                            opts$verbose)
}

##' @rdname gradfps_prox
gradfps_subgrad = function(
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
    p = nrow(S)
    opts = list(
        mu      = 100,
        r1      = sqrt(d * (d + 1)),
        r2      = sqrt(p * (d + 1)),
        eps_abs = 1e-3,
        eps_rel = 1e-3,
        verbose = 0
    )
    opts[names(control)] = control

    gradfps_subgrad_(S, x0, d, lambda,
                     lr, opts$mu, opts$r1, opts$r2,
                     maxiter, opts$eps_abs, opts$eps_rel,
                     opts$verbose)
}

##' @rdname gradfps_prox
gradfps_subgrad_benchmark = function(
    S, Pi, d, lambda, x0 = NULL, lr = 0.001, maxiter = 100L, control = list()
)
{
    # Initial value
    if(is.null(x0))
    {
        e = RSpectra::eigs_sym(S, d, which = "LA")
        x0 = tcrossprod(e$vectors)
    }

    # Default control parameters
    p = nrow(S)
    opts = list(
        mu      = 100,
        r1      = sqrt(d * (d + 1)),
        r2      = sqrt(p * (d + 1)),
        eps_abs = 1e-3,
        eps_rel = 1e-3,
        verbose = 0
    )
    opts[names(control)] = control

    gradfps_subgrad_benchmark_(S, Pi, x0, d, lambda,
                               lr, opts$mu, opts$r1, opts$r2,
                               maxiter, opts$eps_abs, opts$eps_rel,
                               opts$verbose)
}



##' The GradFPS algorithm for sparse principal component analysis
##'
##' The GradFPS algorithm for sparse PCA using online mirror descent method.
##' It solves the optimization problem
##' \deqn{\min\quad-tr(SX) + \lambda ||X||_{1,1} + \frac{\delta}{2}||X||_{r,r}^2,}{min  -tr(AX) + \lambda * ||X||_{1,1} + (\delta / 2) * ||X||_{r,r}^2,}
##' \deqn{s.t.\quad X\in \mathcal{F}^d,}{s.t. X in F^d,}
##' where \eqn{r = 1 + 1 / (\log(p) - 1)}{r = 1 + 1 / (log(p) - 1)}, and
##' \eqn{\mathcal{F}^d}{F^d} is the Fantope.
##'
##' @param S       The sample covariance matrix.
##' @param d       The number of sparse principal components to seek.
##' @param lambda  Sparsity parameter.
##' @param delta   Penalty parameter.
##' @param x0      Initial value for the projection matrix. If \code{NULL}, a
##'                default value based on the regular PCA will be used.
##' @param maxiter Maximum number of iterations.
##' @param control Additional parameters to control the optimization process.
##'
##' @details The \code{control} argument is a list that can supply any of the
##' following parameters:
##'
##' \describe{
##' \item{\code{eps_abs}}{Absolute precision parameter. Default is \code{1e-3}.}
##' \item{\code{eps_rel}}{Relative precision parameter. Default is \code{1e-3}.}
##' \item{\code{fan_maxinc}}{Maximum number of incremental eigenvalues to compute
##'                          in each iteration of the Fantope proximal operator.
##'                          Default is 100.}
##' \item{\code{fan_maxiter}}{Maximum number of iterations in the Fantope proximal operator.
##'                           Default is 10.}
##' \item{\code{verbose}}{Level of verbosity. Default is 1.}
##' }
gradfps_prox_omd = function(
    S, d, lambda, delta, x0 = NULL, lr = 0.001, maxiter = 100L, control = list()
)
{
    # Initial value
    if(is.null(x0))
    {
        e = RSpectra::eigs_sym(S, d, which = "LA")
        x0 = tcrossprod(e$vectors)
    }

    # Default control parameters
    p = nrow(S)
    opts = list(
        mu      = 100,
        r1      = sqrt(d * (d + 1)),
        r2      = sqrt(p * (d + 1)),
        eps_abs = 1e-3,
        eps_rel = 1e-3,
        fan_maxiter = 10,
        fan_maxinc = 100,
        verbose = 0
    )
    opts[names(control)] = control

    gradfps_prox_omd_(S, x0, d, lambda, delta, lr, opts$mu, opts$r1, opts$r2,
                      maxiter, opts$fan_maxinc, opts$fan_maxiter,
                      opts$eps_abs, opts$eps_rel,
                      opts$verbose)
}
