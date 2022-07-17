######### Online learning algorithms #########
library(RSpectra)
library(PRIMME)
library(R6)

# Oja's algorithm (Oja and Karhunen, 1985)
Oja = R6Class("Oja",
    private = list(
        result = NULL,
        lr = NULL
    ),

    public = list(
        initialize = function(dat_init, lr)
        {
            private$result = dat_init$evecs
            private$lr = lr
        },

        step = function(iter, zt, S1t)
        {
            alpha = private$lr / sqrt(iter)
            ngrad = crossprod(zt, zt %*% private$result)
            res = private$result + alpha * ngrad
            res = qr.Q(qr(res))
            private$result = res
            tcrossprod(res)
        },

        eigenvectors = function()
        {
            private$result
        }
    )
)

# Incremental PCA (Arora et al., 2012)
IPCA = R6Class("IPCA",
    private = list(
        d = NULL,
        evals = NULL,
        result = NULL
    ),

    public = list(
        initialize = function(dat_init)
        {
            private$d = length(dat_init$evals)
            private$evals = dat_init$evals
            private$result = dat_init$evecs
        },

        step = function(iter, zt, S1t)
        {
            d = private$d
            U = private$result

            cvec = zt %*% U
            zt = zt - tcrossprod(cvec, U)
            znorm = sqrt(sum(zt^2))
            Q11 = (iter + 1) * diag(private$evals) + crossprod(cvec)
            Q21 = cvec * znorm
            Q22 = znorm^2
            Q = iter / (iter + 1)^2 * cbind(rbind(Q11, Q21), rbind(t(Q21), Q22))

            e = eigen(Q, symmetric = TRUE)
            private$evals = e$values[1:d]
            res = cbind(U, t(zt) / znorm) %*% e$vectors
            private$result = res[, 1:d]
            tcrossprod(private$result)
        },

        eigenvectors = function()
        {
            private$result
        }
    )
)

# CCIPCA (Weng et al., 2003)
CCIPCA = R6Class("CCIPCA",
    private = list(
        d = NULL,
        V = NULL,
        result = NULL
    ),

    public = list(
        initialize = function(dat_init)
        {
            private$d = length(dat_init$evals)
            private$V = sweep(dat_init$evecs, 2, dat_init$evals, "*")
            private$result = dat_init$evecs
        },

        step = function(iter, zt, S1t)
        {
            d = private$d
            V = private$V
            U = private$result
            inc = V
            # Compute the increment
            for(j in 1:d)
            {
                # Modify the input data
                if(j >= 2)
                {
                    u = U[, 1:(j - 1), drop = FALSE]
                    zt = zt - tcrossprod(zt %*% u, u)
                }
                inc[, j] = crossprod(zt, zt %*% V[, j]) / sqrt(sum(V[, j]^2))
            }
            # Update V
            V = (iter - 1) * V / iter + inc / iter
            # Orthonormalization
            U = qr.Q(qr(V))

            private$V = V
            private$result = U
            tcrossprod(U)
        },

        eigenvectors = function()
        {
            private$result
        }
    )
)

# Online GradFPS
grad_eig = function(v, gamma1, gamma2)
{
    emax = RSpectra::eigs_sym(v, 1, which = "LA")
    emin = PRIMME::eigs_sym(v, 1, which = "SA")
    l1 = emax$values
    lp = emin$values
    v1 = emax$vectors
    vp = emin$vectors
    l1_new = max(l1 - gamma1, min(l1, 1))
    lp_new = min(lp + gamma2, max(lp, 0))
    v + (l1_new - l1) * tcrossprod(v1) + (lp_new - lp) * tcrossprod(vp)
}

prox_tr = function(v, alpha, d)
{
    p = nrow(v)
    tr = sum(diag(v))
    shift = (d - tr) / p
    b = alpha / abs(shift) / sqrt(p)
    v + min(b, 1) * shift * diag(p)
}

prox_l1 = function(v, beta) sign(v) * pmax(abs(v) - beta, 0)

GradFPS = R6Class("GradFPS",
    private = list(
        result = NULL,
        lr = NULL,
        d = NULL,
        lambda = NULL,
        mu = NULL,
        mur1 = NULL,
        mur2 = NULL
    ),

    public = list(
        initialize = function(dat_init, lambda, lr)
        {
            p = nrow(dat_init$evecs)
            d = length(dat_init$evals)
            private$result = dat_init$proj
            private$lr = lr
            private$d = d
            private$lambda = lambda
            nv = private$lambda * p + dat_init$norm + 1
            private$mu = nv * sqrt(p / (d + 1))
            private$mur1 = private$mu * sqrt(d * (d + 1))
            private$mur2 = private$mu * sqrt(p * (d + 1))
        },

        step = function(iter, zt, S1t)
        {
            alpha = private$lr / sqrt(iter)
            res = grad_eig(private$result, alpha * private$mur1, alpha * private$mur2)
            res = prox_tr(res, alpha * private$mu, private$d)
            res = res + alpha * S1t
            res = prox_l1(res, alpha * private$lambda)
            xnorm = norm(res, type = "F")
            if(xnorm > sqrt(private$d))
                res = res * (sqrt(private$d) / xnorm)
            private$result = res
            res
        },

        eigenvectors = function()
        {
            e = RSpectra::eigs_sym(private$result, private$d)
            e$vectors
        }
    )
)
