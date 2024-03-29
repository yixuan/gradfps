##' Visualization of Matrices
##'
##' \code{view_matrix()} visualizes a general matrix by mapping its elements to
##' colors. \code{view_evec()} visualizes a set of eigenvectors.
##'
##' @param mat,evec     The matrix to be visualized.
##' @param xlab         X-axis label.
##' @param ylab         Y-axis label.
##' @param legend_title Title of the color bar legend.
##' @param asp          Aspect ratio of the plot.
##' @param bar_height   Height of the color bar.
##' @param font_size    Base font size for the plot.
##' @param padding      The padding between the axis border and the plot area.
##' @param pal          A character vector to specify the the palette of the color bar.
##'
##' @rdname visualization
##' @author Yixuan Qiu \url{https://statr.me}
##'
##' @examples
##' set.seed(123)
##' x = matrix(rnorm(200), 10, 20)
##' view_matrix(x)
##'
##' sigma1 = matrix(0.8, 20, 20) + 0.2 * diag(20)
##' sigma2 = matrix(0.6, 50, 50) + 0.4 * diag(50)
##' s1 = stats::rWishart(1, 100, sigma1)[, , 1]
##' s2 = stats::rWishart(1, 100, sigma2)[, , 1]
##' s = as.matrix(Matrix::bdiag(s1, s2))
##' view_matrix(s)
##'
##' v = eigen(s, symmetric = TRUE)$vectors[, 1:5]
##' view_evec(v)

# Visualization of a matrix by coloring its coefficients
view_matrix = function(mat, xlab = "", ylab = "", legend_title = "Coefficient",
                       bar_height = 10, font_size = 20, padding = 0, pal = NULL)
{
    mat = as.matrix(mat)
    lo = min(mat)
    hi = max(mat)
    # All data in the range [-r, r]
    r = max(abs(c(lo, hi)))

    # ggplot2 format
    gdat = data.frame(
        x = as.integer(col(mat)),
        y = as.integer(row(mat)),
        z = as.numeric(mat)
    )

    # Axis breaks
    breaks_x = pretty(gdat$x, n = 10)
    # Start from 1, not 0
    breaks_x[breaks_x == 0] = 1
    breaks_x = unique(breaks_x)

    breaks_y = pretty(gdat$y, n = 10)
    # Start from 1, not 0
    breaks_y[breaks_y == 0] = 1
    breaks_y = unique(breaks_y)

    # Map the color spectrum to [-r, r]
    ngrid = 1001
    # pal = c("#67001F", "#B2182B", "#D6604D", "#F4A582",
    #         "#FDDBC7", "#FFFFFF", "#D1E5F0",
    #         "#92C5DE", "#4393C3", "#2166AC", "#053061")
    if(is.null(pal))
        pal = c("#67000d", "#a50f15", "#cb181d", "#ef3b2c",
                "#fb6a4a", "#fc9272", "#fcbba1", "#fee0d2",
                "#ffffff",
                "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
                "#4292c6", "#2171b5", "#08519c", "#08306b")
    col_pal = colorRampPalette(pal)(ngrid)

    col_val = seq(-r, r, length.out = ngrid)
    lo_ind = findInterval(lo, col_val)
    hi_ind = findInterval(hi, col_val)
    colors = col_pal[lo_ind:hi_ind]

    ggplot(gdat, aes(x = x, y = y, fill = z)) +
        geom_tile() +
        scale_x_continuous(xlab, limits = c(0.5, max(gdat$x) + 0.5),
                           breaks = breaks_x, expand = c(padding, padding)) +
        scale_y_reverse(ylab, limits = c(max(gdat$y) + 0.5, 0.5),
                        breaks = breaks_y, expand = c(padding, padding)) +
        scale_fill_gradientn(legend_title, colors = colors) +
        guides(fill = guide_colorbar(barheight = bar_height)) +
        coord_fixed() +
        theme_bw(base_size = font_size) +
        theme(axis.title.x = if(xlab == "") element_blank() else element_text(),
              axis.title.y = if(ylab == "") element_blank() else element_text(),
              panel.grid = element_blank())
}

##' @rdname visualization

# Visualization of eigenvectors
view_evec = function(
    evecs,
    xlab = "Index of Variables", ylab = "Index of PCs", legend_title = "Factor\nLoading",
    asp = 0.2, bar_height = 6, font_size = 20, pal = NULL
)
{
    v = as.matrix(evecs)
    lo = min(v)
    hi = max(v)
    # All values in the range [-r, r]
    r = max(abs(c(lo, hi)))

    # ggplot2 format
    gdat = data.frame(
        x = as.integer(row(v)),
        y = as.integer(col(v)),
        z = as.numeric(v)
    )

    # Map the color spectrum to [-r, r]
    ngrid = 1001
    # pal = c("#67001F", "#B2182B", "#D6604D", "#F4A582",
    #         "#FDDBC7", "#FFFFFF", "#D1E5F0",
    #         "#92C5DE", "#4393C3", "#2166AC", "#053061")
    if(is.null(pal))
        pal = c("#67000d", "#a50f15", "#cb181d", "#ef3b2c",
                "#fb6a4a", "#fc9272", "#fcbba1", "#fee0d2",
                "#ffffff",
                "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
                "#4292c6", "#2171b5", "#08519c", "#08306b")
    col_pal = colorRampPalette(pal)(ngrid)

    col_val = seq(-r, r, length.out = ngrid)
    lo_ind = findInterval(lo, col_val)
    hi_ind = findInterval(hi, col_val)
    colors = col_pal[lo_ind:hi_ind]

    ggplot(gdat, aes(x = x, y = y, fill = z)) +
        geom_tile() +
        scale_x_continuous(xlab, expand = c(0, 0)) +
        scale_y_reverse(ylab, breaks = 1:ncol(v), expand = c(0, 0)) +
        scale_fill_gradientn(legend_title, colors = colors) +
        guides(fill = guide_colorbar(barheight = bar_height)) +
        theme_bw(base_size = font_size) +
        theme(aspect.ratio = asp)
}
