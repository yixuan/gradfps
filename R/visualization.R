# Visualization of correlation matrix
view_cor = function(mat, bar_height = 10, font_size = 20)
{
    mat = as.matrix(mat)
    if(max(mat) > 1 || min(mat) < -1)
        stop("matrix values out of range [-1, 1]")

    gdat = data.frame(
        x = as.integer(col(mat)),
        y = as.integer(row(mat)),
        z = as.numeric(mat)
    )

    breaks = pretty(gdat$x, n = 10)
    breaks[breaks == 0] = 1
    breaks = unique(breaks)

    ngrid = 1001
    col_pal = colorRampPalette(c("#67001F", "#B2182B", "#D6604D", "#F4A582",
                                 "#FDDBC7", "#FFFFFF", "#D1E5F0",
                                 "#92C5DE", "#4393C3", "#2166AC", "#053061"))(ngrid)
    col_val = seq(-1, 1, length.out = ngrid)
    low = min(mat)
    low_ind = findInterval(low, col_val)
    colors = col_pal[low_ind:ngrid]

    ggplot(gdat, aes(x = x, y = y, fill = z)) +
        geom_raster() +
        scale_x_continuous("", breaks = breaks, expand = c(0, 0)) +
        scale_y_reverse("", breaks = breaks, expand = c(0, 0)) +
        scale_fill_gradientn("Correlation\nCoefficient", colors = colors) +
        guides(fill = guide_colourbar(barheight = bar_height)) +
        coord_fixed() +
        theme_bw(base_size = font_size) +
        theme(axis.title = element_blank())
}

view_evec = function(evecs, bar_height = 6, asp = 0.2, font_size = 20)
{
    v = as.matrix(round(evecs, 6))

    gdat = data.frame(
        x = as.integer(row(v)),
        y = as.integer(col(v)),
        z = as.numeric(v)
    )

    ggplot(gdat, aes(x = x, y = y, fill = z)) +
        geom_raster() +
        scale_x_continuous("Index of Variables", expand = c(0, 0)) +
        scale_y_reverse("Index of PCs", expand = c(0, 0)) +
        scale_fill_gradient2("Factor\nLoading") +
        guides(fill = guide_colourbar(barheight = bar_height)) +
        theme_bw(base_size = font_size) +
        theme(aspect.ratio = asp)
}
