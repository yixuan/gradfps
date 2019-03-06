# Visualization of a matrix by coloring its coefficients
view_matrix = function(mat, legend_title = "Coefficient", bar_height = 10, font_size = 20)
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
    breaks = pretty(gdat$x, n = 10)
    # Start from 1, not 0
    breaks[breaks == 0] = 1
    breaks = unique(breaks)

    # Map the color spectrum to [-r, r]
    ngrid = 1001
    col_pal = colorRampPalette(c("#67001F", "#B2182B", "#D6604D", "#F4A582",
                                 "#FDDBC7", "#FFFFFF", "#D1E5F0",
                                 "#92C5DE", "#4393C3", "#2166AC", "#053061"))(ngrid)
    col_val = seq(-r, r, length.out = ngrid)
    lo_ind = findInterval(lo, col_val)
    hi_ind = findInterval(hi, col_val)
    colors = col_pal[lo_ind:hi_ind]

    ggplot(gdat, aes(x = x, y = y, fill = z)) +
        geom_raster() +
        scale_x_continuous("", breaks = breaks, expand = c(0, 0)) +
        scale_y_reverse("", breaks = breaks, expand = c(0, 0)) +
        scale_fill_gradientn("Correlation\nCoefficient", colors = colors) +
        guides(fill = guide_colorbar(barheight = bar_height)) +
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
        scale_fill_gradient2("Factor\nLoading", low = "#CD0000", high = "#0000CD") +
        guides(fill = guide_colourbar(barheight = bar_height)) +
        theme_bw(base_size = font_size) +
        theme(aspect.ratio = asp)
}
