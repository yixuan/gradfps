# Code to reproduce Figure 2 and Figure 3

library(ggplot2)
library(dplyr)
library(showtext)
font_add_google("Lato")
showtext_auto()

d200 = read.csv("result/n_50_p_200.csv")
d400 = read.csv("result/n_100_p_400.csv")
d800 = read.csv("result/n_200_p_800.csv")
d1600 = read.csv("result/n_400_p_1600.csv")
d3200 = read.csv("result/n_800_p_3200.csv")
d6400 = read.csv("result/n_1600_p_6400.csv")

gdat = rbind(d200, d400, d800, d1600, d3200, d6400)
gdat$par = factor(gdat$par, levels = c("n = 50, p = 200",
                                       "n = 100, p = 400",
                                       "n = 200, p = 800",
                                       "n = 400, p = 1600",
                                       "n = 800, p = 3200",
                                       "n = 1600, p = 6400"))
gdat = gdat %>% group_by(method, par) %>% mutate(iter = rank(time))

# Show both time and iteration
ggplot(gdat, aes(x = time, y = err)) +
    geom_line(aes(group = method, color = method, linetype = method), size = 0.5) +
    # geom_point(aes(color = method, shape = method), size = 2) +
    geom_point(aes(color = method), shape = 3, size = 1) +
    scale_linetype_manual("Method", values = c("32", "solid"),
                          labels = c("Existing", "Proposed")) +
    # scale_shape_manual("Method", values = c(17, 16)) +
    scale_color_hue("Method", labels = c("Existing", "Proposed")) +
    guides(color = guide_legend(keyheight = 2, keywidth = 3),
           linetype = guide_legend(keyheight = 2, keywidth = 3)) +
    xlab("Elapsed Time (s)") + ylab("Estimation Error") +
    facet_wrap(~par, nrow = 2, scales = "free_x") +
    theme_bw(base_size = 20, base_family = "Lato") +
    theme(plot.title = element_text(hjust = 0.5),
          axis.title = element_text(face = "bold"),
          legend.title = element_text(face = "bold"))

# Show time only
ggplot(gdat, aes(x = time, y = err)) +
    geom_line(aes(group = method, color = method, linetype = method), size = 1.2) +
    scale_linetype_manual("Method", values = c("32", "solid"),
                          labels = c("Existing", "Proposed")) +
    scale_color_hue("Method", labels = c("Existing", "Proposed")) +
    guides(color = guide_legend(keyheight = 2, keywidth = 3),
           linetype = guide_legend(keyheight = 2, keywidth = 3)) +
    xlab("Elapsed Time (s)") + ylab("Estimation Error") +
    facet_wrap(~par, nrow = 2, scales = "free_x") +
    theme_bw(base_size = 20, base_family = "Lato") +
    theme(plot.title = element_text(hjust = 0.5),
          axis.title = element_text(face = "bold"),
          legend.title = element_text(face = "bold"))
# ggsave("comp_efficiency_time.pdf", width = 15, height = 7)

# Show iteration only
ggplot(gdat, aes(x = iter, y = err)) +
    geom_line(aes(group = method, color = method, linetype = method), size = 1.2) +
    scale_linetype_manual("Method", values = c("32", "solid"),
                          labels = c("Existing", "Proposed")) +
    scale_color_hue("Method", labels = c("Existing", "Proposed")) +
    guides(color = guide_legend(keyheight = 2, keywidth = 3),
           linetype = guide_legend(keyheight = 2, keywidth = 3)) +
    xlab("Iteration") + ylab("Estimation Error") +
    facet_wrap(~par, nrow = 2, scales = "free_x") +
    theme_bw(base_size = 20, base_family = "Lato") +
    theme(plot.title = element_text(hjust = 0.5),
          axis.title = element_text(face = "bold"),
          legend.title = element_text(face = "bold"))
# ggsave("comp_efficiency_iter.pdf", width = 15, height = 7)
