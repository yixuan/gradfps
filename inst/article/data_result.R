# Code to reproduce Figure 6 and Figure 7

library(readr)
library(dplyr)
library(Matrix)
library(ggplot2)
library(RSpectra)
library(gradfps)
library(showtext)
font_add_google("Lato")
showtext_auto()

##### Select lambda #####
gene_selected = function(lambda, d, eps = 1e-3)
{
    load(sprintf("result/ctr_%.2f.RData", lambda))
    # plot(res_grf$err_v)
    proj = res_grf$projection
    proj[abs(proj) < eps] = 0
    e = eigs_sym(proj, d, which = "LA")
    ev = e$vectors
    row_max = apply(abs(ev), 1, max)
    ctr_selected = gene_sub[row_max >= eps]

    load(sprintf("result/scz_%.2f.RData", lambda))
    # plot(res_grf$err_v)
    proj = res_grf$projection
    proj[abs(proj) < eps] = 0
    e = eigs_sym(proj, d, which = "LA")
    ev = e$vectors
    row_max = apply(abs(ev), 1, max)
    scz_selected = gene_sub[row_max >= eps]

    res = list(
        ctr = ctr_selected,
        scz = scz_selected,
        common = intersect(ctr_selected, scz_selected),
        union = union(ctr_selected, scz_selected)
    )
    res$overlap = length(res$common) / length(res$union)
    res
}

d = 5
lambdas = c(0.85, 0.86, 0.87, 0.88, 0.89, 0.90)
for(l in lambdas)
{
    r = gene_selected(l, d, 1e-3)
    cat(sprintf("lambda = %f, #ctr = %d, #scz = %d, #common = %d, overlap = %f\n",
                l, length(r$ctr), length(r$scz), length(r$common), r$overlap))
}





##### Analysis #####

# Clustering
cor_clust = function(proj, ev, k, eps = 1e-3)
{
    n = nrow(ev)
    row_max = apply(abs(ev), 1, max)
    zeros = (row_max < eps)
    nzeros = sum(zeros)
    cat(sprintf("%s (%.2f%%) zero rows\n", nzeros, mean(zeros) * 100), sep = "")

    cat("Computing dissimilarity matrix...\n")
    zeros_ind = (1:n)[zeros]
    nonzeros_ind = (1:n)[!zeros]

    proj_sub = proj[nonzeros_ind, nonzeros_ind]
    dissim = max(proj_sub) + 0.001 - proj_sub

    cat("Clustering...\n")
    dist_mat = as.dist(dissim)
    cl = fastcluster::hclust(dist_mat, method = "complete")

    label = cutree(cl, k = k)
    cl_size = as.data.frame(table(label)) %>%
        mutate(label = as.integer(label))
    ord = data.frame(order = cl$order, label = label[cl$order]) %>%
        inner_join(cl_size, by = "label") %>%
        arrange(Freq)

    data.frame(
        order = c(nonzeros_ind[ord$order], zeros_ind),
        label = c(ord$label, rep(k + 1, nzeros)),
        size  = c(ord$Freq, rep(nzeros, nzeros))
    )
}

view_matrix_formatted = function(mat)
{
    view_matrix(mat, legend_title = "Sample\nCorrelation\nCoefficient") +
        guides(fill = guide_colorbar(barwidth = 1.8, barheight = 20)) +
        theme_bw(base_size = 22, base_family = "Lato") +
        theme(axis.title = element_blank(),
              legend.title = element_text(face = "bold"))
}

view_evec_formatted = function(mat)
{
    view_evec(mat, xlab = "Index of Reordered Genes", ylab = "Index of Components",
              legend_title = "Factor Loading    ", asp = 0.4) +
        guides(fill = guide_colorbar(barwidth = 30, barheight = 1.8)) +
        theme_bw(base_size = 26, base_family = "Lato") +
        theme(axis.title = element_text(face = "bold"),
              legend.position = "top",
              legend.title = element_text(face = "bold"))
}

### Control group ###

load("result/ctr_0.85.RData")
set.seed(123)
d = 5
lambda = 0.85
k = 5
proj = res_grf$projection
proj[abs(proj) < 1e-3] = 0
e = eigs_sym(proj, d, which = "LA")
evals = diag(t(e$vectors) %*% cor_sub %*% e$vectors)
evecs = e$vectors[, order(evals, decreasing = TRUE)]
cl = cor_clust(tcrossprod(evecs), evecs, k = k, eps = 1e-3)
ind = (cl$label <= k)
label_ctr = tibble(
    sub_id  = cl$order[ind],
    gene_id = gene_sub[cl$order[ind]],
    label   = cl$label[ind]
)
cor_sub_ctr = cor_sub

### SCZ group ###

load("result/scz_0.85.RData")
set.seed(123)
d = 5
lambda = 0.85
k = 5
plot(res_grf$err_v)
proj = res_grf$projection
proj[abs(proj) < 1e-3] = 0
e = eigs_sym(proj, d, which = "LA")
evals = diag(t(e$vectors) %*% cor_sub %*% e$vectors)
evecs = e$vectors[, order(evals, decreasing = TRUE)]
cl = cor_clust(tcrossprod(evecs), evecs, k = k, eps = 1e-3)
ord = cl$order
ord_sub = ord[1:200]
view_evec(evecs[ord_sub, ], bar_height = 6, asp = 0.2, font_size = 18)
cor_recover = cor_sub[ord_sub, ord_sub]
view_matrix(cor_recover)

ind = (cl$label <= k)
label_scz = tibble(
    sub_id  = cl$order[ind],
    gene_id = gene_sub[cl$order[ind]],
    label   = cl$label[ind]
)
cor_sub_scz = cor_sub

# Compare clusters
gene_id = read_csv("result/gene_id.csv")
# Size of WGCNA modules
gene_id %>% group_by(module_scz) %>% summarize(n = n()) %>% arrange(desc(n)) %>% as.data.frame()
# Cross table
cl_compare = label_scz %>% select(sub_id, gene_id, label) %>%
    inner_join(gene_id, by = "gene_id")
table(cl_compare$label, cl_compare$module_ctr)
table(cl_compare$label, cl_compare$module_scz)
# Two extra genes detected in Module 4
subid = (cl_compare %>% filter(module_scz == "tan"))$sub_id
c1 = data.frame(
    cor = cor_sub_scz[subid[1], label_scz$sub_id],
    sub_id = label_scz$sub_id,
    label = label_scz$label
)
print(c1)
c1 %>% filter(label == 4, cor < 1) %>% summarize(cor = mean(cor))
c2 = data.frame(
    cor = cor_sub_scz[subid[2], label_scz$sub_id],
    sub_id = label_scz$sub_id,
    label = label_scz$label
)
print(c2)
c2 %>% filter(label == 4, cor < 1) %>% summarize(cor = mean(cor))



view_evec_formatted(evecs[label_scz$sub_id, ])
# ggsave("evecs_scz_reorder.pdf", width = 12, height = 7)

view_matrix_formatted(cor_sub_scz[label_scz$sub_id, label_scz$sub_id])
# ggsave("cor_scz_reorder.pdf", width = 9, height = 7)





##### Comparison #####

### Compute sets ###
gene_common = intersect(label_ctr$gene_id, label_scz$gene_id)
label_common = tibble(gene_id = gene_common) %>%
    inner_join(label_ctr, by = "gene_id") %>%
    rename(sub_ctr = sub_id, label_ctr = label) %>%
    inner_join(label_scz, by = "gene_id") %>%
    rename(sub_scz = sub_id, label_scz = label)

gene_id = read_csv("result/gene_id.csv")
gene_ctr_unique = setdiff(label_ctr$gene_id, gene_common)
gene_scz_unique = setdiff(label_scz$gene_id, gene_common)

load("result/cors.RData")

# Common set
cor_com_ctr = cor_sub_ctr[label_common$sub_ctr, label_common$sub_ctr]
cor_com_scz = cor_sub_scz[label_common$sub_scz, label_common$sub_scz]
# Unique to control group
cor_ctru_ctr = cor_ctr[gene_ctr_unique, gene_ctr_unique]
cor_ctru_scz = cor_scz[gene_ctr_unique, gene_ctr_unique]
# Unique to SCZ group
cor_sczu_ctr = cor_ctr[gene_scz_unique, gene_scz_unique]
cor_sczu_scz = cor_scz[gene_scz_unique, gene_scz_unique]

# save(cor_com_ctr, cor_com_scz, cor_ctru_ctr, cor_ctru_scz, cor_sczu_ctr, cor_sczu_scz,
#      gene_common, gene_ctr_unique, gene_scz_unique,
#      file = "result/cor_comparison.RData")

### Plots ###

# load("result/cor_comparison.RData")
thm = theme(legend.margin = margin(), legend.box.margin = margin(r = -10))
view_cor_den = function(cor_ctr, cor_scz)
{
    cor_den = data.frame(
        cor = c(cor_ctr[lower.tri(cor_ctr)], cor_scz[lower.tri(cor_scz)]),
        group = rep(c("Control Group", "Schizophrenia Group"), each = sum(lower.tri(cor_ctr)))
    )
    ggplot(cor_den, aes(x = cor)) +
        geom_density(aes(group = group, color = group, linetype = group, size = group)) +
        geom_hline(yintercept = 0, size = 1.5, color = "grey") +
        scale_color_manual(values = c("#00BFC4", "#F8766D")) +
        scale_linetype_manual(values = c("31", "solid")) +
        scale_size_manual(values = c(1.2, 1.8)) +
        xlim(NA, 1) + xlab("Correlation Coefficient") + ylab("Density") +
        guides(color = guide_legend(title = NULL, keyheight = 1.5, keywidth = 3),
               linetype = guide_legend(title = NULL, keyheight = 1.5, keywidth = 3),
               size = guide_legend(title = NULL, keyheight = 1.5, keywidth = 3)) +
        theme_bw(base_size = 26, base_family = "Lato") +
        theme(legend.position = c(0.28, 0.9),
              legend.title = element_text(face = "bold"),
              legend.text = element_text(size = 24),
              legend.background = element_rect(fill = "#EEEEEE"),
              legend.margin = margin(t = -5, r = 12, b = 12, l = 12),
              axis.title = element_text(face = "bold"))
}

## Common set
view_matrix_formatted(cor_com_ctr) + thm
# ggsave("cor_ctr_common_gene.pdf", width = 9, height = 7)

view_matrix_formatted(cor_com_scz) + thm
# ggsave("cor_scz_common_gene.pdf", width = 9, height = 7)

view_cor_den(cor_com_ctr, cor_com_scz)
# ggsave("cor_den_common_gene.pdf", width = 9, height = 7)

## Unique to control group
view_matrix_formatted(cor_ctru_ctr) + thm
# ggsave("cor_ctr_ctr_unique.pdf", width = 9, height = 7)

view_matrix_formatted(cor_ctru_scz) + thm
# ggsave("cor_scz_ctr_unique.pdf", width = 9, height = 7)

view_cor_den(cor_ctru_ctr, cor_ctru_scz)
# ggsave("cor_den_ctr_unique.pdf", width = 9, height = 7)

## Unique to SCZ group
view_matrix_formatted(cor_sczu_ctr) + thm
# ggsave("cor_ctr_scz_unique.pdf", width = 9, height = 7)

view_matrix_formatted(cor_sczu_scz) + thm
# ggsave("cor_scz_scz_unique.pdf", width = 9, height = 7)

view_cor_den(cor_sczu_ctr, cor_sczu_scz)
# ggsave("cor_den_scz_unique.pdf", width = 9, height = 7)



gene_id %>% filter(gene_id %in% gene_common) %>% as.data.frame
gene_id %>% filter(gene_id %in% gene_ctr_unique) %>% as.data.frame
gene_id %>% filter(gene_id %in% gene_scz_unique) %>% as.data.frame

gene_id %>% filter(gene_id %in% gene_common) %>% group_by(module_ctr) %>% summarize(n = n())
gene_id %>% filter(gene_id %in% gene_ctr_unique) %>% group_by(module_ctr) %>% summarize(n = n())
gene_id %>% filter(gene_id %in% gene_scz_unique) %>% group_by(module_ctr) %>% summarize(n = n())

gene_id %>% filter(gene_id %in% gene_common) %>% group_by(module_scz) %>% summarize(n = n())
gene_id %>% filter(gene_id %in% gene_ctr_unique) %>% group_by(module_scz) %>% summarize(n = n())
gene_id %>% filter(gene_id %in% gene_scz_unique) %>% group_by(module_scz) %>% summarize(n = n())
