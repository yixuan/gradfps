library(readr)
library(dplyr)
library(Matrix)
library(RSpectra)
library(gradfps)  # devtools::install_github("yixuan/gradfps")

##### Reading Data #####

# Construct data set from raw data
dat = read_tsv("DLPFC.ensembl.KNOWN.ADJUSTED.VOOM_NORMALIZED.GE.WEIGHTED_RESIDUALS.tsv")
module_ctr = read_tsv("DLPFC.ensembl.coexpr.Control.clustering_modules.tsv")
module_scz = read_tsv("DLPFC.ensembl.coexpr.SCZ.clustering_modules.tsv")
covariates = read_tsv("DLPFC.ensembl.KNOWN_AND_SVA.ADJUST.SAMPLE_COVARIATES.tsv")

# Meta information for each subject: control or SCZ
meta = covariates %>%
    select("sample_id" = "DLPFC_RNA_isolation: Sample RNA ID", Dx)

# ID of subjects in control group and SCZ group
control = meta %>% filter(Dx == "Control")
scz = meta %>% filter(Dx == "SCZ")

# Sequencing data for subjects in control group and SCZ group
dat_ctr = dat %>% select(one_of(control$sample_id))
dat_scz = dat %>% select(one_of(scz$sample_id))

# Combine module information
module = module_ctr %>% select(gene = Ensembl, gene_name = GeneSymbol, module_ctr = Module) %>%
    inner_join(module_scz %>% select(gene = Gene, gene_name_scz = MAPPED_Gene, module_scz = Module),
               by = "gene")
# Make sure gene names are the same in control and SCZ groups
stopifnot(all(module$gene_name == module$gene_name_scz))
# Map gene ID to name and module
gene_id = dat %>% select(GeneFeature) %>%
    mutate(gene_id = 1:n()) %>%
    inner_join(module %>% select(gene, module_ctr, module_scz), by = c("GeneFeature" = "gene")) %>%
    mutate(module_ctr = factor(module_ctr, levels = sprintf("M%sc", 0:35))) %>%
    arrange(desc(module_ctr))

# Overview of modules and their sizes
gene_id %>% group_by(module_ctr) %>% summarize(n = n()) %>% as.data.frame()
gene_id %>% group_by(module_scz) %>% summarize(n = n()) %>% arrange(desc(n)) %>% as.data.frame()

# write_csv(gene_id, "result/gene_id.csv")


##### Control Group #####

# Compute correlation matrix
mat_ctr = t(as.matrix(dat_ctr))
cor_ctr = cor(mat_ctr)

### All genes ###

# Detect range of lambda
d = 5
maxnvar = 5000
lambda_ctr = lambda_range(cor_ctr, d, 2000, maxnvar)
lambda_ctr
lambda = 0.85  # Also compute for 0.86, 0.87, 0.88, 0.89, 0.90

# Extract active set
gene_sub = active_set(cor_ctr, d, lambda)
cor_sub = cor_sub_ctr = cor_ctr[gene_sub, gene_sub]

# Initial value
e = eigs_sym(cor_sub, d, which = "LA")
plot(e$values)
x0 = tcrossprod(e$vectors)

# Sparse PCA
res_grf = gradfps_prox(cor_sub, d, lambda, x0 = x0, lr = 0.005, maxiter = 200,
                       control = list(fan_maxinc = 50,
                                      eps_abs = 0, eps_rel = 0,
                                      verbose = 1))
save(res_grf, gene_sub, cor_sub, file = sprintf("result/ctr_%.2f.RData", lambda))

##### SCZ Group #####

# Compute correlation matrix
mat_scz = t(as.matrix(dat_scz))
cor_scz = cor(mat_scz)

### All genes ###

# Detect range of lambda
d = 5
maxnvar = 5000
lambda_scz = lambda_range(cor_scz, d, 2000, maxnvar)
lambda_scz
lambda = 0.85  # Also compute for 0.86, 0.87, 0.88, 0.89, 0.90

# Extract active set
gene_sub = active_set(cor_scz, d, lambda)
cor_sub = cor_sub_scz = cor_scz[gene_sub, gene_sub]
save(cor_ctr, cor_scz, cor_sub_ctr, cor_sub_scz, file = "result/cors.RData")

# Initial value
e = eigs_sym(cor_sub, d, which = "LA")
plot(e$values)
x0 = tcrossprod(e$vectors)

# Sparse PCA
res_grf = gradfps_prox(cor_sub, d, lambda, x0 = x0, lr = 0.005, maxiter = 200,
                       control = list(fan_maxinc = 50,
                                      eps_abs = 0, eps_rel = 0,
                                      verbose = 1))
save(res_grf, gene_sub, cor_sub, file = sprintf("result/scz_%.2f.RData", lambda))
