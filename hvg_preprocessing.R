suppressWarnings(suppressMessages(library(dplyr)))
suppressWarnings(suppressMessages(library(Seurat)))
suppressWarnings(suppressMessages(library(textshape)))
project_name <- "sota"
# Use your own count matrix (might or might not have a header)
counts <- read.csv('data.csv', header=FALSE)
counts <- counts[ -c(1:3) ]
library(data.table)
counts <- transpose(counts)
# Use the column names as index (gene names)
counts <- column_to_rownames(counts, 'V1')
counts_seurat <- CreateSeuratObject(counts = counts, project = project_name)
# Normalizing the dta
counts_norm <- NormalizeData(counts_seurat, normalization.method = "LogNormalize", scale.factor = 10000)
num_hvg <- 1200 # Number of top genes
selected <- FindVariableFeatures(counts_norm, nfeatures = num_hvg)
topfeat = selected@assays[["RNA"]]@var.features
saveRDS(topfeat,"topfeat1200.rds")
# Output the dataframe to csv to be processed in knnn.r
hvg <- as.data.frame(GetAssayData(object = selected))[selected@assays$RNA@var.features, ]
write.csv(as.matrix(hvg), sprintf("%s_HVG_%d.csv", project_name, num_hvg))
