suppressWarnings(suppressMessages(library(SingleCellExperiment)))
suppressWarnings(suppressMessages(library(scRNAseq)))
suppressWarnings(suppressMessages(library(scran)))
suppressWarnings(suppressMessages(library(igraph)))
suppressWarnings(suppressMessages(library(BiocSingular)))
# Read the filtered data
hvg <- read.csv("baron3_HVG_1200.csv", header=TRUE)
hvg$X <- NULL
hvg_matrix = as.matrix(hvg)
bsparam()
options(BiocSingularParam.default=ExactParam())
bsparam()
project_name <- "baron3"
num_hvg <- nrow(hvg_matrix)
k <- 5
g <- buildKNNGraph(hvg_matrix, k=k, d=10)
# Output the edge list to txt
write_graph(g, sprintf("%s_HVG_%d_KNN_k%d_d10.txt", project_name, num_hvg, k), format = c("edgelist"))