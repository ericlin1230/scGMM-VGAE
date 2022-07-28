rm(list = ls())
if(!is.null(dev.list())) dev.off()
cat("\f")

########### Simulation using Darmanis as the template
# Load the raw data
Darmanis<-read.csv("/Users/lbyjo/Desktop/Pingzhao Hu/Rawdata/Darmanis.csv")
D_label<-read.csv("/Users/lbyjo/Desktop/Pingzhao Hu/Rawdata/truth.csv")
rownames(Darmanis)<-Darmanis[,1]
Darmanis<-Darmanis[,-1]

# Filter 1200 genes
library(Seurat)
D_seurat <- CreateSeuratObject(counts = Darmanis)
D_lognorm <- NormalizeData(D_seurat, normalization.method = "LogNormalize", scale.factor = 10000)
D.selected <- FindVariableFeatures(D_lognorm, nfeatures = 1200)
top1200_D = D.selected@assays[["RNA"]]@var.features

# Estimate the parameters from Darmanis dataset by SPARSim
library(SPARSim)
Darmanis_norm<-scran_normalization(Darmanis)
cond_index_1_fq<-D_label$X[D_label$truth=='fetal_quiescent']
cond_index_2_fr<-D_label$X[D_label$truth=='fetal_replicating']
cond_index_3_a<-D_label$X[D_label$truth=='astrocytes']
cond_index_4_n<-D_label$X[D_label$truth=='neurons']
cond_index_5_e<-D_label$X[D_label$truth=='endothelial']
cond_index_6_o<-D_label$X[D_label$truth=='oligodendrocytes']
cond_index_7_m<-D_label$X[D_label$truth=='microglia']
cond_index_8_OPC<-D_label$X[D_label$truth=='OPC']

Example_count_matrix_conditions <- list(cond_A = cond_index_1_fq, 
                                        cond_B = cond_index_2_fr, 
                                        cond_C = cond_index_3_a,
                                        cond_D = cond_index_4_n,
                                        cond_E = cond_index_5_e,
                                        cond_F = cond_index_6_o,
                                        cond_G = cond_index_7_m,
                                        cond_H = cond_index_8_OPC)
Darmanis_sim_param <- SPARSim_estimate_parameter_from_data(raw_data = Darmanis, 
                                                          norm_data = Darmanis_norm, 
                                                          conditions = Example_count_matrix_conditions)
# Save the estimated parameters
save(Darmanis_sim_param, file = "/Users/lbyjo/Desktop/Pingzhao Hu/Simulation/DSim_v2/Darmanis_simulation.RData")


# Load the Darmanis parameters for each cluster
Darmanis_C1<-Darmanis_sim_param[[1]]
Darmanis_C2<-Darmanis_sim_param[[2]]
Darmanis_C3<-Darmanis_sim_param[[3]]
Darmanis_C4<-Darmanis_sim_param[[4]]
Darmanis_C5<-Darmanis_sim_param[[5]]
Darmanis_C6<-Darmanis_sim_param[[6]]
Darmanis_C7<-Darmanis_sim_param[[7]]
Darmanis_C8<-Darmanis_sim_param[[8]]

# Change the parameters to only simulate the top1200 genes
cond_A_param <- SPARSim_create_simulation_parameter(
  intensity = Darmanis_C1$intensity[top1200_D], 
  variability = Darmanis_C1$variability[top1200_D], 
  library_size = Darmanis_C1$lib_size, 
  condition_name = "cond_A")

cond_B_param <- SPARSim_create_simulation_parameter(
  intensity = Darmanis_C2$intensity[top1200_D], 
  variability = Darmanis_C2$variability[top1200_D], 
  library_size = Darmanis_C2$lib_size, 
  condition_name = "cond_B")

cond_C_param <- SPARSim_create_simulation_parameter(
  intensity = Darmanis_C3$intensity[top1200_D], 
  variability = Darmanis_C3$variability[top1200_D], 
  library_size = Darmanis_C3$lib_size, 
  condition_name = "cond_C")

cond_D_param <- SPARSim_create_simulation_parameter(
  intensity = Darmanis_C4$intensity[top1200_D], 
  variability = Darmanis_C4$variability[top1200_D], 
  library_size = Darmanis_C4$lib_size, 
  condition_name = "cond_D")

cond_E_param <- SPARSim_create_simulation_parameter(
  intensity = Darmanis_C5$intensity[top1200_D], 
  variability = Darmanis_C5$variability[top1200_D], 
  library_size = Darmanis_C5$lib_size, 
  condition_name = "cond_E")

cond_F_param <- SPARSim_create_simulation_parameter(
  intensity = Darmanis_C6$intensity[top1200_D], 
  variability = Darmanis_C6$variability[top1200_D], 
  library_size = Darmanis_C6$lib_size, 
  condition_name = "cond_F")

cond_G_param <- SPARSim_create_simulation_parameter(
  intensity = Darmanis_C7$intensity[top1200_D], 
  variability = Darmanis_C7$variability[top1200_D], 
  library_size = Darmanis_C7$lib_size, 
  condition_name = "cond_G")

cond_H_param <- SPARSim_create_simulation_parameter(
  intensity = Darmanis_C8$intensity[top1200_D], 
  variability = Darmanis_C8$variability[top1200_D], 
  library_size = Darmanis_C8$lib_size, 
  condition_name = "cond_H")

SPARSim_sim_param <- list(cond_A = cond_A_param, cond_B = cond_B_param, cond_C = cond_C_param, cond_D = cond_D_param,
                          cond_E = cond_E_param, cond_F = cond_F_param, cond_G = cond_G_param, cond_H = cond_H_param)

# Load the cell graph generation package
library(SingleCellExperiment)
library(scRNAseq)
library(scran)
library(igraph)
library(BiocSingular)
bsparam()
options(BiocSingularParam.default=ExactParam())
bsparam()

# Simulation
for (i in 1:10){
  set.seed(i)
  # Simulate dataset
  sim_result <- SPARSim_simulation(dataset_parameter = SPARSim_sim_param)
  D_sim<-data.frame(sim_result$count_matrix)
  write.csv(D_sim,paste0("/Users/lbyjo/Desktop/Pingzhao Hu/Simulation/DSim/DSim_",i,".csv"), row.names = TRUE)
  # Generate cell graph
  D.graph <- buildKNNGraph(D_sim, k=5, d=10)
  write_graph(D.graph, paste0("/Users/lbyjo/Desktop/Pingzhao Hu/Simulation/DSim/DSim_graph_",i,".txt"), format = c("edgelist"))
}

# Create Labels
cluster_D<-c(110,25,62,131,20,38,16,18)
DSim_label<-D_sim[1,]
for (i in 1:8){
  DSim_label[1,(sum(cluster_D[0:(i-1)])+1):sum(cluster_D[0:i])]<-i
}
row.names(DSim_label)<-'assigned_cluster' 
DSim_label<-data.frame(t(DSim_label))
write.csv(DSim_label,"/Users/lbyjo/Desktop/Pingzhao Hu/Simulation/DSim/DSim_label.csv", row.names = TRUE)






########### Simulation using Zheng preset
# Load raw dataset
library(Seurat)
raw=ReadMtx(mtx='/Users/lbyjo/Desktop/Pingzhao Hu/Rawdata/filtered_matrices_mex/hg19/matrix.mtx', 
            cells = '/Users/lbyjo/Desktop/Pingzhao Hu/Rawdata/filtered_matrices_mex/hg19/barcodes.tsv',
            features = '/Users/lbyjo/Desktop/Pingzhao Hu/Rawdata/filtered_matrices_mex/hg19/genes.tsv')
raw=data.frame(raw)

# The raw data contains 32738 genes, while the template for simulation contains 19536 genes
# So select those 19536 genes from the raw data.
library(SPARSim)
data(Zheng_param_preset)
selectgenes<-attributes(Zheng_param_preset$Zheng_C1$intensity)$names
raw_filter<-raw[selectgenes,]
raw_filter[is.na(raw_filter)]<-0

# Filter 1200 genes
Z_seurat <- CreateSeuratObject(counts = raw_filter)
Z_lognorm <- NormalizeData(Z_seurat, normalization.method = "LogNormalize", scale.factor = 10000)
Z.selected <- FindVariableFeatures(Z_lognorm, nfeatures = 1200)
top1200 = Z.selected@assays[["RNA"]]@var.features

# Load the parameter presets for each cluster
Zheng_C1<-Zheng_param_preset$Zheng_C1
Zheng_C2<-Zheng_param_preset$Zheng_C2
Zheng_C3<-Zheng_param_preset$Zheng_C3
Zheng_C4<-Zheng_param_preset$Zheng_C4

# Load the cell graph generation package
library(SingleCellExperiment)
library(scRNAseq)
library(scran)
library(igraph)
library(BiocSingular)
bsparam()
options(BiocSingularParam.default=ExactParam())
bsparam()


#### Simulate same-size dataset (3388 cells)
# Change the parameters to only simulate the top1200 genes
cond_A_param <- SPARSim_create_simulation_parameter(
  intensity = Zheng_C1$intensity[top1200], 
  variability = Zheng_C1$variability[top1200], 
  library_size = Zheng_C1$lib_size, 
  condition_name = "cond_A")

cond_B_param <- SPARSim_create_simulation_parameter(
  intensity = Zheng_C2$intensity[top1200], 
  variability = Zheng_C2$variability[top1200], 
  library_size = Zheng_C2$lib_size, 
  condition_name = "cond_B")

cond_C_param <- SPARSim_create_simulation_parameter(
  intensity = Zheng_C3$intensity[top1200], 
  variability = Zheng_C3$variability[top1200], 
  library_size = Zheng_C3$lib_size, 
  condition_name = "cond_C")

cond_D_param <- SPARSim_create_simulation_parameter(
  intensity = Zheng_C4$intensity[top1200], 
  variability = Zheng_C4$variability[top1200], 
  library_size = Zheng_C4$lib_size, 
  condition_name = "cond_D")

SPARSim_sim_param <- list(cond_A = cond_A_param, cond_B = cond_B_param, cond_C = cond_C_param, cond_D = cond_D_param)

for (i in 1:10){
  set.seed(i)
  # Simulate dataset
  sim_result <- SPARSim_simulation(dataset_parameter = SPARSim_sim_param)
  Zheng_same<-data.frame(sim_result$count_matrix)
  write.csv(Zheng_same,paste0("/Users/lbyjo/Desktop/Pingzhao Hu/Simulation/test/Z_same_",i,".csv"), row.names = TRUE)
  # Generate cell graph
  Z.graph <- buildKNNGraph(Zheng_same, k=5, d=10)
  write_graph(Z.graph, paste0("/Users/lbyjo/Desktop/Pingzhao Hu/Simulation/test/Z_same_graph_",i,".txt"), format = c("edgelist"))
}

# Create Labels
cluster_same<-c(1440,1718,184,46)
Z_same_label<-Zheng_same[1,]
for (i in 1:4){
  Z_same_label[1,(sum(cluster_same[0:(i-1)])+1):sum(cluster_same[0:i])]<-i
}
row.names(Z_same_label)<-'assigned_cluster' 
Z_same_label<-data.frame(t(Z_same_label))
write.csv(Z_same_label,"/Users/lbyjo/Desktop/Pingzhao Hu/Simulation/test/Z_same_label.csv", row.names = TRUE)



##### Simulate half-size dataset (1694 cells)
# Change the parameters to only simulate the top1200 genes
set.seed(123)
cond_A_param <- SPARSim_create_simulation_parameter(
  intensity = Zheng_C1$intensity[top1200], 
  variability = Zheng_C1$variability[top1200], 
  library_size = sample(Zheng_C1$lib_size, size =  720, replace = FALSE), 
  condition_name = "cond_A")

cond_B_param <- SPARSim_create_simulation_parameter(
  intensity = Zheng_C2$intensity[top1200], 
  variability = Zheng_C2$variability[top1200], 
  library_size = sample(Zheng_C2$lib_size, size =  859, replace = FALSE), 
  condition_name = "cond_B")

cond_C_param <- SPARSim_create_simulation_parameter(
  intensity = Zheng_C3$intensity[top1200], 
  variability = Zheng_C3$variability[top1200], 
  library_size = sample(Zheng_C3$lib_size, size =  92, replace = FALSE), 
  condition_name = "cond_C")

cond_D_param <- SPARSim_create_simulation_parameter(
  intensity = Zheng_C4$intensity[top1200], 
  variability = Zheng_C4$variability[top1200], 
  library_size = sample(Zheng_C4$lib_size, size =  23, replace = FALSE), 
  condition_name = "cond_D")

SPARSim_sim_param <- list(cond_A = cond_A_param, cond_B = cond_B_param, cond_C = cond_C_param, cond_D = cond_D_param)

for (i in 1:10){
  set.seed(i)
  # Simulate dataset
  sim_result <- SPARSim_simulation(dataset_parameter = SPARSim_sim_param)
  Zheng_half<-data.frame(sim_result$count_matrix)
  write.csv(Zheng_half,paste0("/Users/lbyjo/Desktop/Pingzhao Hu/Simulation/test/Z_half_",i,".csv"), row.names = TRUE)
  # Generate cell graph
  Z.graph <- buildKNNGraph(Zheng_half, k=5, d=10)
  write_graph(Z.graph, paste0("/Users/lbyjo/Desktop/Pingzhao Hu/Simulation/test/Z_half_graph_",i,".txt"), format = c("edgelist"))
}

# Create Labels
cluster_half<-c(720, 859,92,23)
Z_half_label<-Zheng_half[1,]
for (i in 1:4){
  Z_half_label[1,(sum(cluster_half[0:(i-1)])+1):sum(cluster_half[0:i])]<-i
}
row.names(Z_half_label)<-'assigned_cluster' 
Z_half_label<-data.frame(t(Z_half_label))
write.csv(Z_half_label,"/Users/lbyjo/Desktop/Pingzhao Hu/Simulation/test/Z_half_label.csv", row.names = TRUE)



##### Simulate mid-size dataset (6776 cells)
# Change the parameters to only simulate the top1200 genes
set.seed(123)
cond_A_param <- SPARSim_create_simulation_parameter(
  intensity = Zheng_C1$intensity[top1200], 
  variability = Zheng_C1$variability[top1200], 
  library_size = sample(Zheng_C1$lib_size, size =  2880, replace = TRUE), 
  condition_name = "cond_A")

cond_B_param <- SPARSim_create_simulation_parameter(
  intensity = Zheng_C2$intensity[top1200], 
  variability = Zheng_C2$variability[top1200], 
  library_size = sample(Zheng_C2$lib_size, size =  3436, replace = TRUE), 
  condition_name = "cond_B")

cond_C_param <- SPARSim_create_simulation_parameter(
  intensity = Zheng_C3$intensity[top1200], 
  variability = Zheng_C3$variability[top1200], 
  library_size = sample(Zheng_C3$lib_size, size =  368, replace = TRUE), 
  condition_name = "cond_C")

cond_D_param <- SPARSim_create_simulation_parameter(
  intensity = Zheng_C4$intensity[top1200], 
  variability = Zheng_C4$variability[top1200], 
  library_size = sample(Zheng_C4$lib_size, size =  92, replace = TRUE), 
  condition_name = "cond_D")

SPARSim_sim_param <- list(cond_A = cond_A_param, cond_B = cond_B_param, cond_C = cond_C_param, cond_D = cond_D_param)

for (i in 1:10){
  set.seed(i)
  # Simulate dataset
  sim_result <- SPARSim_simulation(dataset_parameter = SPARSim_sim_param)
  Zheng_mid<-data.frame(sim_result$count_matrix)
  write.csv(Zheng_mid,paste0("/Users/lbyjo/Desktop/Pingzhao Hu/Simulation/test/Z_mid_",i,".csv"), row.names = TRUE)
  # Generate cell graph
  Z.graph <- buildKNNGraph(Zheng_mid, k=5, d=10)
  write_graph(Z.graph, paste0("/Users/lbyjo/Desktop/Pingzhao Hu/Simulation/test/Z_mid_graph_",i,".txt"), format = c("edgelist"))
}

# Create Labels
cluster_mid<-c(2880, 3436,368,92)
Z_mid_label<-Zheng_mid[1,]
for (i in 1:4){
  Z_mid_label[1,(sum(cluster_mid[0:(i-1)])+1):sum(cluster_mid[0:i])]<-i
}
row.names(Z_mid_label)<-'assigned_cluster' 
Z_mid_label<-data.frame(t(Z_mid_label))
write.csv(Z_mid_label,"/Users/lbyjo/Desktop/Pingzhao Hu/Simulation/test/Z_mid_label.csv", row.names = TRUE)






