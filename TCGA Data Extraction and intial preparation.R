######################################################################
# TCGA RNA-Seq Analysis Script: Top Cancers in Women
# Cancers: BRCA, LUAD, THCA, COAD, OV  ---> Major types of cancers in women
# Steps: Download --> Prepare --> Preprocess --> Normalize --> Filter --> DEA
# Author: John Ong’ala Lunalo
######################################################################

# Load Required Packages -----------------------------------------------------
library(doParallel)
library(TCGAbiolinks)
library(SummarizedExperiment)

# Parallel Setup -------------------------------------------------------------
ncores <- detectCores() - 1  # Use all but one core
cl <- makeCluster(ncores)
registerDoParallel(cl)

# Define Tumor Projects ------------------------------------------------------
tumor_projects <- c("TCGA-BRCA", "TCGA-LUAD", "TCGA-THCA", "TCGA-COAD", "TCGA-OV")

# Query RNA-Seq Expression Data ---------------------------------------------
query.exp <- GDCquery(
  project = tumor_projects,
  legacy = FALSE,
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  platform = "Illumina HiSeq",  # Corrected typo
  file.type = "results",
  experimental.strategy = "RNA-Seq",
  sample.type = c("Primary Tumor")
)

# Download Expression Files --------------------------------------------------
GDCdownload(query.exp, files.per.chunk = 20)

# Prepare Expression Object --------------------------------------------------
AllTumor.exp <- GDCprepare(query = query.exp, save = TRUE, save.filename = "TopCancerinWomen.RDATA")

# Save Query & Expression Data -----------------------------------------------
saveRDS(query.exp, "query_exp.RDS")
saveRDS(AllTumor.exp, "AllTumor_exp.RDS")

# (OPTIONAL) Resume from saved objects ---------------------------------------
# AllTumor.exp <- readRDS("AllTumor_exp.RDS")
# query.exp <- readRDS("query_exp.RDS")

# Get Subtype Information ----------------------------------------------------
tumorvec <- c("luad", "brca", "coad", "thca", "ov")
dataSubt_list <- list()
for (itum in tumorvec) {
  tryCatch(
    dataSubt_list[[itum]] <- TCGAquery_subtype(tumor = itum),
    error = function(e) message(paste("Subtype info missing for:", itum))
  )
}

# Download Clinical Data -----------------------------------------------------
dataClin <- lapply(tumor_projects, function(proj) GDCquery_clinic(proj, type = "clinical"))
names(dataClin) <- tumor_projects
saveRDS(dataClin, "dataClin.RDS")

# Identify Tumor and Normal Samples ------------------------------------------
dataSmTP <- TCGAquery_SampleTypes(getResults(query.exp, cols = "cases"), "TP")
dataSmNT <- TCGAquery_SampleTypes(getResults(query.exp, cols = "cases"), "NT")
saveRDS(dataSmTP, "dataSmTP.RDS")

# Preprocessing --------------------------------------------------------------
dataPrep <- TCGAanalyze_Preprocessing(object = AllTumor.exp, cor.cut = 0.6)

# Download Gene Info (needed for GC-content normalization) -------------------
geneInfo <- TCGAbiolinks::get.GRCh.bioMart()

# Normalization --------------------------------------------------------------
dataNorm <- TCGAanalyze_Normalization(
  tabDF = dataPrep,
  geneInfo = geneInfo,
  method = "gcContent"
)

# Filtering ------------------------------------------------------------------
dataFilt <- TCGAanalyze_Filtering(
  tabDF = dataNorm,
  method = "quantile",
  qnt.cut = 0.25  # Keep top 75% expressed genes
)
saveRDS(dataFilt, "dataFilt.RDS")

# Differential Expression Analysis (DEA) -------------------------------------
dataDEGs <- TCGAanalyze_DEA(
  mat1 = dataFilt[, dataSmNT],
  mat2 = dataFilt[, dataSmTP],
  Cond1type = "Normal",
  Cond2type = "Tumor",
  fdr.cut = 0.01,
  logFC.cut = 1,
  method = "glmLRT"  # edgeR-based method
)
saveRDS(dataDEGs, "dataDEGs.RDS")

# Optional: Stop cluster -----------------------------------------------------
stopCluster(cl)

# End of Script --------------------------------------------------------------
message("TCGA RNA-Seq pipeline completed successfully.")
