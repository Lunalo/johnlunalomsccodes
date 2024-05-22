library(doParallel)
ncores <- detectCores() - 1
cl <- makeCluster(ncores)
registerDoParallel(cl)

#setwd("C:\\Users\\John\\OneDrive\\Masters Thesis Analysis\\DataWindows\\Project")

#setwd("/Users/johnlunalo/Library/CloudStorage/OneDrive-Personal/Masters Thesis Analysis/DataWindows/Project")

library(SummarizedExperiment)
library(TCGAbiolinks)
query.exp <- GDCquery(project = c("TCGA-BRCA","TCGA-LUAD","TCGA-THCA","TCGA-COAD","TCGA-OV"),
                      legacy = FALSE,
                      data.category = "Transcriptome Profiling",
                      data.type = "Gene Expression Quantification",
                      platform = "lluminaHiSeq_RNASeq",
                      #platform = "HT_HG-U133A",
                      file.type = "results",
                      experimental.strategy = "RNA-Seq",
                      sample.type = c("Primary Tumor"))


GDCdownload(query.exp, files.per.chunk = 20)

AllTumor.exp <- GDCprepare(query = query.exp, save = TRUE, save.filename = "TopCancerinWomen.RDATA")

#Save query details
saveRDS(query.exp, "query_exp.RDS")
saveRDS(AllTumor.exp, "AllTumor_exp.RDS")
###########################################################################################################

#####RUN FROM HERE

AllTumor_exp <- readRDS("AllTumor_exp.RDS")
query_exp <- readRDS("query_exp.RDS")

#https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations
# get subtype information
tumorvec <- c("luad", "brca", "coad","thca","ov")
dataSubt_list <- list()

for(itum in tumorvec){
  tryCatch(
  dataSubt_list[[itum]]<- TCGAquery_subtype(tumor = itum),
  error= function(e)print(paste(itum, "missing in code list")))
}

# get clinical data
proj <- c("TCGA-BRCA", "TCGA-LUAD", "TCGA-OV", "TCGA-COAD", "TCGA-THCA")
dataClin <- vector("list", length(proj))
for (i in proj){
  dataClin[[i]] <- GDCquery_clinic(project = i,"clinical")
}
saveRDS(dataClin, "dataClin.RDS")
#dataClin <- GDCquery_clinic(project = c("TCGA-HNSC", "TCGA-LGG", "TCGA-SARC", "TCGA-ESCA", "TCGA-UCEC", "TCGA-GBM", "TCGA-KIRC", "TCGA-PRAD", "TCGA-THCA", "TCGA-CESC", "TCGA-COAD", "TCGA-SKCM", "TCGA-LUSC", "TCGA-BRCA", "TCGA-LUAD", "TCGA-OV", "TCGA-UCS", "TCGA-ACC", "TCGA-DLBC", "TCGA-LIHC", "TCGA-KIRP", "TCGA-LAML", "TCGA-KICH", "TCGA-READ", "TCGA-TGCT", "TCGA-PAAD", "TCGA-PCPG", "TCGA-THYM", "TCGA-CHOL", "TCGA-MESO", "TCGA-UVM", "TCGA-BLCA", "TCGA-STAD"),"clinical") 

# Which samples are primary solid tumor
dataSmTP <- TCGAquery_SampleTypes(getResults(query_exp,cols="cases"),"TP") 
saveRDS(dataSmTP, "dataSmTP.RDS")
dataSmTP <- readRDS("dataSmTP.RDS")
# which samples are solid tissue normal
dataSmNT <- TCGAquery_SampleTypes(getResults(query_exp,cols="cases"),"NT")

dataPrep <- TCGAanalyze_Preprocessing(object = AllTumor.exp, cor.cut = 0.6)                      

dataNorm <- TCGAanalyze_Normalization(tabDF = dataPrep,
                                      geneInfo = geneInfo,
                                      method = "gcContent")                

dataFilt <- TCGAanalyze_Filtering(tabDF = dataNorm,
                                  method = "quantile", 
                                  qnt.cut =  0.25) 
saveRDS(dataFilt, "dataFilt.RDS")

dataDEGs <- TCGAanalyze_DEA(mat1 = dataFilt[,dataSmNT],
                            mat2 = dataFilt[,dataSmTP],
                            Cond1type = "Normal",
                            Cond2type = "Tumor",
                            fdr.cut = 0.01 ,
                            logFC.cut = 1,
                            method = "glmLRT")

saveRDS(dataDEGs, "dataDEGs.RDS")
