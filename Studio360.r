#title: "Projeto2"
#author: "Fernando Aguiar, Lucas Gehlen e Lucas Gerlach Nachtigall"
#date: "26/11/2022"

rm(list=ls())

setwd("~/")
dados = read.csv('Studio360.csv',sep=';',dec=',')

################################################################################
# 1.1. REMO��O DAS LINHAS COM NA                                                 

dados = na.omit(dados)

################################################################################
# 1.2 PREPARA��O DA ESTRUTURA                                                   
dados$range <- as.integer(dados$range)
dados$range = factor(dados$range, #
                      levels = 1:3, 
                      labels = c("a", "b", "c"))


################################################################################
# 1.3 DIVIS�O DO CONJUNTO DE DADOS                                              

# NAO PARAMETRICOS

library(caret)
set.seed(1)
indicesTreinamento = createDataPartition(dados$range, p = 2/3, list = FALSE)

dadosTreinamento   = dados[indicesTreinamento,-ncol(dados)]
rotulosTreinamento = dados[indicesTreinamento,ncol(dados)]

dadosTeste         = dados[-indicesTreinamento,-ncol(dados)]
rotulosTeste       = dados[-indicesTreinamento,ncol(dados)]
rm(dados)

################################################################################
# 1.4. DETEC��O E REMO��O DOS OUTLIERS                                         
outliers = c()
for (i in 1:length(dadosTreinamento)) {
  outliers = c(outliers,which(dadosTreinamento[,i] %in% boxplot.stats(dadosTreinamento[,i])$out))
}
outliers = unique(outliers) 
dadosTreinamento   = dadosTreinamento[-outliers,]
rotulosTreinamento = rotulosTreinamento[-outliers]
rm(outliers,i)

################################################################################
# 1.5. NORMALIZA��O DOS DADOS (Escore-Z)                                         

normalizacaoParametros = preProcess(dadosTreinamento,method = c("center","scale"))
dadosTreinamento = predict(normalizacaoParametros, dadosTreinamento)
dadosTeste       = predict(normalizacaoParametros, dadosTeste)
rm(normalizacaoParametros)

################################################################################
# 2. TREINAMENTO                                                               #
################################################################################

################################################################################
# 2.1. ESTIMA��O DOS MODELOS

library(doParallel)
clusters = makePSOCKcluster(detectCores(logical = FALSE))
registerDoParallel(clusters)

set.seed(1)
# Repeated Cross Validation (repete 5 vezes uma 10-fold CV)
paramsTreinamento = trainControl(method = "repeatedcv", number = 10, 
                                 repeats = 5, classProbs = TRUE)

knn.modelo = train(dadosTreinamento,rotulosTreinamento,    
                   preProcess = c("center", "scale"),
                   trControl = paramsTreinamento,
                   tuneLength = 10,   # N�mero m�ximo de combina��o de par�metros
                   method="knn")
# 
set.seed(1)
cart.modelo = train(dadosTreinamento,rotulosTreinamento,    
                    trControl = paramsTreinamento,
                    method="rpart")

library(rpart.plot)
prp(cart.modelo$finalModel)

set.seed(1)
bagging.modelo    = train(dadosTreinamento,rotulosTreinamento,    
                          trControl = paramsTreinamento,
                          method="treebag")

set.seed(1)
rf.modelo         = train(dadosTreinamento,rotulosTreinamento,     
                          trControl = paramsTreinamento,
                          method="rf")

set.seed(1)

svmLinear.modelo  = train(dadosTreinamento,rotulosTreinamento,     
                          trControl = paramsTreinamento,
                          method="svmLinear")

set.seed(1)
svmPolinomial.modelo = train(dadosTreinamento,rotulosTreinamento,     
                             trControl = paramsTreinamento,
                             method="svmPoly")

set.seed(1)
svmRadial.modelo  = train(dadosTreinamento,rotulosTreinamento,     
                          trControl = paramsTreinamento,
                          method="svmRadial")

stopCluster(clusters)

################################################################################
# 3. TESTE                                                                     #
################################################################################

################################################################################
# 3.1. AVALIA��O DE DESEMPENHO
knn.predicao           = predict(knn.modelo, dadosTeste)
cart.predicao          = predict(cart.modelo, dadosTeste)
bagging.predicao       = predict(bagging.modelo, dadosTeste)
rf.predicao            = predict(rf.modelo, dadosTeste)
#ada.predicao           = predict(ada.modelo, dadosTeste)
svmLinear.predicao     = predict(svmLinear.modelo, dadosTeste)
svmPolinomial.predicao = predict(svmPolinomial.modelo, dadosTeste)
svmRadial.predicao     = predict(svmRadial.modelo, dadosTeste)

resultado.teste = data.frame(Classificador=character(), Acuracia=double(),
                             AcuraciaInferior=double(), AcuraciaSuperior=double(),
                             stringsAsFactors = FALSE)

res = confusionMatrix(knn.predicao, as.factor(rotulosTeste))
resultado.teste[1,] = data.frame("K-nearest neighbor",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

res = confusionMatrix(cart.predicao,rotulosTeste)
resultado.teste[2,] = data.frame("�rvore de Decis�o",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

res = confusionMatrix(bagging.predicao,rotulosTeste)
resultado.teste[3,] = data.frame("Bagging",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

res = confusionMatrix(rf.predicao,rotulosTeste)
resultado.teste[4,] = data.frame("Random Forest",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

#res = confusionMatrix(ada.predicao,rotulosTeste)
#resultado.teste[5,] = data.frame("Boosting",res$overall["Accuracy"],
#                                 res$overall["AccuracyLower"], 
#                                 res$overall["AccuracyUpper"])

res = confusionMatrix(svmLinear.predicao,rotulosTeste)
resultado.teste[6,] = data.frame("SVM Linear",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

res = confusionMatrix(svmPolinomial.predicao,rotulosTeste)
resultado.teste[7,] = data.frame("SVM Polinomial",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

res = confusionMatrix(svmRadial.predicao,rotulosTeste)
resultado.teste[8,] = data.frame("SVM Radial",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

ggplot(resultado.teste, aes(Classificador, Acuracia)) + geom_point() + 
  geom_errorbar(aes(ymin=AcuraciaInferior , ymax=AcuraciaSuperior), width=.2,
                position=position_dodge(.9)) + ylim(0.85, 1) +
  ggtitle("Acur�cia do Teste") 

# param�tricos

################################################################################
# 2.1. ESTIMA��O DOS MODELOS

library(mclust)
set.seed(1)
gmm.modelo = MclustDA(dadosTreinamento,rotulosTreinamento,verbose=FALSE)

set.seed(1)
# Define o n�mero de componentes para cada GMM
gmm.modelo.componentes = MclustDA(dadosTreinamento,
                                  rotulosTreinamento, 4,verbose=FALSE)

set.seed(1)
# Define o a matriz de covari�ncia como diagonal
gmm.modelo.modeloVVI = MclustDA(dadosTreinamento, rotulosTreinamento, 
                                modelNames = c("VVI"),verbose=FALSE)

set.seed(1)
# Define a matriz de covari�ncia completa
gmm.modelo.modeloVVV = MclustDA(dadosTreinamento,
                                rotulosTreinamento, 
                                modelNames = c("VVV"),verbose=FALSE)

# C�digo que detecta n�mero de cores e define threads de execu��o 
library(doParallel)
numeroCores = detectCores(logical = FALSE) # Detecta o n�mero de cores f�sicos
clusters = makePSOCKcluster(numeroCores)
registerDoParallel(clusters)

library(tictoc)
tic("Treinamento Rede Neural")
set.seed(1)
paramsTreinamento = trainControl(classProbs = TRUE)
tic("mlp")
mlp.modelo = train(dadosTreinamento, rotulosTreinamento,
                   trControl = paramsTreinamento,
                   method="mlp")
toc()

set.seed(1)
tic("nnet")
nnet.modelo = train(dadosTreinamento, rotulosTreinamento,
                    trControl = paramsTreinamento,
                    method="nnet")
toc()
toc()

stopCluster(clusters)

################################################################################
# 2.2. AVALIAR DADOS DE TREINAMENTO

gmm.predicao              = predict(gmm.modelo,dadosTreinamento)
gmm.componentes.predicao  = predict(gmm.modelo.componentes,dadosTreinamento)
gmm.modeloVVI.predicao    = predict(gmm.modelo.modeloVVI,dadosTreinamento)
gmm.modeloVVV.predicao    = predict(gmm.modelo.modeloVVV,dadosTreinamento)
mlp.predicao              = predict(mlp.modelo,dadosTreinamento)
nnet.predicao             = predict(nnet.modelo,dadosTreinamento)

resultado.treinamento = data.frame(Classificador=character(), Acuracia=double(),
                                   AcuraciaInferior=double(), AcuraciaSuperior=double(),
                                   stringsAsFactors = FALSE)

res = confusionMatrix(gmm.predicao$classification,rotulosTreinamento)
resultado.treinamento[1,] = data.frame("GMM Default",res$overall["Accuracy"],
                                       res$overall["AccuracyLower"], 
                                       res$overall["AccuracyUpper"])

res = confusionMatrix(gmm.componentes.predicao$classification,rotulosTreinamento)
resultado.treinamento[2,] = data.frame("GMM 4 Componentes",res$overall["Accuracy"],
                                       res$overall["AccuracyLower"], 
                                       res$overall["AccuracyUpper"])

res = confusionMatrix(gmm.modeloVVI.predicao$classification,rotulosTreinamento)
resultado.treinamento[3,] = data.frame("GMM Diagonal",res$overall["Accuracy"],
                                       res$overall["AccuracyLower"], 
                                       res$overall["AccuracyUpper"])

res = confusionMatrix(gmm.modeloVVV.predicao$classification,rotulosTreinamento)
resultado.treinamento[4,] = data.frame("GMM Completo",res$overall["Accuracy"],
                                       res$overall["AccuracyLower"], 
                                       res$overall["AccuracyUpper"])

res = confusionMatrix(mlp.predicao,rotulosTreinamento)
resultado.treinamento[5,] = data.frame("MLP",res$overall["Accuracy"],
                                       res$overall["AccuracyLower"], 
                                       res$overall["AccuracyUpper"])

res = confusionMatrix(nnet.predicao,rotulosTreinamento)
resultado.treinamento[6,] = data.frame("NNET",res$overall["Accuracy"],
                                       res$overall["AccuracyLower"], 
                                       res$overall["AccuracyUpper"])

ggplot(resultado.treinamento, aes(Classificador, Acuracia)) + geom_point() + 
  geom_errorbar(aes(ymin=AcuraciaInferior , ymax=AcuraciaSuperior), width=.2,
                position=position_dodge(.9)) + ylim(0.9, 1) +
  ggtitle("Acur�cia do Treinamento") 

################################################################################
# 3. TESTE                                                                     #
################################################################################

################################################################################
# 3.1. AVALIA��O DE DESEMPENHO
gmm.predicao              = predict(gmm.modelo,dadosTeste)
gmm.componentes.predicao  = predict(gmm.modelo.componentes,dadosTeste)
gmm.modeloVVI.predicao    = predict(gmm.modelo.modeloVVI,dadosTeste)
gmm.modeloVVV.predicao    = predict(gmm.modelo.modeloVVV,dadosTeste)
mlp.predicao              = predict(mlp.modelo,dadosTeste)
nnet.predicao             = predict(nnet.modelo,dadosTeste)

resultado.teste = data.frame(Classificador=character(), Acuracia=double(),
                             AcuraciaInferior=double(), AcuraciaSuperior=double(),
                             stringsAsFactors = FALSE)

res = confusionMatrix(gmm.predicao$classification,rotulosTeste)
resultado.teste[1,] = data.frame("GMM Default",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

res = confusionMatrix(gmm.componentes.predicao$classification,rotulosTeste)
resultado.teste[2,] = data.frame("GMM 4 Componentes",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

res = confusionMatrix(gmm.modeloVVI.predicao$classification,rotulosTeste)
resultado.teste[3,] = data.frame("GMM Diagonal",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

res = confusionMatrix(gmm.modeloVVV.predicao$classification,rotulosTeste)
resultado.teste[4,] = data.frame("GMM Completo",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

res = confusionMatrix(mlp.predicao,rotulosTeste)
resultado.teste[5,] = data.frame("MLP",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

res = confusionMatrix(nnet.predicao,rotulosTeste)
resultado.teste[6,] = data.frame("NNET",res$overall["Accuracy"],
                                 res$overall["AccuracyLower"], 
                                 res$overall["AccuracyUpper"])

ggplot(resultado.teste, aes(Classificador, Acuracia)) + geom_point() + 
  geom_errorbar(aes(ymin=AcuraciaInferior , ymax=AcuraciaSuperior), width=.2,
                position=position_dodge(.9)) + ylim(0.9, 1) +
  ggtitle("Acur�cia do Teste") 
