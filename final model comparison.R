#NAME: Rupesh Kumar Dey
#TP: TP 061720
#Title: Final 3 models' comparison
########################################################
#Loading necessary packages
library(ISLR)
library(ggplot2) #Library used for plotting graphical representation of the data / model
library(caTools) #The library used for data splitting.
library (caret) #library used for confusionMatrix and knn
library(e1071) #the library used for SVM model.
library(neuralnet) #package used for implementing neural network

#Importing dataset
#Adjust directory accordingly
data <- read.table("D:/Rupesh/Documents/MASTERS IN AI/AML Module 2/BONN dataset/dataset_features.csv", na.strings = NA, sep=",")
data<-data[2:11501,2:11] #selecting relevant columns and rows. 

#checking if there are NA fields in the data
sum (is.na(data)) 

#labelling groups 1,2,3 as epileptic and groups 4 and 5 as non-epileptic. 
data$V11 <- factor(data$V11, levels=c(1,2,3,4,5), labels=c(1,1,1,0,0))

#As the data is in the form of char when imported, this step is done to convert dataset features from character to numerical
data$V2 <- as.numeric(data$V2)
data$V3 <- as.numeric(data$V3)
data$V4 <- as.numeric(data$V4)
data$V5 <- as.numeric(data$V5)
data$V6 <- as.numeric(data$V6)
data$V7 <- as.numeric(data$V7)
data$V8 <- as.numeric(data$V8)
data$V9 <- as.numeric(data$V9)
data$V10 <- as.numeric(data$V10)

#determining features / parameters significance. 
analy<-aov(as.numeric(V11)~.,data = data)
summary(analy)

#Removing insignificant feature column
data<- data[,-4]

###########################################################

#Data normalization
data.norm <- data[,-9]
#Defining the normalization function
norm <- function(x) {  return ((x - min(x)) / (max(x) - min(x)))} 
#Normalizing the x values
data.norm <-as.data.frame(lapply(data.norm, norm))
data.norm
#appending the yvalue label to the normalized data
V11<-data$V11
data.norm <- cbind(data.norm, V11) 

################################################################

#Splitting between training set and test set. 
set.seed(123)
split <- sample.split(data.norm$V11, SplitRatio = 0.7)
training.set <- subset(data.norm, split == TRUE)
test.set = subset(data.norm, split == FALSE)

###########################################################################

#SVM MODEL TRAINING
start.time <- Sys.time()
svm <- svm (V11~., data = training.set, epsilon = 0, cost = 8)
end.time<-Sys.time()
svm.time.train<-end.time-start.time

# Testing SVM MODEL calculating accuracy. 
start.time <- Sys.time()
pred.svm = predict (svm, test.set)
end.time<-Sys.time()
svm.time.test<-end.time-start.time

#CALCULATING CONFISION MATRIX AND ACCURACY
cm.svm = table(Predicted = pred.svm, Actual = test.set$V11)
cm.svm
err.svm<-1-sum(diag(cm.svm))/sum(cm.svm)
err.svm
acc.svm<-(1-err.svm)*100
#CALCULATING SENSITIVITY OF THE SVM MODEL
sen.svm<-confusionMatrix(pred.svm, test.set$V11)
sen.svm <- (sen.svm$byClass[1])*100

###########################################################################

#KNN MODEL TRAINING
ctrl.10 <- trainControl(method="repeatedcv",repeats = 10)

start.time <- Sys.time()
knnFit.10 <- train(V11 ~ ., data = training.set, method = "knn", trControl = ctrl.10, tuneLength = 20)
end.time<-Sys.time()
knn.time.train<-end.time-start.time

# Testing KNN MODEL 
start.time <- Sys.time()
knnPredict.10 <- predict(knnFit.10,newdata = test.set)
end.time<-Sys.time()
knn.time.test<-end.time-start.time

###########################################################################

#CALCULATING CONFISION MATRIX AND ACCURACY KNN
cm.knn = table(Predicted = knnPredict.10, Actual = test.set$V11)
cm.knn
err.knn<-1-sum(diag(cm.knn))/sum(cm.knn)
acc.knn<-(1-err.knn)*100

#calculating sensitivity for KNN model
sen.knn<-confusionMatrix(knnPredict.10, test.set$V11)
sen.knn<-(sen.knn$byClass[1])*100

####################################################################

#ANN MODEL TRAIN
start.time <- Sys.time()
nn3 <- neuralnet(V11 ~ V2+V3+V4+V6+V7+V8+V9+V10, data = training.set,hidden=5,threshold = 0.1,stepmax = 1e+7,act.fct = "logistic",linear.output=FALSE, err.fct = "ce",likelihood = TRUE,)
end.time<-Sys.time()
ann.time.train<-end.time-start.time

temp.test <- subset(test.set, select = c("V2","V3","V4","V6","V7","V8","V9","V10"))
#TEST ANN MODEL
start.time <- Sys.time()
nn3.results <- compute(nn3, temp.test)
end.time<-Sys.time()
ann.time.test<-end.time-start.time

#CONFUSTION MATRIX AND ACCURACY CALCULATION
results3 <- data.frame(prediction3 = nn3.results$net.result[,2],actual = test.set$V11)
pred3 <- ifelse(results3[,1]>0.5, 1, 0)
pred3<-factor(pred3, levels=c(1,0), labels=c(1,0))
cm3 = table(Predicted= pred3,Actual = test.set$V11)
err.3<-1-sum(diag(cm3))/sum(cm3)
err.3
acc.3<-(1-err.3)*100

#SENSITIVITY CALCULATION. 
sen.3<-confusionMatrix(pred3, test.set$V11 )
sen.3<-(sen.3$byClass[1])*100

############################################################
#Visualizing models' accuracy 
counts <- c(signif(acc.svm,digits = 4),signif(acc.3,digits = 4),signif(acc.knn,digits = 4))
x<-barplot(counts, main="Accuracy of the best SVM vs ANN vs KNN models",
           xlab="Type of Machine Learning Models",ylab="Accuracy (%)",names.arg=c("SVM","ANN","KNN"),ylim = c( 0 , 100 ),col=rainbow(3))
y<-as.matrix(counts)
text(x,y+2,labels=as.character(y))

#Visualizing model's sensitivity 
counts <- c(signif(sen.svm,digits = 4),signif(sen.3,digits = 4),signif(sen.knn,digits = 4))
x<-barplot(counts, main="Sensitivity of the best SVM vs ANN vs KNN models",
           xlab="Type of Machine Learning Models",ylab="Sensitivity (%)",names.arg=c("SVM","ANN","KNN"),ylim = c( 0 , 100 ),col = rainbow(3))
y<-as.matrix(counts)
text(x,y+2,labels=as.character(y))

###############################################
#ROC curve
library(ROCR)
predvec.svm <- ifelse(pred.svm=="1", 1, 0) #predict holding test
realvec.svm <- ifelse(test.set$V11=="1", 1, 0) #compare against test
preda.svm <- prediction(predvec.svm,realvec.svm)
perf.svm <- performance(preda.svm, measure = "tpr", x.measure = "fpr")
plot(perf.svm, main = "ROC curve for SVM Classifier",colorize = T, lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
#converting auc to numeric form from the graph area
auc.svm = as.numeric(performance(preda.svm, "auc")@y.values)
auc.svm = round(auc.svm, 3)
legend("bottom", c("AUC",auc.svm*100))

###############################################
#ROC curve
library(ROCR)
predvec.knn <- ifelse(knnPredict.10=="1", 1, 0) #predict holding test
realvec.knn <- ifelse(test.set$V11=="1", 1, 0) #compare against test
preda.knn <- prediction(predvec.knn,realvec.knn)
perf.knn <- performance(preda.knn, measure = "tpr", x.measure = "fpr")
plot(perf.knn, main = "ROC curve for KNN Classifier",colorize = T, lwd = 3,print.cutoffs.at=seq(0,1,0.3),text.adj= c(-0.2,1.7))
abline(a = 0, b = 1, lwd = 2, lty = 2)
#converting auc to numeric form from the graph area
auc.knn = as.numeric(performance(preda.knn, "auc")@y.values)
auc.knn = round(auc.knn, 3)
legend("bottom", c("AUC",auc.knn*100))

###############################################
#ROC curve
library(ROCR)
predvec.ann <- ifelse(pred3=="1", 1, 0) #predict holding test
realvec.ann <- ifelse(test.set$V11=="1", 1, 0) #compare against test
preda.ann <- prediction(predvec.ann,realvec.ann)
perf.ann <- performance(preda.ann, measure = "tpr", x.measure = "fpr")
plot(perf.ann, main = "ROC curve for ANN Classifier",colorize = T, lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
#converting auc to numeric form from the graph area
auc.ann = as.numeric(performance(preda.ann, "auc")@y.values)
auc.ann = round(auc.ann, 3)
legend("bottom", c("AUC",auc.ann*100))

