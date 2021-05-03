#NAME: Rupesh Kumar Dey
#TP: TP 061720
#Title: SVM model
################################################################
#Importing necessary library packages
library(ggplot2) #Library use presenting grpahical representation of the data / model
library(e1071) #the library used for SVM model.
library(caTools) #The library used for data splitting. 
library(caret) #additional library to use confusionMatrix() function 

#Importing dataset
#Adjust directory accordingly
data <- read.table("D:/Rupesh/Documents/MASTERS IN AI/AML Module 2/BONN dataset/dataset_features.csv", na.strings = "", sep=",")
data<-data[2:11491,2:11] #selecting relevant columns and rows. 

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

#Splitting dataset between training set and test set. 
set.seed(123)
split <- sample.split(data.norm$V11, SplitRatio = 0.7)
training.set <- subset(data.norm, split == TRUE)
test.set = subset(data.norm, split == FALSE)

#checking purity of data
prop.table(table(training.set$V11))
prop.table(table(test.set$V11))

###########################################################################
#Default SVM Model with RBF kernal
svm.rbf <- svm(V11~., data = training.set)
summary(svm.rbf)

#predicting and Generating Confusion Matrix for rbf model
pred.rbf = predict(svm.rbf, test.set)
cm.rbf = table(Predicted = pred.rbf, Actual = test.set$V11)
cm.rbf
err.rbf<-1-sum(diag(cm.rbf))/sum(cm.rbf) #calculating the error %
acc.rbf<-(1-err.rbf)*100
###########################################################################
# SVM Model with linear kernal
svm.linear = svm (V11~., data = training.set,kernel = "linear")
summary (svm.linear)

#Predicting and generating Confusion Matrix
pred.lin = predict (svm.linear, test.set)
cm.lin = table(Predicted = pred.lin, Actual = test.set$V11)
cm.lin
err.lin<-1-sum(diag(cm.lin))/sum(cm.lin)
acc.lin<-(1-err.lin)*100
###########################################################################
# SVM model with sigmoid kernal
svm.sigmoid = svm (V11~., data = training.set,kernel = "sigmoid")
summary (svm.sigmoid)

#Predicting and generating Confusion Matrix
pred.sig = predict (svm.sigmoid, test.set)
cm.sig = table(Predicted = pred.sig, Actual = test.set$V11)
cm.sig
err.sig<-1-sum(diag(cm.sig))/sum(cm.sig)
acc.sig<-(1-err.sig)*100
##########################################################################
# Simple Bar Plot plotting the comparison of accuracy between 3 different kernels
counts <- c(signif(acc.rbf,digits = 4),signif(acc.lin,digits=4),signif(acc.sig,digits=4))
x<-barplot(counts, main="Accuracy of SVM with rbf vs linear vs sigmoid kernals",
        xlab="Type of Kernal",ylab="Accuracy (%)",names.arg=c("rbf","linear","sigmoid"),ylim = c( 0 , 100 ) )
y<-as.matrix(counts)
text(x,y+2,labels=as.character(y))
###########################################################################
#model hyper parameter tuning using tune function. The Kernal used is rbf as it gave the best accuracy when comparing kernals:
svm.rbf.tuned = tune(svm, V11~., data=training.set,
                   ranges = list(epsilon = seq (0, 2, 0.1), cost = 2^(0:3)))
#Plotting the tuning parameters
plot (svm.rbf.tuned) 
summary (svm.rbf.tuned)
opt_model = svm.rbf.tuned$best.model 
summary(opt_model) #extracting the parameters of the best model
###########################################################################
#Building and training the Best model based on tuning parameters
#training time evaluated as well
start.time <- Sys.time()
svm.tuned <- svm (V11~., data = training.set, epsilon = 0, cost = 4)
end.time<-Sys.time()
end.time-start.time

# Predicting and generating Confusion Matrix (prediction time is calculated as well)
start.time <- Sys.time()
pred.tuned = predict (svm.tuned, test.set)
end.time<-Sys.time()
end.time-start.time

#Confusion matrix
cm.tuned = table(Predicted = pred.tuned, Actual = test.set$V11)
cm.tuned
err.tuned<-1-sum(diag(cm.tuned))/sum(cm.tuned)
err.tuned
acc.tuned<-(1-err.tuned)*100
###########################################################################
#Visualizing tuned model accuracy to previous models. 
counts <- c(signif(acc.tuned,digits = 4),signif(acc.rbf,digits = 4),signif(acc.lin,digits=4),signif(acc.sig,digits=4))
x<-barplot(counts, main="Accuracy of SVM with rbf.tuned vs rbf vs linear vs sigmoid kernals",
           xlab="Type of Kernal",ylab="Accuracy (%)",names.arg=c("rbf.tuned","rbf","linear","sigmoid"),
           ylim = c( 0 , 100 ),col=rgb(0.2,0.4,0.6,0.6) )
y<-as.matrix(counts)
text(x,y+2,labels=as.character(y))
###########################################################################
#evaluating sensitivity
sen.tuned<-confusionMatrix(pred.tuned, test.set$V11)
sen.rbf<-confusionMatrix(pred.rbf, test.set$V11 )
sen.lin<-confusionMatrix(pred.lin, test.set$V11 )
sen.sig<-confusionMatrix(pred.sig, test.set$V11 )

#converting to %
sen.tuned <- (sen.tuned$byClass[1])*100
sen.rbf <-(sen.rbf$byClass[1])*100
sen.lin<-(sen.lin$byClass[1])*100
sen.sig <-(sen.sig$byClass[1])*100
###########################################################################
#Visualizing model's sensitivity results
counts <- c(signif(sen.tuned,digits = 4),signif(sen.rbf,digits = 4),signif(sen.lin,digits = 4),signif(sen.sig,digits=4))
x<-barplot(counts, main="Sensitivity of different types of SVM Classifiers with different hyperparameters",
           xlab="Type of Kernal",ylab="Sensitivity (%)",names.arg=c("rbf.tuned","rbf","linear","sigmoid"),ylim = c( 0 , 100 ) )
y<-as.matrix(counts)
text(x,y+2,labels=as.character(y))

############################################################################
#For reference only
# #ROC curve
# library(ROCR)
# predvec <- ifelse(pred.tuned=="1", 1, 0) #predict holding test
# realvec <- ifelse(test.set$V11=="1", 1, 0) #compare against test
# pred <- prediction(predvec,realvec)
# perf <- performance(pred, measure = "tpr", x.measure = "fpr")
# plot(perf, main = "ROC curve for Naive Bayes Classifier",col = "blue", lwd = 3,print.auc = TRUE)
# abline(a = 0, b = 1, lwd = 2, lty = 2)
# 
# #converting auc to numeric form from the graph area
# auc = as.numeric(performance(pred, "auc")@y.values)
# auc = round(auc, 3)
# auc
