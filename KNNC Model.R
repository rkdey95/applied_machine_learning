#NAME: RUPESH KUMAR DEY
#TP: TP061720
#Title: KNN model
########################################
#loading necessary packages
library(ISLR)
library(caret) #Required package for implementing the knnc classifier. 
library(caTools) #The library used for data splitting.

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

########################################################
#Splitting between training set and test set. 
set.seed(123)
split <- sample.split(data.norm$V11, SplitRatio = 0.7)
training.set <- subset(data.norm, split == TRUE)
test.set = subset(data.norm, split == FALSE)

#checking purity of data
prop.table(table(training.set$V11))
library (ineq)
ineq(training.set$V11,type="Gini")
prop.table(table(test.set$V11))

set.seed(400)
######################################################################
#Defining the cross validation function technique
ctrl.3 <- trainControl(method="repeatedcv",repeats = 3)
ctrl.5 <- trainControl(method="repeatedcv",repeats = 5)
ctrl.7 <- trainControl(method="repeatedcv",repeats = 7)
ctrl.10 <- trainControl(method="repeatedcv",repeats = 10)
##########################################################################
#Training the 3 different KNN models with 3x CV, 5x CV and 10 x CV. Tunelength ie. the number of instances to be taken into account for classiciation is set at 20
knnFit.3 <- train(V11 ~ ., data = training.set, method = "knn", trControl = ctrl.3 , tuneLength = 20)#preProcess = c("center","scale")
knnFit.5 <- train(V11 ~ ., data = training.set, method = "knn", trControl = ctrl.5 , tuneLength = 20)#preProcess = c("center","scale")
knnFit.7 <- train(V11 ~ ., data = training.set, method = "knn", trControl = ctrl.7, tuneLength = 20)
knnFit.10 <- train(V11 ~ ., data = training.set, method = "knn", trControl = ctrl.10, tuneLength = 20)

#Tuning the best model of cv x 10 with tuneLength = 30
knnFit.10.30 <- train(V11 ~ ., data = training.set, method = "knn", trControl = ctrl.10, tuneLength = 30)

#Output of KNN fit
knnFit.3
knnFit.5
knnFit.7
knnFit.10
knnFit.10.30

# Visualizing the KNN fitted model to training set
plot(knnFit.3)
plot(knnFit.5)
plot(knnFit.7)
plot(knnFit.10)
plot(knnFit.10.30)

#Predicting the output of the KNN model by feeding input from test set. 
knnPredict.3 <- predict(knnFit.3,newdata = test.set )
knnPredict.5 <- predict(knnFit.5,newdata = test.set )
knnPredict.7 <- predict(knnFit.7,newdata = test.set )
knnPredict.10 <- predict(knnFit.10,newdata = test.set)
knnPredict.10.30 <- predict(knnFit.10.30,newdata = test.set)
##########################################################################
#KNN 3X CV
#Computing confusion matrix and accuracy
cm.3 = table(Predicted = knnPredict.3, Actual = test.set$V11)
cm.3
err.3<-1-sum(diag(cm.3))/sum(cm.3)
acc.3<-(1-err.3)*100
##########################################################################
#KNN 5X CV
cm.5 = table(Predicted = knnPredict.5, Actual = test.set$V11)
cm.5
err.5<-1-sum(diag(cm.5))/sum(cm.5)
acc.5<-(1-err.5)*100
##########################################################################
#KNN 7X CV
cm.7 = table(Predicted = knnPredict.7, Actual = test.set$V11)
cm.7
err.7<-1-sum(diag(cm.7))/sum(cm.7)
acc.7<-(1-err.7)*100
###########################################################################
#KNN 10X CV
cm.10 = table(Predicted = knnPredict.10, Actual = test.set$V11)
cm.10
err.10<-1-sum(diag(cm.10))/sum(cm.10)
acc.10<-(1-err.10)*100
###########################################################################
#Visualizing model's accuracy with tuneLength = 20
counts <- c(signif(acc.3,digits = 4),signif(acc.5,digits = 4),signif(acc.7,digits = 4),signif(acc.10,digits=4))
x<-barplot(counts, main="Accuracy of KNN with 3 times CV vs 5 times CV vs 10 times CV",
           xlab="Number of times for Cross Validation",ylab="Accuracy (%)",names.arg=c("3","5","7","10"),ylim = c( 0 , 100 ) )
y<-as.matrix(counts)
text(x,y+2,labels=as.character(y))

#######################################
#KNN 10X CV with tuneLength = 30
cm.10.30 = table(Predicted = knnPredict.10.30, Actual = test.set$V11)
cm.10.30
err.10.30<-1-sum(diag(cm.10.30))/sum(cm.10.30)
acc.10.30<-(1-err.10.30)*100

###########################################################################
#Visualizing model's accuracy with 10 x CV with tunelength = 30

counts <- c(signif(acc.3,digits = 4),signif(acc.5,digits = 4),signif(acc.7,digits = 4),signif(acc.10,digits=4),signif(acc.10.30,digits=4))
x<-barplot(counts, main="Accuracy of different types of KNN Classifiers with different hyperparameters",
           xlab="Number of times for Cross Validation",ylab="Accuracy (%)",names.arg=c("3","5","7","10","10 Tune Length 30"),ylim = c( 0 , 100 ) )
y<-as.matrix(counts)
text(x,y+2,labels=as.character(y))
###########################################################################

#Calculating the sensitivity of each model
sen.3 <- confusionMatrix(knnPredict.3, test.set$V11)
sen.5 <-confusionMatrix(knnPredict.5, test.set$V11)
sen.10<-confusionMatrix(knnPredict.10, test.set$V11)
sen.10.30 <-confusionMatrix(knnPredict.10.30, test.set$V11)

sen.3 <- (sen.3$byClass[1])*100
sen.5 <-(sen.5$byClass[1])*100
sen.10<-(sen.10$byClass[1])*100
sen.10.30 <-(sen.10.30$byClass[1])*100
###########################################################################
#Visualizing model's sensitivity 
counts <- c(signif(sen.3,digits = 4),signif(sen.5,digits = 4),signif(sen.10,digits = 4),signif(sen.10,digits=4),signif(sen.10.30,digits=4))
x<-barplot(counts, main="Sensitivity of different types of KNN Classifiers with different hyperparameters",
           xlab="Number of times for Cross Validation",ylab="Sensitivity (%)",names.arg=c("3","5","7","10","10 Tune Length 30"),ylim = c( 0 , 100 ) )
y<-as.matrix(counts)
text(x,y+2,labels=as.character(y))
