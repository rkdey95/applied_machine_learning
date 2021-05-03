#NAME: Rupesh Kumar Dey
#TP: TP 061720
#Title: ANN Model
################################################################
#importing necessary packages
library(ggplot2) #Library use presenting grpahical representation of the data / model
library(caTools) #The library used for data splitting.
library (caret)

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
##############
########################################################
#Splitting between training set and test set. 
set.seed(123)
split <- sample.split(data.norm$V11, SplitRatio = 0.7)
training.set <- subset(data.norm, split == TRUE)
test.set = subset(data.norm, split == FALSE)

#checking purity of data
prop.table(table(training.set$V11))
prop.table(table(test.set$V11))
memory.limit(size = 2e+9)

library(neuralnet) #package used for implementing neural network
#############################################
#Building the neural network with single node 1 layer
set.seed(123)
#Implementing the basic NN model with only 1 neuron. 
#training time evaluated
start.time <- Sys.time()
nn1 <- neuralnet(V11 ~ V2+V3+V4+V6+V7+V8+V9+V10, data = training.set,hidden=1,threshold = 0.1,stepmax = 1e+7,act.fct = "logistic",linear.output=FALSE, err.fct = "ce",likelihood = TRUE,)
end.time<-Sys.time()
end.time-start.time

#Implementing the tuned NN model with only 3 neuron. 
#training time evaluated
start.time <- Sys.time()
nn2 <- neuralnet(V11 ~ V2+V3+V4+V6+V7+V8+V9+V10, data = training.set,hidden=3,threshold = 0.1,stepmax = 1e+7,act.fct = "logistic",linear.output=FALSE, err.fct = "ce",likelihood = TRUE,)
end.time<-Sys.time()
end.time-start.time

#Implementing the tuned NN model with only 5 neuron. 
#training time evaluated
start.time <- Sys.time()
nn3 <- neuralnet(V11 ~ V2+V3+V4+V6+V7+V8+V9+V10, data = training.set,hidden=5,threshold = 0.1,stepmax = 1e+7,act.fct = "logistic",linear.output=FALSE, err.fct = "ce",likelihood = TRUE,)
end.time<-Sys.time()
end.time-start.time

#Implementing the tuned NN model with only 3+2 neuron. 
#training time evaluated
start.time <- Sys.time()
nn4 <- neuralnet(V11 ~ V2+V3+V4+V6+V7+V8+V9+V10, data = training.set,hidden=c(3,2),threshold = 0.1,stepmax = 1e+7,act.fct = "logistic",linear.output=FALSE, err.fct = "ce",likelihood = TRUE,)
end.time<-Sys.time()
end.time-start.time

#Implementing the tuned NN model with only 5+3 neuron. 
#training time evaluated
start.time <- Sys.time()
nn5 <- neuralnet(V11 ~ V2+V3+V4+V6+V7+V8+V9+V10, data = training.set,hidden=c(5,3),threshold = 0.1,stepmax = 1e+7,act.fct = "logistic",linear.output=FALSE, err.fct = "ce",likelihood = TRUE,)
end.time<-Sys.time()
end.time-start.time
 
###########################
#PLotting and visualizing the neural networks
plot(nn1)
plot(nn2)
plot(nn3)
plot(nn4)
plot(nn5)

############################
#Test the resulting output
#creating a subset of the test set with the features (x-values) extracted
temp.test <- subset(test.set, select = c("V2","V3","V4","V6","V7","V8","V9","V10"))
#computing the classification probabilities for each class based on x values from the test set using the 4 built NN models
nn1.results <- compute(nn1, temp.test)
nn2.results <- compute(nn2, temp.test)
nn3.results <- compute(nn3, temp.test)
nn4.results <- compute(nn4, temp.test)
nn5.results <- compute(nn5, temp.test)

###########################
#1 neuron NN
#creating data frame comparing the predicted values by the model to the actual values.
results1 <- data.frame(prediction1 = nn1.results$net.result[,2],actual = test.set$V11)
#converting the probability values created by the model into classes of 1 and 0 setting the threshold probability for 1 at 0.5
pred1 <- ifelse(results1[,1]>0.5, 1, 0)
#converting the predicted classes from numerical to factors. 
pred1<-factor(pred1, levels=c(1,0), labels=c(1,0))
#creating confusion matrix and accuracy
cm1 = table(Predicted= pred1,Actual = test.set$V11)
err.1<-1-sum(diag(cm1))/sum(cm1)
err.1
acc.1<-(1-err.1)*100
###########################
#3 neuron NN
results2 <- data.frame(prediction2 = nn2.results$net.result[,2],actual = test.set$V11)
pred2 <- ifelse(results2[,1]>0.5, 1, 0)
pred2<-factor(pred2, levels=c(1,0), labels=c(1,0))
cm2 = table(Predicted= pred2,Actual = test.set$V11)
err.2<-1-sum(diag(cm2))/sum(cm2)
err.2
acc.2<-(1-err.2)*100
###########################
#5 neuron NN
results3 <- data.frame(prediction3 = nn3.results$net.result[,2],actual = test.set$V11)
pred3 <- ifelse(results3[,1]>0.5, 1, 0)
pred3<-factor(pred3, levels=c(1,0), labels=c(1,0))
cm3 = table(Predicted= pred3,Actual = test.set$V11)
err.3<-1-sum(diag(cm3))/sum(cm3)
err.3
acc.3<-(1-err.3)*100
###########################
#3+2 neuron NN
results4 <- data.frame(prediction4 = nn4.results$net.result[,2],actual = test.set$V11)
pred4 <- ifelse(results4[,1]>0.5, 1, 0)
pred4<-factor(pred4, levels=c(1,0), labels=c(1,0))
cm4 = table(Predicted= pred4,Actual = test.set$V11)
err.4<-1-sum(diag(cm4))/sum(cm4)
err.4
acc.4<-(1-err.4)*100
###########################
#5+3 neuron NN
results5 <- data.frame(prediction5 = nn5.results$net.result[,2],actual = test.set$V11)
pred5 <- ifelse(results5[,1]>0.5, 1, 0)
pred5<-factor(pred5, levels=c(1,0), labels=c(1,0))
cm5 = table(Predicted= pred5,Actual = test.set$V11)
err.5<-1-sum(diag(cm5))/sum(cm5)
err.5
acc.5<-(1-err.5)*100
###########################################################################
#Visualizing the model's accuracy 
counts <- c(signif(acc.1,digits = 4),signif(acc.2,digits = 4),signif(acc.3,digits=4),signif(acc.4,digits=4),signif(acc.5,digits = 4))
x<-barplot(counts, main="Accuracy of NN with different number of neurons and hidden layers",
           xlab="Types of NN with n Hidden Layers",ylab="Accuracy (%)",names.arg=c("Basic NN (1 Neuron)","3 Neurons","5 Neurons","3+2 Neurons","5+3 Neurons"),ylim = c( 0 , 100 ) )
y<-as.matrix(counts)
text(x,y+2,labels=as.character(y))

###########################################################################
#Calculating sensitivity
sen.1<-confusionMatrix(pred1,test.set$V11)
sen.2<-confusionMatrix(pred2, test.set$V11 )
sen.3<-confusionMatrix(pred3, test.set$V11 )
sen.4<-confusionMatrix(pred4, test.set$V11 )
sen.5<-confusionMatrix(pred5, test.set$V11 )

sen.1 <- (sen.1$byClass[1])*100
sen.2 <-(sen.2$byClass[1])*100
sen.3<-(sen.3$byClass[1])*100
sen.4 <-(sen.4$byClass[1])*100
sen.5<-(sen.5$byClass[1])*100

###########################################################################
#Visualizing model's sensitivity 
counts <- c(signif(sen.1,digits = 4),signif(sen.2,digits = 4),signif(sen.3,digits = 4),signif(sen.4,digits=4),signif(sen.5,digits=4))
x<-barplot(counts, main="Sensitivity of different types of ANN Classifiers with different number of neurons",
           xlab="Types of NN with n Hidden Layers",ylab="Sensitivity (%)",names.arg=c("Basic NN (1 Neuron)","3 Neurons","5 Neurons","3+2 Neurons","5+3 Neurons"),ylim = c( 0 , 100 ) )
y<-as.matrix(counts)
text(x,y+2,labels=as.character(y))
