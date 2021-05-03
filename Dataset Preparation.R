#NAME: Rupesh Kumar Dey
#TP: TP 061720
#TITLE: Data Preparation
################################################################
#Loading necessary packages:
library(TSEntropies) #Required library for ApEn and SampEn functions
library(pracma) #Required library for computing Hurst Exponent
library(moments) #Required library to compute skewness and kurtosis
library(ForeCA) #Required library for calculating spectral entropy
library(nonlinearTseries) #Required library for DFA
library(readr) #library to read the data
library(ggplot2) #imported for plotting and visualizing the data. 
library(dplyr)

#Reading the raw data obtained from UCI Machine Learning Repository. 
#Adjust directory accordingly
data <- read.table("D:/Rupesh/Documents/MASTERS IN AI/AML Module 2/BONN dataset/data_raw.csv", na.strings = "", sep=",")
filter <- data[c(2:11501),c(2:179)]

#checking if the raw data has any missing values and determining missing data location in the dataset
sum (is.na(filter)) 
pos<-which(is.na(filter), arr.ind=TRUE)

#Averaging the values for the particular missing rows and replacing them in the missing dataset point
mean117<-mean(as.numeric(filter[117,-10]))
mean91<-mean(as.numeric(filter[91,-11]))
mean105<-mean(as.numeric(filter[105,-18]))
mean173<-mean(as.numeric(filter[173,-18]))
mean147<-mean(as.numeric(filter[147,-25]))
mean176<-mean(as.numeric(filter[176,-25]))
mean168<-mean(as.numeric(filter[168,-36]))
mean183<-mean(as.numeric(filter[83,-47]))
mean159<-mean(as.numeric(filter[159,-58]))
mean162<-mean(as.numeric(filter[162,-64]))

filter[117,10]<-mean117
filter[91,11]<-mean91
filter[105,18]<-mean105
filter[173,18]<-mean173
filter[147,25]<-mean147
filter[176,25]<-mean176
filter[168,36]<-mean168
filter[183,47]<-mean183
filter[159,58]<-mean159
filter[162,64]<-mean162

#rechecking again if there's any missing values
sum (is.na(filter)) 
pos<-which(is.na(filter), arr.ind=TRUE)

#determining Apen
#For each row of the raw data the ApEn is calculated and saved into the variable Ap. 
for (row in 1:nrow(filter)){
  if (!exists("Ap")){
    Ap <- ApEn(filter[row,], dim = 2, lag = 1, r = 0.2 * sd(filter[row,]))
  }
  if (exists("Ap")){
    temp_Ap <-ApEn(filter[row,], dim = 2, lag = 1, r = 0.2 * sd(filter[row,]))
    Ap<-rbind(Ap, temp_Ap)
    rm(temp_Ap)
  }
}

#Determining SampEn
#For each row of the raw data the SampEn is calculated and saved into the variable Samp. 
for (row in 1:nrow(filter)){
  if (!exists("Samp")){
    Samp <- SampEn(filter[row,], dim = 2, lag = 1, r = 0.2 * sd(filter[row,]))
  }
  if (exists("Samp")){
    temp_Samp <-SampEn(filter[row,], dim = 2, lag = 1, r = 0.2 * sd(filter[row,]))
    Samp<-rbind(Samp, temp_Samp)
    rm(temp_Samp)
  }
}

#Calculating Hurst Exponent
for (row in 1:nrow(filter)){
  if (!exists("HE")){
    HE <- hurstexp(as.numeric(filter[row,]), d = 50, display = TRUE)
  }
  if (exists("HE")){
    temp_HE <-hurstexp(as.numeric(filter[row,]), d = 50, display = TRUE)
    HE<-rbind(HE, temp_HE)
    rm(temp_HE)
  }
}
#The hurstexp() function returns 5 different values of HE. Only the HE using R/S theory is extracted from the list of data for each row.
for (row in 1:nrow(HE)){
  if (!exists("Hrs")){
    Hrs <- HE[row,1]
  }
  if (exists("HE")){
    temp_Hrs <- HE[row,1]
    Hrs<-rbind(Hrs, temp_Hrs)
    rm(temp_Hrs)
  }
}

#Calculating skewness
for (row in 1:nrow(filter)){
  if (!exists("skew")){
    skew <- skewness(as.numeric(filter[row,]))
  }
  if (exists("skew")){
    temp_skew <-skewness(as.numeric(filter[row,]))
    skew<-rbind(skew, temp_skew)
    rm(temp_skew)
  }
}

#Calculating kurtosis
for (row in 1:nrow(filter)){
  if (!exists("kurt")){
    kurt <- kurtosis(as.numeric(filter[row,]))
  }
  if (exists("kurt")){
    temp_kurt <-kurtosis(as.numeric(filter[row,]))
    kurt<-rbind(kurt, temp_kurt)
    rm(temp_kurt)
  }
}

#Calculating Spectral Entropy
for (row in 1:nrow(filter)){
  if (!exists("spect")){
    spect <- spectral_entropy(as.numeric(filter[row,]))
  }
  if (exists("spect")){
    temp_spect <-spectral_entropy(as.numeric(filter[row,]))
    spect<-rbind(spect, temp_spect)
    rm(temp_spect)
  }
}

#Calculating mean
for (row in 1:nrow(filter)){
  if (!exists("mn")){
    mn <- mean(as.numeric(filter[row,]))
  }
  if (exists("mn")){
    temp_mn <-mean(as.numeric(filter[row,]))
    mn<-rbind(mn, temp_mn)
    rm(temp_mn)
  }
}

#Calculating variance
for (row in 1:nrow(filter)){
  if (!exists("variance")){
    variance <- var(as.numeric(filter[row,]))
  }
  if (exists("variance")){
    temp_variance <-var(as.numeric(filter[row,]))
    variance<-rbind(variance, temp_variance)
    rm(temp_var)
  }
}

#Calculating DFA 
for (row in 1:nrow(filter)){
  if (!exists("fluc")){
    fluc <- estimate(dfa(as.numeric(filter[row,]),  window.size.range = c(16, 200),  npoints = 45,  do.plot = FALSE),do.plot=FALSE)
  }
  if (exists("fluc")){
    temp_fluc <-estimate(dfa(as.numeric(filter[row,]),  window.size.range = c(16, 200),  npoints = 45,  do.plot = FALSE),do.plot=FALSE)
    fluc<-rbind(fluc, temp_fluc)
    rm(temp_fluc)
  }
}


#Cleaning the dataset to remove repeated rows due to data processing when computing features
#The data is saved as numerical. 
Ap<-as.numeric(Ap[2:11501,])
Samp<-as.numeric(Samp[2:11501,])
Hrs<-as.numeric(unlist(Hrs[3:11502,]))
kurt<-as.numeric(kurt[2:11501,])
skew<-as.numeric(skew[2:11501,])
mn<-as.numeric(mn[2:11501,])
variance<-as.numeric(variance[2:11501,])
spect<-as.numeric(spect[2:11501,])
fluc<-as.numeric(fluc[2:11501,])
yval<- as.numeric(data[c(2:11501),180])

#the final compiled data frame with all the features and yvalues
epi_data<-data.frame(Ap,Samp,Hrs,kurt,skew,mn,variance,spect,fluc,yval)

#Creating a temporary y-Value variable in factorized form of Yes and No (Epileptic vs non-epileptic) for data plotting and visualization
Epileptic<-factor(epi_data$yval, levels=c(1,2,3,4,5), labels=c("YES","YES","YES","NO","NO"))

# #Visualizing the dataset features with respect to each class:
# Only 3 features were considered which are samply entropy, Hurst exponent and variance
# 1 dimensional plot

#Others were coded as backup

# p1<-ggplot(epi_data, aes(x=1:11500,y=Ap))+geom_point(aes(color=Epileptic))+
# labs(title = "Data Visualization / Distribution Based on Approximate Entropy",x="Data Points",y="Approximate Entropy")
# 
 p2<-ggplot(epi_data, aes(x=1:11500,y=Samp))+geom_point(aes(color=Epileptic))+
   labs(title = "Data Visualization / Distribution Based on Sample Entropy",x="Data Points",y="Sample Entropy")
# 
 p3<-ggplot(epi_data, aes(x=1:11500,y=Hrs))+geom_point(aes(color=Epileptic))+
   labs(title = "Data Visualization / Distribution Based on Hurst Exponent",x="Data Points",y="Hurst Exponent")
# 
# p4<-ggplot(epi_data, aes(x=1:11500,y=kurt))+geom_point(aes(color=Epileptic))+
#   labs(title = "Data Visualization / Distribution Based on Kurtosis",x="Data Points",y="Kurtosis")
# 
# p5<-ggplot(epi_data, aes(x=1:11500,y=skew))+geom_point(aes(color=Epileptic))+
#   labs(title = "Data Visualization / Distribution Based on Skewness",x="Data Points",y="Skewness")
# 
# p6<-ggplot(epi_data, aes(x=1:11500,y=mn))+geom_point(aes(color=Epileptic))+
#   labs(title = "Data Visualization / Distribution Based on Mean",x="Data Points",y="Mean")
# 
 p7<-ggplot(epi_data, aes(x=1:11500,y=variance))+geom_point(aes(color=Epileptic))+
   labs(title = "Data Visualization / Distribution Based on Variance",x="Data Points",y="Variance")
# 
 p8<-ggplot(epi_data, aes(x=1:11500,y=spect))+geom_point(aes(color=Epileptic))+
   labs(title = "Data Visualization / Distribution Based on Spectral Entropy",x="Data Points",y="Spectral Entropy")
# 
# p9<-ggplot(epi_data, aes(x=1:11500,y=fluc))+geom_point(aes(color=Epileptic))+
#   labs(title = "Data Visualization / Distribution Based on Detrended Fluctuations Entropy",x="Data Points",y="Detrended Fluctuations")

############################################################
# creating a duplicate of the dataframe for data plotting purposes
t.plot<-data.frame(Ap,Samp,Hrs,kurt,skew,mn,variance,spect,fluc,Epileptic)
#determining features / parameters significance. 
analy<-aov(as.numeric(Epileptic)~.,data = t.plot)
summary(analy)

#calling scatterplot3d package for plotting data.3 dimensional plot 
library(scatterplot3d)
colors <- c("#E69F00", "#56B4E9")
colors <- colors[as.numeric(t.plot$Epileptic)]
scatterplot3d(t.plot[,c(3,6,7)], pch = 16, color=colors,xlab="Hurst Exponent",ylab = "Variance",zlab = "Spectral Entropy"):
  legend("bottom", legend = levels(Epileptic),
         col =  c("#E69F00", "#56B4E9"), 
         pch = c(16, 17, 18), 
         inset = -0.25, xpd = TRUE, horiz = TRUE,title="Epileptic")


#Plotting data visualization for 2D plot
ggplot(t.plot, aes(x=Hrs,y=variance))+geom_point(aes(color=Epileptic))+
     labs(title = "Data Visualization / Distribution of Variance against Hurst Exponent",x="Hurst Exponent",y="Variance")

ggplot(t.plot, aes(x=Hrs,y=spect))+geom_point(aes(color=Epileptic))+
  labs(title = "Data Visualization / Distribution of Spectral Entropy against Hurst Exponent",x="Hurst Exponent",y="Spectral Entropy")

ggplot(t.plot, aes(x=spect,y=variance))+geom_point(aes(color=Epileptic))+
  labs(title = "Data Visualization / Distribution of Variance against Spectral Entropy",x="Spectral Entropy",y="Variance")


#############
#Plotting barplots to visualize the value ranges of each feature for both class. 
plot(t.plot$Epileptic,t.plot$Ap,xlab= "Epileptic",col=rainbow(2),
     main = "Visualizing values of Approximate Entropy for each class")+
title(ylab="Approximate Entropy", line=2)

plot(t.plot$Epileptic,t.plot$Samp, xlab= "Epileptic",col=rainbow(2),
     main = "Visualizing values of Sample Entropy for each class")+
  title(ylab="Sample Entropy", line=2)


plot(t.plot$Epileptic,t.plot$Hrs,xlab= "Epileptic",col=rainbow(2),
     main = "Visualizing values of Hurst Exponent for each class")+
  title(ylab="Hurst Exponent", line=2)


plot(t.plot$Epileptic,t.plot$skew, xlab= "Epileptic",col=rainbow(2),
     main = "Visualizing values of Skewness for each class")+
  title(ylab="Skewness", line=2)


plot(t.plot$Epileptic,t.plot$mn, xlab= "Epileptic",col=rainbow(2),
     main = "Visualizing values of Mean for each class")+
  title(ylab="Mean", line=2)


plot(t.plot$Epileptic,t.plot$variance, xlab= "Epileptic",col=rainbow(2),
     main = "Visualizing values of Variance for each class")+
  title(ylab="Variance", line=2)


plot(t.plot$Epileptic,t.plot$spect, xlab= "Epileptic",col=rainbow(2),
     main = "Visualizing values of Spectral Entropy for each class")+
  title(ylab="Spectral Entropy", line=2)


plot(t.plot$Epileptic,t.plot$fluc, xlab= "Epileptic",col=rainbow(2),
     main = "Visualizing values of Detrended Fluctuation for each class")+
  title(ylab="Detrended Fluctuation", line=2)

plot(t.plot$Epileptic,t.plot$kurt, xlab= "Epileptic",col=rainbow(2),
     main = "Visualizing values of Kurtosis for each class")+
  title(ylab="Kurtosis", line=2)

###########################################################
#writing to the .csv file in the project directory with cleaned dataset. 
#Adjust directory accordingly
write.csv(epi_data,"D:/Rupesh/Documents/MASTERS IN AI/AML Module 2/BONN dataset/dataset_features.csv", row.names = TRUE)

