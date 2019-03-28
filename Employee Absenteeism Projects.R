#                   Employee Absenteeism

##########################       1. data exploration      ##########################


# clean the environment
rm(list=ls(all=T))

# set working Directory
setwd("C:/Users/Rohit/Desktop/Data science projects - Employee absentieesm 123/project in R")


# Read data
#install.packages("xlsx")
library(xlsx)
dataset=read.xlsx("Absenteeism_at_work_Project.xls" , sheetIndex = 1)

# see the dimension of data
dim(dataset)


# see the structure of data 
str(dataset)  # all the variable are in numeric, so we don't need to convert into
              # categorical variable into numeric.


# observe summary of the model
summary(dataset) # we can see there are some missing value, so we need to impute it

# let's see uniqe value
lapply(dataset, unique) # Only ID , Day of the week, Seasons doesn't contains any Missing value



# Before data exploration though Graph we need to Impute missing value

##########################    Missing Value Analysis  ##############################
 
# Create dataframe with missing value
Missing_val=data.frame(apply(dataset, 2, function(x){sum(is.na(x))}))

# convert row names into columns
Missing_val$columns = row.names(Missing_val)

# Rename the variable name
names(Missing_val)[1] = "missing_percentage"

# calculate percentage
Missing_val$missing_percentage = Missing_val$missing_percentage/nrow(dataset)*100

# Arrange in descending order
Missing_val= Missing_val[order(-Missing_val$missing_percentage),]

# Reset index
row.names(Missing_val) = NULL

# Rearrange the columns
Missing_val=Missing_val[,c(2,1)]


# Write output results back to the disk
write.xlsx(Missing_val,"Missing_per.xlsx",row.names = F)

###  Plot bar graph for missing values
# loading Library
library(ggplot2)
ggplot(data = Missing_val[1:18,], aes(x= reorder(columns,missing_percentage),
                               y=missing_percentage))+
  geom_bar(stat = 'identity', fill = "red")+
  xlab("parameters")+ ggtitle("Missing vlaue in Percentage (Employees Absenteeism)")
  


##  checking which imputation methods work well

# Actual data is 40
dataset[50,9]

# make it null
dataset[50,9]=NA

#### Now we will impute this NA with Mean and KNN just to check which one of the two works best
# Mean = 36.44429
# KNN = 39.23948

# Mean Method to impute Missing value 
#dataset$Age[is.na(dataset$Age)] = mean(dataset$Age, na.rm = T)


### KNN Imputation
# import library
#install.packages("DMwR")
library(DMwR)
dataset = knnImputation(dataset,k=5)

#           Here, My missing value was 40 
#                           KNN gives 39.23948
#                     and, Mean gives 36.44429, KNN gives more accurate value, so It wins..... 

# checking nulity once again just to be sure
data.frame(apply(dataset, 2, function(x){sum(is.na(x))}))


######################################################################################
########################    data Exploration through Visualization   ################

# Installing important  Packages for visualization
#install.packages("lattice")
library(lattice)

##### Plotting "Absenteeism.time.in.hours" with others variables


# ID
ggplot(data = dataset, aes(x=ID,y=Absenteeism.time.in.hours))+
  geom_bar(stat = 'identity', fill = "red")


# Month.of.absence
ggplot(data = dataset, aes(x=Month.of.absence,
       y=Absenteeism.time.in.hours))+
  geom_bar(stat = 'identity', fill = "navy blue")

# season
ggplot(data = dataset, aes(x=Seasons,y=Absenteeism.time.in.hours))+
  geom_bar(stat = 'identity', fill = "navy blue")

#Disciplinary.failure
ggplot(data = dataset, aes(x=Disciplinary.failure,y=Absenteeism.time.in.hours))+
  geom_bar(stat = 'identity', fill = "navy blue")





ggplot(data = dataset, aes(x=Pet, y=Absenteeism.time.in.hours))+
  geom_bar(stat = 'identity')



#################################################################################
##############################   Outlier Analysis ######################################



# seperate numeric columns

numeric_columns = c("Transportation.expense","Distance.from.Residence.to.Work",
                    "Service.time","Age","Work.load.Average.day.","Hit.target",
                    "Weight","Height","Body.mass.index")
numeric_columns


## Normality check before removing outliers
# Histogram using ggplot
# Instlal and load packages
#install.packages("gridExtra")
library(gridExtra)

for(i in 1:length(numeric_columns))
{
  assign(paste0("gn",i),ggplot(dataset,aes_string(x=numeric_columns[i]))+
           geom_histogram(fill="cornsilk",colour="black")+
           geom_density()+
           theme_bw()+
           labs(x=numeric_columns[i])+
           ggtitle(paste("Histogram of ",numeric_columns[i])))
}

gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,gn7,gn8,gn9,ncol=2)
                              
            

# All numeric variable are not normally distributed



### outlier Analysis
# Boxplot distribution and outlier check

for(i in 1:length(numeric_columns)){
  assign(paste0("gn",i),ggplot(dataset,aes_string(y=numeric_columns[i],x="Absenteeism.time.in.hours",
                                                      fill=dataset$Absenteeism.time.in.hours))+
           geom_boxplot(outlier.colour = "red",fill="skyblue",outlier.shape = 18,
                        outlier.size = 1,notch = F)+
           theme_bw()+
           labs(y=numeric_columns[i],x="Absenteeism.time.in.hours")+
           ggtitle(paste("Box Plot of Employees Absenteeism for",numeric_columns[i])))
}
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,gn7,gn8,gn9,ncol=2)



# our new numeric variable which contains outliers-
numeric_columns_outliers = c("Transportation.expense","Service.time","Age",
                             "Work.load.Average.day.","Hit.target","Height")


# Our sample data is less, So we will Impute these outlies with KNN Imputation Method

for(i in numeric_columns_outliers){
  print(i)
  val=dataset[,i][dataset[,i] %in% boxplot.stats(dataset[,i])$out]
  dataset[,i][dataset[,i] %in% val]= NA
}

dataset = knnImputation(dataset , k=5)


# Now we will check the outliers again just to be sure thay have imputed
for(i in 1:length(numeric_columns)){
  assign(paste0("gn",i),ggplot(dataset,aes_string(y=numeric_columns[i],x="Absenteeism.time.in.hours",
                                                  fill=dataset$Absenteeism.time.in.hours))+
           geom_boxplot(outlier.colour = "red",fill="skyblue",outlier.shape = 18,
                        outlier.size = 1,notch = F)+
           theme_bw()+
           labs(y=numeric_columns[i],x="Absenteeism.time.in.hours")+
           ggtitle(paste("Box Plot of Employees Absenteeism for",numeric_columns[i])))
}
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,gn7,gn8,gn9,ncol=2)

                  #  Outliers have been Removed......




#########################################################################################
##########################   Feature Selection   ####################################

# correlation Plot
#Install library
#install.packages("corrgram")
library(corrgram)
corrgram(dataset[,numeric_columns], order=F,
         lower.panel=panel.pie,text.panel=panel.txt,main="correlation plot")

           # Here from correlaion plot "Weight" and Body mass index are hoghly corrilated,
#            so we need to remove one out of them



## ANOVA test for Categprical variable
summary(aov(formula = Absenteeism.time.in.hours~ID,data = dataset))
summary(aov(formula = Absenteeism.time.in.hours~Reason.for.absence,data = dataset))
summary(aov(formula = Absenteeism.time.in.hours~Month.of.absence,data = dataset))
summary(aov(formula = Absenteeism.time.in.hours~Day.of.the.week,data = dataset))
summary(aov(formula = Absenteeism.time.in.hours~Seasons,data = dataset))
summary(aov(formula = Absenteeism.time.in.hours~Disciplinary.failure,data = dataset))
summary(aov(formula = Absenteeism.time.in.hours~Education,data = dataset))
summary(aov(formula = Absenteeism.time.in.hours~Social.drinker,data = dataset))
summary(aov(formula = Absenteeism.time.in.hours~Social.smoker,data = dataset))
summary(aov(formula = Absenteeism.time.in.hours~Son,data = dataset))
summary(aov(formula = Absenteeism.time.in.hours~Pet,data = dataset))


## Dimension Reduction - We will discard those variable whose p value > 0.05
dataset_deleted = subset(dataset, select = -c(Weight,ID, Month.of.absence, Seasons, Pet, Social.smoker))



#########################################################################################
##########################         Feature scaling       ###############################

## Normality check before removing outliers
# Histogram using ggplot
# Instlal and load packages
#install.packages("gridExtra")
library(gridExtra)

for(i in 1:length(numeric_columns))
{
  assign(paste0("gn",i),ggplot(dataset,aes_string(x=numeric_columns[i]))+
           geom_histogram(fill="cornsilk",colour="black")+
           geom_density()+
           theme_bw()+
           labs(x=numeric_columns[i])+
           ggtitle(paste("Histogram of ",numeric_columns[i])))
}

gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,ncol=2)

                 # WE can see no any numeric variable are normally distributed





##  Normalization for Non uniformly distributed features
numeric_columns_after_deletd = c("Transportation.expense","Distance.from.Residence.to.Work",
                                 "Service.time","Age","Work.load.Average.day.","Hit.target",
                                 "Height","Body.mass.index")

for(i in numeric_columns_after_deletd){
   print(i)
   dataset_deleted[,i]=(dataset_deleted[,i]-min(dataset_deleted[,i]))/
     (max(dataset_deleted[,i]-min(dataset_deleted[,i])))
}




# Creating dummy variables for categorical variables
#install.packages("mlr")
library(mlr)
#install.packages("dummies")
library(dummies)
categorical_columns= c( "Reason.for.absence","Day.of.the.week" ,
                        "Disciplinary.failure","Education","Son","Social.drinker" )

dataset_deleted_dummy = dummy.data.frame(dataset_deleted, categorical_columns)




#######################################################################################################
#################################     Machine Learning Model      #############################################



##################################  Multiple Linear Regression ###############################

# Load Library
#install.packages("MASS")
library(MASS)


# check Multicollinearity
#install.packages("usdm")
library(usdm)
vif(dataset_deleted[,-21])

# checking coliinearity
vifcor(dataset_deleted[,-21], th=0.9)
    # No variable from the 15 input variables has collinearity problem. 



# run regression model
LR_model = lm(Absenteeism.time.in.hours~., data=dataset_deleted_dummy)

summary(LR_model) # R-squared:  0.1442,	Adjusted R-squared:  0.1277 


## Dimension Reduction - Discard  "Transportation.expense" p>0.45
dataset_deleted = subset(dataset_deleted, select = -c(Transportation.expense))


#----------------------###################-------------------####################-------------------#


# Run regression model again to check p value , R-squared and Adjusted R-squared
LR_model = lm(Absenteeism.time.in.hours~., data=dataset_deleted)

summary(LR_model) # R-squared:  0.1436,	Adjusted R-squared:  0.1282


## Dimension Reduction - Discard  "Work.load.Average.day." p>0.34
dataset_deleted = subset(dataset_deleted, select = -c(Work.load.Average.day.))



#----------------------###################-------------------####################-------------------#



# Run regression model again to check p value , R-squared and Adjusted R-squared
LR_model = lm(Absenteeism.time.in.hours~., data=dataset_deleted)

summary(LR_model) #R-squared: 0.1425,	Adjusted R-squared:  0.1283


## Dimension Reduction - Discard  "Body.mass.index" p>0.21
#dataset_deleted = subset(dataset_deleted, select = -c(Body.mass.index))

# 
# # Run regression model again to check p value , R-squared and Adjusted R-squared
# LR_model = lm(Absenteeism.time.in.hours~., data=dataset_deleted)
# 
# summary(LR_model) # R-squared:  0.1407,	Adjusted R-squared:  0.1277



                   # when we remove "Body.mass.index" adjusted R-squared  decreased
                   # so i will stop to remove any variable






# Splitting the dataset into the Training set and Test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset_deleted_dummy$Absenteeism.time.in.hours, SplitRatio = 0.8)
training_set = subset(dataset_deleted_dummy, split == TRUE)
test_set = subset(dataset_deleted_dummy, split == FALSE)





# Run Multiple Linear Regression
LR_model = lm(Absenteeism.time.in.hours~., data=training_set)

# Summary of the model
summary(LR_model)


#Lets predict for training data
pred_LR_train = predict(LR_model, training_set[,-59])



#Lets predict for testing data
pred_LR_test = predict(LR_model, test_set[,-59])


###  Error Matrics
library(caret)
# For training data 
print(postResample(pred = pred_LR_train, obs = training_set[,59]))
#                   RMSE       Rsquared       MAE  
#                 0.9214076   0.1437901     0.4393489    #after doing standardization with target var
#                12.3453271   0.1437901     5.8865431    #after doing Normalization without target var
#                 0.10288293  0.14370300    0.04905541   #after doing Normalization with target var

# For testing data 
print(postResample(pred = pred_LR_test, obs = test_set[,13]))
#                     RMSE      Rsquared       MAE 
#                   0.9529124   0.1216392   0.4445753   #after doing standardization with target var
#                  12.7674386  0.121639 2   5.9565687   #after doing Normalization without target var
#                   0.10638327 0.12179758   0.04962822  #after doing Normalization with target var


#install.packages("DMwR")
library(DMwR)


regr.eval(test_set[,13], pred_LR_test, stats = c('mape',"mse"))
#                 mape        mse 
#             1.9241666   0.9080420   #after doing standardization with target var
#             Inf       163.007488    #after doing Normalization without target var    
#             Inf         0.01131740  #after doing Normalization with target var








####################################################################################################
########################################  Decision Tree  ###########################################


# Load Library
# install.packages("rpart")
# install.packages("MASS")
library(rpart)
library(MASS)

## rpart for regression
DT_model= rpart(Absenteeism.time.in.hours ~ ., data = training_set, method = "anova")

summary(DT_model)
plot(DT_model)
test(DT_model, pretty=0) 
 



#write rules into disk
write(capture.output(summary(DT_model)), "Rules.txt")


#Lets predict for training data
pred_DT_train = predict(DT_model, training_set[,names(test_set) != "Absenteeism.time.in.hours"])

#Lets predict for training data
pred_DT_test = predict(DT_model,test_set[,names(test_set) != "Absenteeism.time.in.hours"])


# For training data 
print(postResample(pred = pred_DT_train, obs = training_set[,59]))
             #   RMSE     Rsquared       MAE 
#              0.8157438  0.3289047    0.3734649   # after doing standardization with target var
#             10.9296081  0.3289047    5.0038078   #after doing Normalization without target var
#              0.09108002 0.32890501   0.04169836  #after doing Normalization with target var

# For testing data 
print(postResample(pred = pred_DT_test, obs = test_set[,59]))
#            RMSE      Rsquared       MAE 
#           0.9809011   0.1157153    0.4098303     #after doing standardization with target var
#           13.1424404  0.1157153    5.4910435     #after doing Normalization without target var
#           0.10952034  0.11571537   0.04575868    #after doing Normalization with target var
 
#install.packages("DMwR")
library(DMwR)

regr.eval(test_set[,59], pred_DT_test, stats = c('mape',"mse"))
#                   mape        mse 
#                 1.7483151   0.9621669    # after doing standardization with target var
#                  Inf      172.723740     #after doing Normalization without target var
#                  Inf        0.0119947    #after doing Normalization with target var



###########################################################################################################
#-----------------------------------------------  Random Forest ---------------------------------------------

# Fitting Random Forest Regression to the dataset
# install.packages('randomForest')
library(randomForest)
set.seed(1234)
RF_model= randomForest(x = training_set[,-59],
                         y = training_set[,59],
                         ntree = 500)



#Lets predict for training data
pred_RF_train = predict(RF_model, training_set[,-59])

#Lets predict for testing data
pred_RF_test = predict(RF_model, test_set[,-59])



# For training data 
print(postResample(pred = pred_RF_train, obs = training_set[,59]))
#                      RMSE        Rsquared       MAE   
#                    0.5240608    0.7989361    0.2252558     # after doing standardization with target var
#                    7.1883783    0.7850687    3.1048255     #after doing Normalization without target var
#                    0.06027517   0.78762025   0.02559567    #after doing Normalization with target var

# For testing data 
print(postResample(pred = pred_RF_test, obs = test_set[,59]))
#                      RMSE      Rsquared        MAE 
#                      0.9124303   0.1945163   0.3981753   # after doing standardization with target var
#                     12.3568438   0.1814765   5.4416842   #after doing Normalization without target var
#                      0.09982425  0.22453252  0.04399986  #after doing Normalization with target var
 

#install.packages("DMwR")
library(DMwR)

regr.eval(test_set[,13], pred_RF_test, stats = c('mape',"mse"))
#                 mape         mse 
#               1.6789037    0.8325290     # after doing standardization with target var
#                 Inf      152.691587      #after doing Normalization without target var
#                 Inf        0.00996488    #after doing Normalization with target var

#################################################################################################
# ------------------------------ Support Vector Regression --------------------------------------#

# Fitting SVR to the dataset
#install.packages('e1071')
library(e1071)
SVR_model = svm(formula = Absenteeism.time.in.hours ~ .,
                data = training_set,
                type = 'eps-regression',
                kernel = 'radial')



#Lets predict for training data
pred_SVR_train = predict(SVR_model, training_set[,-59])

#Lets predict for testing data
pred_SVR_test = predict(SVR_model, test_set[,-59])


### Error Matrics
# For training data 
print(postResample(pred = pred_SVR_train, obs = training_set[,59]))
#                   RMSE        Rsquared       MAE    
#                  0.9351332   0.2012483    0.3017589    # after doing standardization with target var
#                 12.529193    0.201238     4.043109     #after doing Normalization without target var
#                 0.10088912   0.38665837   0.03102347   #after doing Normalization with target var 

# For testing data 
print(postResample(pred = pred_SVR_test, obs = test_set[,59]))
#                  RMSE        Rsquared       MAE 
#                 0.9036999   0.3865451    0.2779306    # after doing standardization with target var
#                12.1075765   0.3866099    3.7232733    #after doing Normalization without target var
#                 0.10088912  0.38665837   0.03102347   #after doing Normalization with target var



#install.packages("DMwR")
library(DMwR)

regr.eval(test_set[,59], pred_SVR_test, stats = c('mape',"mse"))
#             mape        mse 
#           0.7723146   0.8166736       # after doing standardization with target var
#           Inf       146.593408        #after doing Normalization without target var
#           Inf         0.01017861      #after doing Normalization with target var







###########----------------#################-----------------####################------------------

#   AS WE HAVE SEEN THAT WE ARE NOT GETTING ERROR METRICS UT TO THE MARK,
#   THEREFORE, WE WILL USE PRINCIPAL COMPONENT ANALYSIS TO DISCARD IRRELEVANT VARIABLES,
#  AND WE WILL CHECK THE ERROR MATRICS ONE BY ONE IN EACH MODEL........
#  LET'S GO............

##########-----------------##################------------------#################---------------------




##########################################################################################################
#- - - - - - - - - - - - - - - - - - - -  Principal COmponent Analysis - - - - - - - - - - - - - - - - - -



#----------------------Dimensionality Reduction using PCA-------------------------------#


#principal component analysis
prin_comp = prcomp(training_set)

#compute standard deviation of each principal component
std_dev = prin_comp$sdev

#compute variance
pr_var = std_dev^2

#proportion of variance explained
prop_varex = pr_var/sum(pr_var)

#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

#add a training set with principal components
training_set_pca = data.frame(Absenteeism.time.in.hours = training_set$Absenteeism.time.in.hours, prin_comp$x)

# From the above plot selecting 30 components since it explains almost 95+ % data variance
training_set_pca =training_set_pca[,1:30]


 
#transform test into PCA
test_set_pca = predict(prin_comp, newdata = test_set)
test_set_pca = as.data.frame(test_set_pca)


#select the first 30 components
test_set_pca=test_set_pca[,1:30]





#########################################################################################################################
#-----------------------------   Evaluating all the model again after applying PCA  ----------------------



#-----------------------------  Multiple Linear Regression -----------------------------------------


# Run Multiple Linear Regression
LR_model = lm(Absenteeism.time.in.hours~., data=training_set_pca)

# Summary of the model
summary(LR_model)


#Lets predict for training data
pred_LR_train = predict(LR_model, training_set_pca)



#Lets predict for testing data
pred_LR_test = predict(LR_model, test_set_pca)


###  Error Matrics
library(caret)
# For training data 
print(postResample(pred = pred_LR_train, obs = training_set$Absenteeism.time.in.hours))
#                   RMSE       Rsquared       MAE  
#                 0.9214076    0.1437901     0.4393489      #after doing standardization with target var
#                12.3453271    0.1437901     5.8865431      #after doing Normalization without target var
#                 0.10288293   0.14370300    0.04905541     #after doing Normalization with target var
#                 1.292606e-05 1.000000e+00  3.526994e-06   # pc = 45
#                 0.0002390449 0.9999999997  0.0001243538   # pc=30
#                 0.0003255971 0.9999999994  0.0002113523   # pc=25
#                 0.0004725597 0.9999999987  0.0003663976   # pc=20



# For testing data 
print(postResample(pred = pred_LR_test, obs = test_set$Absenteeism.time.in.hours))
#                     RMSE      Rsquared       MAE 
#                   0.9529124   0.1216392     0.4445753    #after doing standardization with target var
#                  12.7674386   0.121639 2    5.9565687    #after doing Normalization without target var
#                   0.10638327  0.12179758    0.04962822   #after doing Normalization with target var
#                  2.004035e-05 1.000000e+00  6.058988e-06 # pc=45
#                  0.0002935940 0.9999999995  0.0001368412 # pc=30
#                  0.0004117391 0.9999999991  0.0002782557 # pc=25
#                  0.0005550411 0.9999999983  0.0004310795 # pc=20


#install.packages("DMwR")
library(DMwR)


regr.eval(test_set$Absenteeism.time.in.hours, pred_LR_test, stats = c('mape',"mse"))
#                 mape        mse 
#             1.9241666   0.9080420     #after doing standardization with target var
#             Inf       163.007488      #after doing Normalization without target var    
#             Inf         0.01131740    #after doing Normalization with target var
#             Inf         4.016157e-10  # pc=45
#             Inf         8.619741e-08  # pc=30
#             Inf         1.695291e-07  # pc=25
#             Inf         3.080706e-07  # pc=20




#########################################################################################################################

# _-_  WE WILL MAKE COMMENT ALL OTHER MODEL BECAUSE MULTIPLE LINEAR REGRESSION IS GIVING BETTER ACCURAY, RMSE AND R-SQU.

#########################################################################################################################




#--------------------------------   Decision Tree  ----------------------------------------------------
  
#    
#    
#    # Load Library
#    # install.packages("rpart")
#    # install.packages("MASS")
#    library(rpart)
#    library(MASS)
#    
#    ## rpart for regression
#    DT_model= rpart(Absenteeism.time.in.hours ~ ., data = training_set_pca, method = "anova")
#    
#    summary(DT_model)
#    plot(DT_model)
#    #test(DT_model, pretty=0) 
#    
#    
#    
#    
#    #write rules into disk
#    write(capture.output(summary(DT_model)), "Rules.txt")
#    
#    
#    #Lets predict for training data
#    pred_DT_train = predict(DT_model, training_set_pca)
#    
#    #Lets predict for training data
#    pred_DT_test = predict(DT_model,test_set_pca)
#    
#    
#    ###  Error Matrics
#    library(caret)
#    # For training data 
#    print(postResample(pred = pred_DT_train, obs = training_set$Absenteeism.time.in.hours))
#    #   RMSE     Rsquared       MAE 
#    #              0.8157438  0.3289047    0.3734649   # after doing standardization with target var
#    #             10.9296081  0.3289047    5.0038078   #after doing Normalization without target var
#    #              0.09108002 0.32890501   0.04169836  #after doing Normalization with target var
#    #             5.5615277   0.8262346    1.7624425   # pc= 45
#    #             5.5615277   0.8262346    1.7624425   # pc=30
#    #             5.5615277 0.8262346      1.7624425   # pc=25
#    #             5.5615277 0.8262346      1.7624425   # pc=20
#    
#    
#    # For testing data 
#    print(postResample(pred = pred_DT_test, obs = test_set$Absenteeism.time.in.hours))
#    #            RMSE      Rsquared       MAE 
#    #           0.9809011   0.1157153    0.4098303     #after doing standardization with target var
#    #           13.1424404  0.1157153    5.4910435     #after doing Normalization without target var
#    #           0.10952034  0.11571537   0.04575868    #after doing Normalization with target var
#    #           5.2662989   0.8566885    1.5566350     # pc = 45
#    #           5.2662989   0.8566885    1.5566350     # pc= 30
#    #           5.2662989   0.8566885    1.5566350     # pc= 25
#    #           5.2662989   0.8566885    1.5566350     # pc= 20
#    
#    
#    
#    #install.packages("DMwR")
#    library(DMwR)
#    
#    regr.eval(test_set$Absenteeism.time.in.hours, pred_DT_test, stats = c('mape',"mse"))
#    #                   mape        mse 
#    #                 1.7483151   0.9621669      # after doing standardization with target var
#    #                  Inf      172.723740       #after doing Normalization without target var
#    #                  Inf        0.0119947      #after doing Normalization with target var
#    #                  Inf       27.7339         # pc = 45
#    #                  Inf       27.7339         # pc=30
#    #                  Inf       27.7339         # pc=25
#    #                  Inf       27.7339         # pc=20
#    
#    #-------------------------------------------  Random Forest ---------------------------------------------
#    
#    # Fitting Random Forest Regression to the dataset
#    # install.packages('randomForest')
#    library(randomForest)
#    set.seed(1234)
#    RF_model= randomForest(y = training_set_pca$Absenteeism.time.in.hours,
#                           x = training_set_pca,
#                           ntree = 500)
#    
#    
#    
#    #Lets predict for training data
#    pred_RF_train = predict(RF_model, training_set_pca)
#    
#    #Lets predict for testing data
#   pred_RF_test = predict(RF_model,test_set_pca)
#    
#    
#    
#    # For training data 
#    print(postResample(pred = pred_RF_train, obs = training_set$Absenteeism.time.in.hours))
#    #                      RMSE        Rsquared       MAE   
#    #                    0.5240608    0.7989361    0.2252558     # after doing standardization with target var
#    #                    7.1883783    0.7850687    3.1048255     # after doing Normalization without target var
#    #                    0.06027517   0.78762025   0.02559567    # after doing Normalization with target var
#    #                    1.4203183    0.9934495 0.3168067        # pc = 45
#    
#    
#    # For testing data 
#    print(postResample(pred = pred_RF_test, obs = test_set$Absenteeism.time.in.hours))
#    #                      RMSE      Rsquared        MAE 
#    #                      0.9124303   0.1945163   0.3981753   # after doing standardization with target var
#    #                     12.3568438   0.1814765   5.4416842   #after doing Normalization without target var
#    #                      0.09982425  0.22453252  0.04399986  #after doing Normalization with target var
#    #                      0.10507904  0.14538501 s 0.04798118 # pc = 10
#    
#    #install.packages("DMwR")
#    library(DMwR)
#    
#    regr.eval(test_set[,9], pred_RF_test, stats = c('mape',"mse"))
#    #                 mape         mse 
#    #               1.6789037    0.8325290     # after doing standardization with target var
#    #                 Inf      152.691587      #after doing Normalization without target var
#    #                 Inf        0.00996488    #after doing Normalization with target var
#    #                 Inf        0.0110416     # pc 
#    
#    
#    #################################################################################################
#    # ------------------------------ Support Vector Regression --------------------------------------#
#    
#    # Fitting SVR to the dataset
#    #install.packages('e1071')
#    library(e1071)
#    SVR_model = svm(formula = Absenteeism.time.in.hours ~ .,
#                    data = training_set_pca,
#                    type = 'eps-regression',
#                    kernel = 'radial')
#    
#    
#    
#    #Lets predict for training data
#    pred_SVR_train = predict(SVR_model, training_set_pca)
#    
#    #Lets predict for testing data
#    pred_SVR_test = predict(SVR_model, test_set_pca)
#    
#    
#    ### Error Matrics
#    # For training data 
#    print(postResample(pred = pred_SVR_train, obs = training_set$Absenteeism.time.in.hours))
#    #                   RMSE        Rsquared       MAE    
#    #                  0.9351332   0.2012483    0.3017589    # after doing standardization with target var
#    #                 12.529193    0.201238     4.043109     #after doing Normalization without target var
#    #                  0.10088912  0.38665837   0.03102347   #after doing Normalization with target var 
#    #                  7.5104335   0.7601637    1.7502205    # pc=45
#    #                  5.4500204   0.9164919    1.5886365    # pc=30
#    #                  5.5022943   0.9021344    1.5425403    # pc=25
#    #                  5.7796339   0.8857643    1.5385141    # pc=20
#    
#    
#    # For testing data 
#    print(postResample(pred = pred_SVR_test, obs = test_set$Absenteeism.time.in.hours))
#    #                  RMSE        Rsquared       MAE 
#    #                 0.9036999   0.3865451    0.2779306    # after doing standardization with target var
#    #                12.1075765   0.3866099    3.7232733    #after doing Normalization without target var
#    #                 0.10088912  0.38665837   0.03102347   #after doing Normalization with target var
#    #                 9.512273    0.691653     3.030982     # pc=45
#    #                 8.6579865   0.7318553    2.6280033    # pc=30        
#    #                 8.4668141   0.7603665    2.5056577    # pc=25
#    #                 8.4502098   0.7344911    2.2842206    # pc=20
#    
#    #install.packages("DMwR")
#    library(DMwR)
#    
#    regr.eval(test_set$Absenteeism.time.in.hours, pred_SVR_test, stats = c('mape',"mse"))
#    #             mape        mse 
#    #           0.7723146   0.8166736       # after doing standardization with target var
#    #           Inf       146.593408        #after doing Normalization without target var
#    #           Inf         0.01017861      #after doing Normalization with target var
#    #           Inf        90.48335         # pc= 45
#    #           Inf        74.96073         # pc = 30
#    #           Inf        71.68694         # pc=25
#    #           Inf        71.40605         # pc=20






#######################################################################################################################
# ---------------------------------------------  Calculating loss for every month  ---------------------------------------


df=dataset
list = c("Work.load.Average.day.","Hit.target","Absenteeism.time.in.hours")
for (i in list){
  df[,i] = (df[,i] - min(df[,i]))/((max(df[,i]))- (min(df[,i])))
}

a = aggregate(df$Work.load.Average.day., by = list(Month.of.absence=df$Month.of.absence), FUN=sum)
b = aggregate(df$Hit.target, by = list(Month.of.absence=df$Month.of.absence), FUN=sum)
c = aggregate(df$Absenteeism.time.in.hours, by = list(Month.of.absence=df$Month.of.absence), FUN=sum)
d = aggregate(df$ID, by = list(Month.of.absence=df$Month.of.absence), FUN=unique)

# Assumption is every month has 30 days

# we are comparing the wirkload (in hours) for every month  ( except output) and comparing 
# it with the Hit target.
# Also, we are subracting the Absenteeism time in hours to get the loss


for (i in c(1,2,3,4,5,6,7,8,9,10,11,12)){
  print(i)
  # calculating Loss
  loss = (b[2][1,]-(b[2][i,]/(a[2][i,]*131*length(d[2][i,][[1]])))*((a[2][i,]*30*length(d[2][i,][[1]]))-c[2][i,]))/b[2][i,]*100
  
  print(loss)
}

loss = as.data.frame(loss)

# output:              Loss for every month in hours
#    
#    1] 1             # january
#    [1] 77.09924
#    [1] 2            # February
#    [1] -17.82773
#    [1] 3            # March
#    [1] -19.63254
#    [1] 4            # April
#    [1] -20.24614
#    [1] 5            # May
#    [1] -16.93994
#    [1] 6            # June
#    [1] -19.23701
#    [1] 7            # July
#    [1] -17.37784
#    [1] 8            # Aug
#    [1] -18.43026
#    [1] 9            # september
#    [1] -15.41679
#    [1] 10           # october
#    [1] -12.58831
#    [1] 11           # November
#    [1] -12.46213
#    [1] 12           # december
#    [1] 377.1544


###################################################  THANK YOU #################################################
