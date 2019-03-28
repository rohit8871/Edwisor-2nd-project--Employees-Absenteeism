#!/usr/bin/env python
# coding: utf-8

# # Employee Absenteeism
# # 1. Data preprocessing

# In[2]:


# Importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[12]:


# set working directory
os.chdir("C:/Users/Rohit/Desktop/Data science projects - Employee absentieesm 123")


# In[13]:


# Importing the datasets
dataset= pd.read_excel('Absenteeism_at_work_Project.xls')


# In[ ]:


#dataset.head(5)


# In[ ]:


dataset.describe()


# In[ ]:


#dataset['Transportation expense'].value_counts()


# In[ ]:


dataset.index


# # 1.1 Data Exploration with Visualization

# In[4]:


# Import Libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


dataset.columns


# ## Relation of 'Absenteeism time in hour' with others
# 

# #### "ID"  with respect to  'Absenteeism time in hour'

# In[10]:


plt.figure(figsize=(15,8))

sns.countplot(x='ID',data=dataset)


# In[11]:


plt.figure(figsize=(15,8))

sns.barplot(x='ID', y='Absenteeism time in hours',data=dataset,ci=9)


# #### 'Reason for absence'    With respect to    'Absenteeism time in hour'

# In[12]:


plt.figure(figsize=(15,8))

sns.countplot(x='Reason for absence',data=dataset)


# In[13]:


plt.figure(figsize=(15,8))

sns.barplot(x='Reason for absence', y='Absenteeism time in hours',data=dataset)


# In[14]:


# strip plot for Column 'Day of the Week'
plt.figure(figsize=(15,8))

sns.stripplot(x='Reason for absence', y='Absenteeism time in hours',data=dataset,jitter=0.0)


# In[15]:


plt.figure(figsize=(15,8))

sns.boxplot(x='Reason for absence', y='Absenteeism time in hours',data=dataset)


# In[16]:


dataset.columns


# ### 'Month of absence'    With respect to   'Absenteeism time in hours'

# In[17]:


# Total month should be 12
dataset['Month of absence'].nunique()


# In[18]:


plt.figure(figsize=(15,8))
sns.countplot(x='Month of absence',data=dataset)


# In[19]:


plt.figure(figsize=(15,8))
sns.barplot(x='Month of absence', y='Absenteeism time in hours',data=dataset,ci=10)


# In[20]:


# strip plot for Column 'Day of the Week'
plt.figure(figsize=(15,8))
sns.stripplot(x='Month of absence', y='Absenteeism time in hours',data=dataset,jitter=0.2)


# In[21]:


plt.figure(figsize=(15,8))
sns.boxplot(x='Month of absence', y='Absenteeism time in hours',data=dataset)


# ### 'Day of the week' With respect to 'Absenteeism time in hours'

# In[22]:


sns.countplot(x='Day of the week',data=dataset)


# In[23]:


dataset['Day of the week'].value_counts()


# In[24]:


sns.barplot(x='Day of the week', y='Absenteeism time in hours',data=dataset,ci=10)


# In[25]:



sns.stripplot(x='Day of the week', y='Absenteeism time in hours',data=dataset,jitter=0.2)


# In[26]:


sns.boxplot(x='Day of the week', y='Absenteeism time in hours',data=dataset)


# In[27]:


dataset.columns


# ### 'Seasons'  With respect to  'Absenteeism time in hours'

# In[28]:


sns.countplot(x='Seasons',data=dataset)


# In[29]:


dataset['Seasons'].value_counts()


# In[30]:


sns.barplot(x='Seasons', y='Absenteeism time in hours',data=dataset,ci=10)


# In[31]:


sns.stripplot(x='Seasons', y='Absenteeism time in hours',data=dataset,jitter=0.2)


# In[32]:


sns.boxplot(x='Seasons', y='Absenteeism time in hours',data=dataset)


# In[33]:


dataset.columns


# ### 'Transportation expense'  With respect to   'Absenteeism time in hours'

# In[34]:


plt.figure(figsize=(15,8))
sns.countplot(x='Transportation expense',data=dataset)


# In[35]:


dataset['Transportation expense'].value_counts().head()


# In[36]:


plt.figure(figsize=(15,8))
sns.barplot(x='Transportation expense', y='Absenteeism time in hours',data=dataset)


# In[37]:


plt.figure(figsize=(15,8))
sns.stripplot(x='Transportation expense', y='Absenteeism time in hours',data=dataset,jitter=0.2)


# ### 'Distance from Residence to Work'  With respect to   'Absenteeism time in hours'

# In[38]:


plt.figure(figsize=(15,8))
sns.countplot(x='Distance from Residence to Work',data=dataset)


# In[39]:


dataset['Distance from Residence to Work'].value_counts()


# In[40]:


plt.figure(figsize=(15,8))
sns.barplot(x='Distance from Residence to Work', y='Absenteeism time in hours',data=dataset)


# In[41]:


plt.figure(figsize=(15,8))
sns.stripplot(x='Distance from Residence to Work', y='Absenteeism time in hours',data=dataset,jitter=0.2)


# In[42]:


plt.figure(figsize=(15,8))
sns.boxplot(x='Distance from Residence to Work', y='Absenteeism time in hours',data=dataset)


#  
# 
# ### 'Service time'       With respect to    "'Absenteeism time in hours'"

# In[43]:


plt.figure(figsize=(15,8))
sns.countplot(x='Service time',data=dataset)


# In[44]:


dataset['Service time'].value_counts()


# In[45]:


plt.figure(figsize=(15,8))
sns.barplot(x='Service time', y='Absenteeism time in hours',data=dataset)


# In[46]:


plt.figure(figsize=(15,8))
sns.stripplot(x='Service time', y='Absenteeism time in hours',data=dataset,jitter=0.3)


# In[47]:


plt.figure(figsize=(15,8))
sns.boxplot(x='Service time', y='Absenteeism time in hours',data=dataset)


#  ### " Age "   With respect to   'Absenteeism time in hours'

# In[48]:


plt.figure(figsize=(15,8))
sns.countplot(x='Age',data=dataset)


# In[49]:


dataset['Age'].value_counts().head()


# In[50]:


plt.figure(figsize=(15,6.5))
sns.barplot(x='Age', y='Absenteeism time in hours',data=dataset)


# In[51]:


plt.figure(figsize=(15,6.5))
sns.stripplot(x='Age', y='Absenteeism time in hours',data=dataset,jitter=0.3)


# In[52]:


plt.figure(figsize=(15,6.5))
sns.boxplot(x='Age', y='Absenteeism time in hours',data=dataset)


# ### 'Hit target'    With respect to    'Absenteeism time in hours'

# In[53]:


plt.figure(figsize=(15,6))
sns.countplot(x='Hit target',data=dataset)


# In[54]:


dataset['Hit target'].value_counts()


# In[55]:


plt.figure(figsize=(15,6))
sns.barplot(x='Hit target', y='Absenteeism time in hours',data=dataset,ci=10)


# In[56]:


plt.figure(figsize=(15,6))
sns.stripplot(x='Hit target', y='Absenteeism time in hours',data=dataset,jitter=0.3)


# In[57]:


plt.figure(figsize=(15,6))
sns.boxplot(x='Hit target', y='Absenteeism time in hours',data=dataset)


# In[58]:


dataset['Hit target'].unique()


# In[59]:


dataset.columns


# ### 'Disciplinary failure'  With respect to   'Absenteeism time in hours'

# In[60]:


dataset['Disciplinary failure'].value_counts()


# In[61]:


sns.barplot(x='Disciplinary failure', y='Absenteeism time in hours',data=dataset)


# In[62]:


sns.stripplot(x='Disciplinary failure', y='Absenteeism time in hours',data=dataset,jitter=0.2)


# In[63]:


sns.boxplot(data=dataset,x='Disciplinary failure',y='Absenteeism time in hours')


# ### 'Education'  With respect to  'Absenteeism time in hours'

# In[64]:


sns.countplot(x='Education',data=dataset)


# In[65]:


dataset['Education'].value_counts()


# In[66]:


sns.barplot(x='Education', y='Absenteeism time in hours',data=dataset)


# In[67]:


sns.stripplot(x='Education', y='Absenteeism time in hours',data=dataset,jitter=0.2)


# In[68]:


sns.boxplot(data=dataset,x='Education',y='Absenteeism time in hours')


#  ### 'Son'   With respect to   'Absenteeism time in hours'

# In[69]:


sns.countplot(x='Son',data=dataset)


# In[70]:


sns.barplot(x='Son', y='Absenteeism time in hours',data=dataset)


# In[71]:


sns.stripplot(x='Son', y='Absenteeism time in hours',data=dataset,jitter=0.2)


# In[72]:


sns.boxplot(data=dataset,x='Son',y='Absenteeism time in hours')


# ### 'Social drinker'   With respect to    'Absenteeism time in hours'

# In[73]:


sns.countplot(x='Social drinker',data=dataset)


# In[74]:


sns.barplot(x='Social drinker', y='Absenteeism time in hours',data=dataset)


# In[75]:


sns.stripplot(x='Social drinker', y='Absenteeism time in hours',data=dataset,jitter=0.2)


# In[76]:


sns.boxplot(data=dataset,x='Social drinker',y='Absenteeism time in hours')


# ### 'Social smoker'    With respect to    'Absenteeism time in hours'

# In[77]:


sns.countplot(x='Social smoker',data=dataset)


# In[78]:


dataset['Social smoker'].value_counts()


# In[79]:


sns.barplot(x='Social smoker', y='Absenteeism time in hours',data=dataset)


# In[80]:


sns.stripplot(x='Social smoker', y='Absenteeism time in hours',data=dataset,jitter=0.2)


# In[81]:


sns.boxplot(data=dataset,x='Social smoker',y='Absenteeism time in hours')


# ### 'Pet'  With respect to   'Absenteeism time in hours'

# In[82]:


sns.countplot(x='Pet',data=dataset)


# In[83]:


sns.barplot(x='Pet', y='Absenteeism time in hours',data=dataset)


# In[84]:


sns.stripplot(x='Pet', y='Absenteeism time in hours',data=dataset,jitter=0.3)


# In[85]:


sns.boxplot(data=dataset,x='Pet',y='Absenteeism time in hours')


# ### 'Weight'   With respect to   'Absenteeism time in hours'

# In[86]:


plt.figure(figsize=(15,6))
sns.countplot(x='Weight',data=dataset)


# In[87]:


plt.figure(figsize=(15,6))
sns.barplot(x='Weight', y='Absenteeism time in hours',data=dataset)


# In[88]:


plt.figure(figsize=(15,6))
sns.stripplot(x='Weight', y='Absenteeism time in hours',data=dataset,jitter=0.3)


# In[89]:


plt.figure(figsize=(15,6))
sns.boxplot(data=dataset,x='Weight',y='Absenteeism time in hours')


# ### 'Height'   With respect to  'Absenteeism time in hours'

# In[90]:


plt.figure(figsize=(15,6))
sns.countplot(x='Height',data=dataset)


# In[91]:


plt.figure(figsize=(15,6))
sns.barplot(x='Height', y='Absenteeism time in hours',data=dataset)


# In[92]:


plt.figure(figsize=(15,6))
sns.stripplot(x='Height', y='Absenteeism time in hours',data=dataset,jitter=0.3)


# In[93]:


plt.figure(figsize=(15,6))

sns.boxplot(data=dataset,x='Height',y='Absenteeism time in hours')


# ### 'Body mass index' With respect to   'Absenteeism time in hours'

# In[94]:


plt.figure(figsize=(15,6))
sns.countplot(x='Body mass index',data=dataset)


# In[95]:


plt.figure(figsize=(15,6))
sns.barplot(x='Body mass index', y='Absenteeism time in hours',data=dataset)


# In[96]:


plt.figure(figsize=(15,6))
sns.stripplot(x='Body mass index', y='Absenteeism time in hours',data=dataset,jitter=0.3)


# In[97]:


plt.figure(figsize=(15,6))
sns.boxplot(data=dataset,x='Body mass index',y='Absenteeism time in hours')


# In[98]:


dataset.head()


# In[99]:


dataset.columns


# # 1.2 Missing Value Analysis

# In[100]:


# Importing the datasets again
dataset= pd.read_excel('Absenteeism_at_work_Project.xls')
dataset.shape


# In[101]:


# create data frame with missing value
missing_val = pd.DataFrame(dataset.isnull().sum())
missing_val


# In[102]:


# Reset Index
missing_val=missing_val.reset_index()
missing_val


# In[103]:


#checking missing values after Imputation
plt.figure(figsize=(8,5))
sns.heatmap(dataset.isnull(), cmap="viridis")
print("Total NA value in the dataset is =" +str(dataset.isnull().sum().sum()))


# ##### Here we can observe yellow das above Heat Map  which purely indicates about missing values

#     

# In[104]:


# Rename variable
missing_val=missing_val.rename(columns = {'index':'variables',0:'missing_percentage'})
missing_val


# In[105]:


#  Missing Percentage
missing_val['missing_percentage']=(missing_val['missing_percentage']/len(dataset))*100
missing_val


# In[106]:


# descending order
missing_val=missing_val.sort_values('missing_percentage', ascending = False).reset_index(drop=True)
missing_val


# #### we can see above Table except last 3 variables all contains NA values.
# #### we need to impute this missing values by Missing Value Imputation Method

# In[107]:


# Importing the datasets again
dataset= pd.read_excel('Absenteeism_at_work_Project.xls')


# ###### Imputer Library

# In[108]:


# # Importing the datasets
# dataset= pd.read_csv('Data.csv')
# X=dataset.iloc[:,:-1].values
# y=dataset.iloc[:,3].values


# In[109]:


# # Taking care of missing values
# from sklearn.preprocessing import Imputer
# imputer= Imputer(missing_values="NaN", strategy ="mean", axis=0)
# imputer=imputer.fit(X[:,0:20])
# X[:,0:20]=imputer.transform(X[:,0:20])
# imputer=imputer.fit(y)
# y=imputer.transform(y)


# In[110]:


# # change X and y into Dataframe
# X=pd.DataFrame(X)
# y=pd.DataFrame(y)
# 
# # checking null values
# X.isnull().sum()


# In[111]:


# #checking missing values in "X" are there or not
# plt.figure(figsize=(10,5))
# sns.heatmap(X.isnull(), cmap="viridis")


# In[112]:


# # checking missing values in "y" are there or not
# plt.figure(figsize=(10,5))
# sns.heatmap(y.isnull(), cmap="viridis")


# #### Now finally imputing numeric missing values by MEAN method and categorical missing values by MEDIAN method
# 

# In[14]:


numeric_columns=['Transportation expense','Distance from Residence to Work','Service time','Age',
                 'Work load Average/day ','Hit target','Weight','Height','Body mass index',\
                 'Absenteeism time in hours']
categorical_columns=['Reason for absence', 'Month of absence', 'Day of the week','Seasons',
       'Disciplinary failure', 'Education', 'Son', 'Social drinker',
       'Social smoker', 'Pet']


# In[15]:


# imputing numeric missing values by MEAN method 
for i in numeric_columns:
    print(i)
    dataset.loc[:,i]=dataset.loc[:,i].fillna(dataset.loc[:,i].mean())


# In[16]:


# Imputing numeric missing values by MEDIAN method 
for i in categorical_columns:
    print(i)
    dataset.loc[:,i]=dataset.loc[:,i].fillna(dataset.loc[:,i].median())


# In[17]:


#checking missing values after Imputation
plt.figure(figsize=(10,5))
sns.heatmap(dataset.isnull(), cmap="viridis")
print("Total NA value in the dataset is =" +str(dataset.isnull().sum().sum()))


# In[18]:


dataset.isnull().sum()


# ###### Upto here i have dealt with Mission values problem

#   

# In[19]:


# copy dataset
#dataset_=dataset.copy()
#dataset=dataset_.copy()


# # 1.3 Outlier Analysis

# ## 1.3.a)  Checking outlier values with Box Plot

# In[20]:


# seperating categorical and numerical columns
numeric_columns=['Transportation expense','Distance from Residence to Work','Service time',                 'Age','Work load Average/day ','Hit target','Weight','Height','Body mass index',                 'Absenteeism time in hours']
categorical_columns=['ID','Reason for absence', 'Month of absence', 'Day of the week','Seasons',
       'Disciplinary failure', 'Education', 'Son', 'Social drinker',
       'Social smoker', 'Pet']


# In[21]:


dataset.columns


#   

# ###### Now we will find out the outlier values one by one in "Numeric variables" with the help of BOX plot

# In[22]:



sns.boxplot(dataset['Transportation expense'])


# In[23]:


sns.boxplot(dataset[ 'Distance from Residence to Work'])


# In[24]:


sns.boxplot(x=dataset[ 'Service time'])


# In[25]:


sns.boxplot(x=dataset['Age'])


# In[26]:


sns.boxplot(x=dataset['Work load Average/day '])


# In[27]:


sns.boxplot(x=dataset[ 'Hit target'])


# In[28]:


sns.boxplot(x=dataset['Weight'])


# In[29]:


#sns.boxplot(dataset['Height'])
sns.boxplot(dataset['Height'])


# In[30]:


sns.boxplot(x=dataset[ 'Body mass index'])


# # 1.3.b) Removing outliers values

# In[31]:


# copy dataset
#dataset_=dataset.copy()
#dataset=dataset_.copy()
dataset.isnull().sum()


# In[32]:


# Below are the Numeric varible with outliers
numeric_columns_outliers=['Transportation expense','Service time','Age','Hit target','Height',                 'Work load Average/day ']


# In[33]:


numeric_columns_outliers


# In[34]:


# for i in numeric_columns_outliers:
#     print(i)
#     q75,q25=np.percentile(dataset.loc[:,i],[75,25])
#     
#     # Inter quartile range
#     iqr=q75-q25
#     
#     #Lower Fence
#     min=q25-(iqr*1.5)
#     # Upper fence
#     max=q75+(iqr*1.5)
#     print(min)
#     print(max)
#     
# # Droping outliers values
# dataset=dataset.drop(dataset[dataset.loc[:,i]<min].index)
# dataset=dataset.drop(dataset[dataset.loc[:,i]>max].index)
# 


# ## 1.3.c) Detect and replace outliers with NA

# In[35]:


for i in numeric_columns_outliers:
    print(i)
    q75,q25=np.percentile(dataset.loc[:,i],[75,25])
    
    # Inter quartile range
    iqr=q75-q25
    
    #Lower Fence
    min=q25-(iqr*1.5)
    # Upper fence
    max=q75+(iqr*1.5)
    print(min)
    print(max)
    
    # Replacing all the outliers value to NA
    dataset.loc[dataset[i]< min,i] = np.nan
    dataset.loc[dataset[i]> max,i] = np.nan
# Checking if there is any missing value
dataset.isnull().sum().sum()


# In[36]:


missing_val=pd.DataFrame(dataset.isnull().sum())
missing_val


# In[37]:


#checking missing values after Imputation by Mean Method
plt.figure(figsize=(10,5))
sns.heatmap(dataset.isnull(), cmap="viridis")
print("Total NA value in the dataset is =" +str(dataset.isnull().sum().sum()))


# In[38]:


# imputing numeric missing values by MEAN method 
for i in numeric_columns:
    print(i)
    dataset.loc[:,i]=dataset.loc[:,i].fillna(dataset.loc[:,i].mean())


# In[39]:


#checking missing values after Imputation by Mean Method
plt.figure(figsize=(10,5))
sns.heatmap(dataset.isnull(), cmap="viridis")
print("Total NA value in the dataset is =" +str(dataset.isnull().sum().sum()))


# #### Upto here we Imputed outliers values 
# ####  Now, observe below in Box plot-  we have removed almost all the outliers values
# 

# In[40]:


sns.boxplot(x=dataset['Transportation expense'])


# In[41]:


sns.boxplot(x=dataset[ 'Service time'])


# In[42]:


sns.boxplot(x=dataset['Age'])


# In[43]:


sns.boxplot(x=dataset[ 'Hit target'])


# In[44]:


sns.boxplot(dataset['Height'])


# In[45]:


sns.boxplot(x=dataset['Work load Average/day '])


#  

# # 1.4 Feature Selection

# ### 1.4.1 Correlation Analysis

# In[46]:


from scipy import stats


# In[146]:


# Import Library
from scipy.stats import chi2_contingency


# In[47]:


# Correlation Analysis
dataset_corr=dataset.loc[:,numeric_columns]

# set the width and height of plot
f, ax=plt.subplots(figsize=(10,10))

# Generate correlation Matrix
corr=dataset_corr.corr()

#plot using seaborn Library
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),            cmap=sns.diverging_palette(250,10,as_cmap=True),
           square=True,ax=ax,annot = True)


# In[48]:


numeric_columns


# ##### Here  " Body Mass Index "  and  "Weight" are highly correlated. Therefore i have decided to remove one out of them

# ### 1.4.2 ANOVA Analysis

# In[49]:


#loop for ANOVA test Since the target variable is continuous
for i in categorical_columns:
    f, p = stats.f_oneway(dataset[i], dataset["Absenteeism time in hours"])
    print("P value for variable "+str(i)+" is "+str(p))


# Here P value of " Month of Absence" is 0.19141618354249687 which is greater than 
# the threshold value 0.05. 
# 
# My Hypothesis is-
# 
# H0= Independent varaible does not explain our Target variable.
# 
# H1= Independent varaible explain our Target variable.
# 
# 
# If p> 0.05 then select the NULL Hypothesis (H0), that means this particular Independent 
# variable is not going to explain my Target Variable.
# 
# 
# Now, in my case -
# 
# P value for variable "Month of absence" is 0.19141618354249687, that means 'Month of absence' is not 
# explaining "Absenteeism time in hours"
# 
# 
# conclusion-
# 
# I will Remove 'Month of absence' by Anova statistical Analysis  and,
# 
# I will remove 'Weight' by correlation Plot Analysis.

#    

#   

# In[50]:


# copy dataset
#dataset_=dataset.copy()
#dataset=dataset_.copy()


# In[51]:


# Droping the variables which are not Important
to_drop = ['Weight','Month of absence']
dataset = dataset.drop(to_drop, axis = 1)


# In[52]:


dataset.shape


# # 1.5 Feature scaling

# In[53]:


numeric_columns=['Transportation expense','Distance from Residence to Work','Service time',                 'Age','Work load Average/day ','Hit target','Height','Body mass index',                 'Absenteeism time in hours']


# In[54]:


# Checking if there is any normally distributed variable in data
for i in numeric_columns:
    if i == 'Absenteeism time in hours':
        continue
    sns.distplot(dataset[i],bins = 'auto')
    plt.title("Checking Distribution for Variable "+str(i))
    plt.ylabel("Density")
    plt.show()


# In[55]:


# Since there is no normally distributed curve we will use Normalizationg for Feature Scalling
# #Normalization
for i in numeric_columns:
    if i == 'Absenteeism time in hours':
        continue
    dataset[i] = (dataset[i] - dataset[i].min())/(dataset[i].max()-dataset[i].min())


# In[56]:


dataset.head()


# In[57]:


dataset__=dataset.copy()
#dataset=dataset__.copy()


#   

# # 2 Machine Leaning Model

# In[58]:


# Important categorical variable 
categorical_columns=['ID','Reason for absence','Day of the week','Seasons',
       'Disciplinary failure', 'Education', 'Son', 'Social drinker',
       'Social smoker', 'Pet']
dataset[categorical_columns].nunique()


# In[59]:


# Get dummy variables for categorical variables
dataset = pd.get_dummies(data = dataset, columns = categorical_columns )

# Copying dataframe
dataset1 = dataset.copy()


# In[60]:


dataset.head()


# In[61]:


dataset.nunique()


# In[165]:


#dataset['Reason for absence_1.0'].head()


# In[62]:


### AVoiding dummy variable trap
# selecting columns to Avoid dummy variable trap
drop_columns=['ID_1',"Reason for absence_1.0","Day of the week_2","Seasons_1",              "Disciplinary failure_0.0","Education_1.0","Son_0.0","Social drinker_0.0",              "Social smoker_0.0","Pet_0.0"]
dataset = dataset.drop(drop_columns, axis = 1)


# In[63]:


dataset.shape


# In[64]:


dataset1.shape


# In[65]:


dataset.head()
#pd.DataFrame(X).head()


# In[66]:


# splitting the dataset into X and y
X = dataset.drop("Absenteeism time in hours", axis = 1)
y=dataset.iloc[:,8].values


# In[77]:


y


# In[68]:


# splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.30, random_state=0)


# In[69]:


type(X_train)


# In[70]:


y_train.shape


# In[71]:


X_test.shape


# In[72]:


y_test.shape


# ## 2.1 Multiple Linear Regression

# In[73]:


# Importing Library for Linear Regression
from sklearn.linear_model import LinearRegression

# Fitting simple linear regression to the training data
regressor=LinearRegression()
LR_model=regressor.fit(X_train,y_train)
    


# In[74]:


# predicting the test set results
y_pred= LR_model.predict(X_test)
y_pred


# In[75]:


y_test


# In[78]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[79]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = LR_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[80]:


# Calculating RMSE for test data to check accuracy
from sklearn.metrics import mean_squared_error
y_pred= LR_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# ###### Here we can compare RMSE for training and Test Data -
# ######  RMSE for Training Data is 3.1339768352912394e-14
# ######  RMSE For Test data = 2.985187537059385e-14
# 
# ##### There is not much difference between them.
# 
# ##### Therefore, we conclude that there is no overfitting in Linear Regression Model But there might be Multicollinearity in the model because Accuracy is 100%
# 

# In[184]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# # 2.2 Support vector Regressor

# In[105]:


# Importing Library for SVR
from sklearn.svm import SVR

# Fitting SVR to the dataset
regressor = SVR(kernel = 'rbf')
SVR_model=regressor.fit(X_train,y_train)
SVR_model


# In[82]:


# predicting the test results
y_pred= SVR_model.predict(X_test)
y_pred


# In[83]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[84]:


# Calculating RMSE for test data to check accuracy
from sklearn.metrics import mean_squared_error
y_pred= SVR_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[85]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = SVR_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[86]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# #### SVR is not giving better accuracy.
# #### We will see it after dimensionality reduction

# ## 2.3  Decision Tree

# In[87]:


# Importing Library for Decision Tree
from sklearn.tree import DecisionTreeRegressor

# # Fitting Decision Tree Regression to the dataset
regressor = DecisionTreeRegressor(random_state = 0)
DT_model=regressor.fit(X_train, y_train)
DT_model


# In[88]:


# Predicting a new result
y_pred = DT_model.predict(X_test)
y_pred


# In[193]:


y_test


# In[89]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = DT_model, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[90]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = DT_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[91]:


# Calculating RMSE for test data to check accuracy
y_pred= DT_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# 
# ##### RMSE For Training data = 3.022810778490771e-16
# ##### RMSE For Test data = 1.5186527932039984
# ##### There is slight differnece in RMSE of both Partition. 
# ##### RMSE for training set is less than test set
# ##### This means that my  Decision Tree model is overfitting

# In[92]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# # 2.4 Random Forest

# In[93]:


# Importing Library for Random Forest
from sklearn.ensemble import RandomForestRegressor
# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
RF_model=regressor.fit(X_train, y_train)


# In[94]:


# Predicting a new result
y_pred = regressor.predict(X_test)
y_pred


# In[200]:


y_test


# In[95]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = RF_model, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[96]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = RF_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[97]:


# Calculating RMSE for test data to check accuracy
y_pred= RF_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# ##### RMSE For Training data =0.5188828253960112
# ##### RMSE For Test data =  0.7915322565173188
# ##### There is slight differnece in RMSE of both Partition.
# ##### RMSE for training set is slightly  less than test set
# ##### This means that my Random Forest model may be overfitting, so we will look after Dimensionality Reduction in PCA

# In[204]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# # 2.5 Gradiet Boosting

# In[205]:


# Importing library for Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
# Building model on top of training dataset
GB_model = GradientBoostingRegressor().fit(X_train, y_train)


# In[206]:


# Predicting a new result
y_pred = GB_model.predict(X_test)
y_pred


# In[207]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = GB_model, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[208]:


# Calculating RMSE for training data to check for over fitting
pred_train = GB_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))
 


# In[209]:


# Calculating RMSE for test data to check accuracy
pred_test = GB_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,pred_test))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# ###### we can compare RMSE of Training and test data
# ###### RMSE For Training data = 0.0016934191806534681
# ###### RMSE For Test data = 0.4755633246140562
# ###### There is some  DIfference, So chances of overfitting may occur
# ###### But we can see the result of same model after Dimensionality reduction using  PVA
# 

# In[212]:


# calculate R^2 value to check the goodness of fit
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,pred_test)))


# In[213]:


# Building the optimal model using Backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((740,1)).astype(int), values=X ,axis=1)
X_opt = X[:, 0:92]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# ###### We can see in warning message above there is strong collinearity in the model. 
# ###### Therefore we will remove redundant variable by PCA

# # 3 Applying Principal component Analysis

# In[214]:


# Copying dataframe
#dataset1 = dataset.copy()
#dataset = dataset1.copy()
dataset.shape


# In[215]:


# splitting the dataset into X and y
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,8].values
X.shape


# In[216]:


# splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.30, random_state=0)


# In[217]:


X_train.shape


# In[218]:


X_test.shape


# In[219]:


X.shape


# In[98]:


# Performing PCA on Train and test data seperately
from sklearn.decomposition import PCA

#Data has 92 variables so no of components of PCA = 92
pca=PCA(n_components=92)

# Fitting this object to the Training and test set
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# The amount of variance that each PC explains
explained_variance = pca.explained_variance_ratio_

# Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

plt.plot(var1)
plt.show()

explained_variance


# In[99]:


# From the above plot selecting 30 components since it explains almost 95+ % data variance
pca = PCA(n_components=30)

# Fitting the selected components to the data
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[100]:


# Building the optimal model using Backward elimination
import statsmodels.formula.api as sm

X_train=np.append(arr=np.ones((518,1)).astype(int), values=X_train ,axis=1)
X_train=pd.DataFrame(X_train)
# selecting training columns
X_opt = X_train.iloc[:, 0:30]

regressor_OLS=sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()


# ###### Now we can see there is no warning of multicollinearity in my training dataset
# ###### But
# ######  There are some columns whose p value is greater than 0.05 , Therefore we need to remove them.
# 

# In[223]:


# Dropping irrelevant columns in Training dataset
X_train = X_train.drop([3,13,22], axis = 1)
X_train.shape


# In[224]:


X_train.head()


# ###### we will run our backward elimination code 2nd time just to see p values for Train dataset

# In[225]:


# Building the optimal model using Backward elimination
import statsmodels.formula.api as sm
X_train=pd.DataFrame(X_train)
# selecting training columns
X_opt = X_train.iloc[:, 0:28]

regressor_OLS=sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()


# ###### Now we can see above all the columns have p value less than 0.05 (significant value)
# ###### And also it does not have any multicollinearity

#   

#  ###### we will dropping the same columns  in X_test as we have done in X_train just to make equal no. of columns

# In[226]:


# Dropping the same columns in  test dataset
X_test=pd.DataFrame(X_test)
X_test = X_test.drop([3,13,22], axis = 1)
X_test.shape


# In[101]:


# Building the optimal model using Backward elimination
import statsmodels.formula.api as sm

X_test=np.append(arr=np.ones((222,1)).astype(int), values=X_test ,axis=1)
X_test=pd.DataFrame(X_test)
# selecting training columns
X_opt = X_test.iloc[:, 0:28]

regressor_OLS=sm.OLS(endog=y_pred, exog=X_opt).fit()
regressor_OLS.summary()


# In[228]:


X_train.shape


# In[229]:


X_test.shape


# # 3.a) Multiple Linear Regression after Dimensionality 
# #       Reduction

# In[102]:


# Importing Library for Linear Regression
from sklearn.linear_model import LinearRegression

# Fitting simple linear regression to the training data
regressor=LinearRegression()
LR_model=regressor.fit(X_train,y_train)


# In[103]:


# predicting the test set results
y_pred= LR_model.predict(X_test)
y_pred


# In[104]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = LR_model, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy of Multiple Linear Regression ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracy.std()))


# In[233]:


# Calculating RMSE for training data to check for over fitting
pred_train = LR_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[234]:


# Calculating RMSE for test data to check accuracy
from sklearn.metrics import mean_squared_error
y_pred= LR_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# ###### RMSE For Training data = 0.0005389956165122671
# ###### RMSE For Test data = 0.0042534939026206324
# 
# ##### Not much difference, Therefore there is no over fitting

# In[235]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# # 3.b) Support Vector Regression after Dimensionality Reduction

# In[236]:


# Importing Library for SVR
from sklearn.svm import SVR

# Fitting SVR to the dataset
regressor = SVR(kernel = 'rbf')
SVR_model=regressor.fit(X_train,y_train)
SVR_model


# In[237]:


# predicting the test results
y_pred= SVR_model.predict(X_test)
y_pred


# In[238]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator =SVR_model, X = X_train, y = y_train, cv =5 )
accuracy=accuracies.mean()
print(' Accuracy of SVR_Model ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[239]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = SVR_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[240]:


# Calculating RMSE for test data to check accuracy
from sklearn.metrics import mean_squared_error
y_pred= SVR_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[241]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# # 3.c) Decision Tree after Dimensionality Reduction

# In[242]:


# Importing Library for Decision Tree
from sklearn.tree import DecisionTreeRegressor

# # Fitting Decision Tree Regression to the dataset
regressor = DecisionTreeRegressor(random_state = 0)
DT_model=regressor.fit(X_train, y_train)
DT_model


# In[243]:


# Predicting a new result
y_pred = DT_model.predict(X_test)
y_pred


# In[244]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = DT_model, X = X_train, y = y_train, cv =5 )
accuracy=accuracies.mean()
print(' Accuracy of Decision Tree='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[245]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = DT_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[246]:


# Calculating RMSE for test data to check accuracy
y_pred= DT_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# ##### compare RMSE of train and test above there is some difference, so this may be overfitting after doing PCA

# In[247]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# In[ ]:





# # 3.d) Random Forest after Dimensionality Reduction

# In[248]:


# Importing Library for Random Forest
from sklearn.ensemble import RandomForestRegressor
# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
RF_model=regressor.fit(X_train, y_train)


# In[249]:


# Predicting a new result
y_pred = regressor.predict(X_test)
y_pred


# In[250]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = RF_model, X = X_train, y = y_train, cv =5 )
accuracy=accuracies.mean()
print(' Accuracy of Random forest ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[251]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = RF_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[252]:


# Calculating RMSE for test data to check accuracy
y_pred= RF_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[253]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# In[ ]:





# # 3.e) Gradiet Boosting after Dimensionality Reduction

# In[254]:


# Importing library for Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
# Building model on top of training dataset
GB_model = GradientBoostingRegressor().fit(X_train, y_train)


# In[255]:


# Predicting a new result
y_pred = GB_model.predict(X_test)
y_pred


# In[256]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = GB_model, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy of Gradiet Boosting ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[257]:


# Calculating RMSE for training data to check for over fitting
pred_train = GB_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))
 


# In[258]:


# Calculating RMSE for test data to check accuracy
pred_test = GB_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,pred_test))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[259]:


# calculate R^2 value to check the goodness of fit
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,pred_test)))


# # 4 Model Selection

# #### Accuracy of Multiple Linear Regression =0.9999999957045496
# ####   Accuracy of SVR_Model =0.30815910537930513
# ####    Accuracy of Decision Tree=0.9234293909600103
# ####       Accuracy of Random forest =0.9725959025281747
# ####       Accuracy of Gradiet Boosting =0.9686809161455935
# 
# ## Multiple Linear Regression has highest Accuracy

#  

# ###  Comparing RMSE value for all the Model

# ##### Multiple Linear Regression

# In[ ]:


#RMSE For Training data = 0.0005389956165122671
#RMSE For Test data = 0.0042534939026206324
# difference (test- training) is = 0.003985


# ##### SVR 

# In[ ]:


#Root Mean Squared Error For Training data = 11.199683762156287
#Root Mean Squared Error For Test data = 9.724575339184948
# difference (test- training) is -1.4751


# ##### Decision tree

# In[ ]:


#Root Mean Squared Error For Training data = 3.022810778490771e-16
#Root Mean Squared Error For Test data = 1.8599622199011085
# difference (test- training) is 1.8599


# ##### Random forest

# In[ ]:


#Root Mean Squared Error For Training data = 0.7829312646083995
#Root Mean Squared Error For Test data = 1.0642907496085376
# difference (test- training) is 1.8599


# ##### Gradient Boosting

# In[ ]:


#Root Mean Squared Error For Training data = 0.0016674934820754559
#Root Mean Squared Error For Test data = 0.6505230395132787
# difference (test- training) is 0.7910


# #### After comparing RMSE value of all the MODELs .
# #### We can conlcude that the difference in RMSE of Train and Test Data is less in  case of Multiple Linear Regression which is around  0.003985
# ### Therefore, there is very less chance of having Overfitting in Multiple Linear Regression
# ### as compared to other four models

#  

#  

# ## we can also see the Result of K fold cross validation of Multiple Linear regression
# #### Accuracy standard Deviation = 0, which simply prove that there is no overfitting, this model has test 10 times by taking different 10 parts.

# In[ ]:


#Accuracy of all  the partitions=[1.         0.99999998   1.         1.         1.         1.
#                                  1.         0.99999999   1.         1.        ]


# ## Therefore, I Choose Multiple Linear regression as the best Model.

# In[193]:


# Writing xlsx file
dataset.to_excel("dataset_output_from_python.xlsx", index=False)

