#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing usual libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing dataset csv to pandas dataframe

automobile = pd.read_csv("CarPrice_Assignment.csv")
automobile.head()


# In[3]:


#checking number of rows and columns

automobile.shape


# In[4]:


#checking dtypes and null values of columns

automobile.info()


# In[5]:


#checking summary of numeric variables

automobile.describe()


# In[6]:


#checking number of columns of each data type for general EDA

automobile.dtypes.value_counts()


# In[7]:


#cleaning Car Name to keep only brand(company) name and remove model names 

automobile['CarName']=automobile['CarName'].apply(lambda x:x.split(' ', 1)[0])
automobile.rename(columns = {'CarName':'companyname'}, inplace = True)
automobile.head()


# In[8]:


#checking unique values in company name column

automobile.companyname.unique()


# In[9]:


#counting number of unique company names

automobile.companyname.nunique()


# In[10]:


# Fixing values in company name

automobile.companyname = automobile.companyname.str.lower()

def replace_name(a,b):
    automobile.companyname.replace(a,b,inplace=True)

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

automobile.companyname.unique()


# In[11]:


#counting number of unique company names

automobile.companyname.nunique()


# In[12]:


#plotting count of company names

plt.figure(figsize=(30, 8))
plt1=sns.countplot(x=automobile.companyname, data=automobile, order= automobile.companyname.value_counts().index)
plt.title('Company Wise Popularity', size=14)
plt1.set_xlabel('Car company', fontsize=14)
plt1.set_ylabel('Frequency of Car Body', fontsize=14)
plt1.set_xticklabels(plt1.get_xticklabels(),rotation=360, size=14)
plt.show()


# In[13]:


#plotting company wise average price of car

plt.figure(figsize=(30, 6))

df = pd.DataFrame(automobile.groupby(['companyname'])['price'].mean().sort_values())
df=df.reset_index(drop=False)
plt1=sns.barplot(x="companyname", y="price", data=df)
plt1.set_title('Car Range vs Average Price', size=14)
plt1.set_xlabel('Car company', fontsize=14)
plt1.set_ylabel('Price', fontsize=14)
plt1.set_xticklabels(plt1.get_xticklabels(),rotation=360, size=14)
plt.show()


# In[14]:


#Binning the Car Companies based on avg prices of each Company.

def replace_values(a,b):
    automobile.companyname.replace(a,b,inplace=True)

replace_values('chevrolet','Low_End')
replace_values('dodge','Low_End')
replace_values('plymouth','Low_End')
replace_values('honda','Low_End')
replace_values('subaru','Low_End')
replace_values('isuzu','Low_End')
replace_values('mitsubishi','Budget')
replace_values('renault','Budget')
replace_values('toyota','Budget')
replace_values('volkswagen','Budget')
replace_values('nissan','Budget')
replace_values('mazda','Budget')
replace_values('saab','Medium')
replace_values('peugeot','Medium')
replace_values('alfa-romero','Medium')
replace_values('mercury','Medium')
replace_values('audi','Medium')
replace_values('volvo','Medium')
replace_values('bmw','High_End')
replace_values('porsche','High_End')
replace_values('buick','High_End')
replace_values('jaguar','High_End')

automobile.rename(columns = {'companyname':'segment'}, inplace = True)
automobile.head()


# In[19]:


#Visualizing Numeric Variables
#checking distribution and spread of car price

plt.figure(figsize=(20,6))

plt.subplot(1,2,1)
plt.title('Car Price Distribution Plot')
sns.distplot(automobile.price)

plt.subplot(1,2,2)
plt.title('Car Price Spread')
sns.boxplot(y=automobile.price)

plt.show()


# In[20]:


# checking numeric columns

automobile.select_dtypes(include=['float64','int64']).columns


# In[21]:


#function to plot scatter plot numeric variables with price

def pp(x,y):
    sns.pairplot(automobile, x_vars=[x,y], y_vars='price',height=4, aspect=1, kind='scatter')
    plt.show()

pp('carlength', 'carwidth')
pp('carwidth', 'curbweight')


# In[22]:


#function to plot scatter plot numeric variables with price

def pp(x,y,z):
    sns.pairplot(automobile, x_vars=[x,y,z], y_vars='price',height=4, aspect=1, kind='scatter')
    plt.show()

pp('wheelbase', 'compressionratio', 'enginesize')
pp('boreratio', 'horsepower', 'peakrpm')
pp('stroke', 'highwaympg', 'citympg')


# In[23]:


#converting cylinder number to numeric and replacing values

def replace_values(a,b):
    automobile.cylindernumber.replace(a,b,inplace=True)

replace_values('four','4')
replace_values('six','6')
replace_values('five','5')
replace_values('three','3')
replace_values('twelve','12')
replace_values('two','2')
replace_values('eight','8')

automobile.cylindernumber=automobile.cylindernumber.astype('int')


# In[24]:


automobile.symboling.unique()


# In[25]:


#converting symboling to categorical because the numeric values imply weight

def replace_values(a,b):
    automobile.symboling.replace(a,b,inplace=True)

replace_values(3,'Very_Risky')
replace_values(2,'Moderately_Risky')
replace_values(1,'Neutral')
replace_values(0,'Safe')
replace_values(-1,'Moderately_Safe')
replace_values(-2,'Very_Safe')


# In[26]:


# Converting variables with 2 values to 1 and 0

automobile['fueltype'] = automobile['fueltype'].map({'gas': 1, 'diesel': 0})
automobile['aspiration'] = automobile['aspiration'].map({'std': 1, 'turbo': 0})
automobile['doornumber'] = automobile['doornumber'].map({'two': 1, 'four': 0})
automobile['enginelocation'] = automobile['enginelocation'].map({'front': 1, 'rear': 0})


# In[27]:


#dropping card_Id because it has all unique values


# In[28]:


#numeric variables

num_vars=automobile.select_dtypes(include=['float64','int64']).columns


# In[29]:


# plotting heatmap to check correlation amongst variables

plt.figure(figsize = (20,10))  
sns.heatmap(automobile[num_vars].corr(),cmap="YlGnBu",annot = True)


# In[30]:


#dropping variables which are highly correlated to other variables

automobile.drop(['compressionratio','carwidth','curbweight','wheelbase','citympg'], axis =1, inplace = True)
automobile.head()


# In[31]:


#getting dummies for categorical variables

df = pd.get_dummies(automobile)
df.head()


# In[32]:


#checking column names for dummy variables

df.columns


# In[33]:


#DIVIDING INTO TRAIN AND TEST
# importing necessary libraries and functions

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[34]:


#SCALING NUMERIC VARIABLES
# for scaling

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[35]:


# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['fueltype', 'aspiration', 'doornumber', 'enginelocation', 'enginesize','horsepower', 
            'peakrpm', 'highwaympg', 'carlength', 'carheight', 'boreratio', 'stroke', 'price']


df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

df_train.head()


# In[36]:


#Dividing into X and Y sets for the Model Building
#dividing into x and y sets where y has the variable we have to predict

y_train = df_train.pop('price')
X_train = df_train


# In[37]:


# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[38]:


# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)
 
rfe = RFE(lm)
ref = RFE(10)        # running RFE
rfe = rfe.fit(X_train, y_train)


# In[39]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[40]:


#checking RFE columns
col = X_train.columns[rfe.support_]
col


# In[41]:


#Building model using statsmodel, for the detailed statistics
# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]


# In[42]:


# Adding a constant variable 
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)


# In[43]:


#function for checking VIF

def checkVIF(X):
    vif = pd.DataFrame()
    vif['variable'] = X.columns    
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)


# In[44]:


# building MODEL #1

lm = sm.OLS(y_train,X_train_rfe).fit() # fitting the model
print(lm.summary()) # model summary


# In[45]:


#dropping constant to calculate VIF

X_train_rfe.drop('const', axis = 1, inplace=True)


# In[46]:


#checking VIF

checkVIF(X_train_rfe)


# In[47]:


#dopping boreratio

X_train_new = X_train_rfe.drop(["boreratio"], axis = 1)


# In[48]:


#building MODEL #2 after dropping boreratio

X_train_new = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_new).fit() # fitting the model
print(lm.summary()) # model summary


# In[49]:


#dropping constant to calculate VIF

X_train_new.drop('const', axis=1, inplace=True)


# In[50]:


#checking VIF

checkVIF(X_train_new)


# In[51]:


#dopping enginelocation because it has the highest p-value and also high VIF. it has very few values for rear as we saw earlier

X_train_new.drop(["enginelocation"], axis=1, inplace=True)


# In[52]:


#building MODEL #3 after dropping enginelocation

X_train_new = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_new).fit() # fitting the model
print(lm.summary()) # model summary


# In[53]:


#dropping constant to calculate VIF

X_train_new.drop('const', axis=1, inplace=True)
#checking VIF

checkVIF(X_train_new)


# In[54]:


#dopping horsepower 

X_train_new.drop(["horsepower"], axis=1, inplace=True)
#building MODEL #4 after dropping horsepower

X_train_new = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_new).fit() # fitting the model
print(lm.summary()) # model summary


# In[55]:


#dropping constant to calculate VIF

X_train_new.drop('const', axis=1, inplace=True)
#checking VIF

checkVIF(X_train_new)


# In[56]:


#dopping carlength 

X_train_new.drop(["carlength"], axis=1, inplace=True)
#building MODEL #5 after dropping carlength

X_train_new = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_new).fit() # fitting the model
print(lm.summary()) # model summary


# In[57]:


#dropping constant to calculate VIF

X_train_vif=X_train_new.drop('const', axis=1)
#checking VIF

checkVIF(X_train_vif)


# In[58]:


#Residual Analysis of the train data
#calculating price on train set using the model built

y_train_price = lm.predict(X_train_new)


# In[59]:


# Plot the histogram of the error terms

fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label


# In[60]:


# Plotting y_train and y_train_price to understand the residuals.

plt.figure(figsize = (8,6))
plt.scatter(y_train,y_train_price)
plt.title('y_train vs y_train_price', fontsize=20)              # Plot heading 
plt.xlabel('y_train', fontsize=18)                          # X-label
plt.ylabel('y_train_price', fontsize=16)                          # Y-label


# In[61]:


# Actual vs Predicted for TRAIN SET

plt.figure(figsize = (8,5))
c = [i for i in range(1,144,1)]
d = [i for i in range(1,144,1)]
plt.plot(c, y_train_price, color="green", linewidth=1, linestyle="-")     #Plotting Actual
plt.plot(d, y_train, color="yellow",  linewidth=1, linestyle="-")  #Plotting predicted
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Car Price', fontsize=16)  
plt.show()


# In[62]:


# Error terms for TRAIN SET
plt.figure(figsize = (8,5))
c = [i for i in range(1,144,1)]
plt.scatter(c,y_train-y_train_price)

plt.title('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label


# In[63]:


#Making Predictions
# Applying the scaling on the test sets

num_vars = ['fueltype', 'aspiration', 'doornumber', 'enginelocation', 'enginesize','horsepower', 
            'peakrpm', 'highwaympg', 'carlength', 'carheight', 'boreratio', 'stroke', 'price']

df_test[num_vars] = scaler.transform(df_test[num_vars])


# In[64]:


# Dividing into X_test and y_test

y_test = df_test.pop('price')
X_test = df_test


# In[65]:


X_train_new.drop('const', axis=1, inplace=True)


# In[66]:


# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)
# Making predictions
y_pred = lm.predict(X_test_new)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[67]:


#Model Evaluation
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label


# In[68]:


# Actual vs Predicted
c = [i for i in range(1,63,1)]
d = [i for i in range(1,63,1)]
plt.plot(c, y_pred, color="green", linewidth=1, linestyle="-")     #Plotting Actual
plt.plot(d, y_test, color="yellow",  linewidth=1, linestyle="-")  #Plotting predicted
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Car Price', fontsize=16)  
plt.show()


# In[69]:


# Error terms

fig = plt.figure()
c = [i for i in range(1,63,1)]
plt.scatter(c,y_test-y_pred)

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label


# In[70]:


#RMSE score for test set

import numpy as np
from sklearn import metrics
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[71]:


#RMSE score for train set

import numpy as np
from sklearn import metrics
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_train, y_train_price)))


# In[72]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[73]:


r2_score(y_train, y_train_price)

