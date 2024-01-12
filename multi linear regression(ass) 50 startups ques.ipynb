# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:30:55 2023

@author: dell
"""
### importing the data   ###

import pandas as pd
import numpy as np
df = pd.read_csv("C:\\Users\\dell\\Downloads\\50_Startups.csv")
df
df.shape
df.info()


# EDA #
#EDA----->EXPLORATORY DATA ANALYSIS
#BOXPLOT AND OUTLIERS CALCULATION #
import seaborn as sns
import matplotlib.pyplot as plt
data = ['R_D_Spend','Administration','Marketing_Spend','Profit']
for column in data:
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.boxplot(x=df[column])
    plt.title(" Horizontal Box Plot of column")
    plt.show()
#so basically we have seen the ouliers at once without doing everytime for each variable using seaborn#

"""removing the ouliers"""
# List of column names with continuous variables
continuous_columns = ['R_D_Spend','Administration','Marketing_Spend','Profit']  
# Create a new DataFrame without outliers for each continuous column
data_without_outliers = df.copy()
for df.cloumns in continuous_columns:
    Q1 = data_without_outliers[column].quantile(0.25)
    Q3 = data_without_outliers[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker_Length = Q1 - 1.5 * IQR
    upper_whisker_Length = Q3 + 1.5 * IQR
    data_without_outliers = data_without_outliers[(data_without_outliers[column] >= lower_whisker_Length) & (data_without_outliers[column]<= upper_whisker_Length)]
# Print the cleaned data without outliers
print(data_without_outliers)
df = data_without_outliers
print(df)
df.shape
df.info() 

# constructing histograms,calculating skewness and kurtosis values#

df.hist()
df.skew()
df.kurt()
df.describe()

"""Standardising the data"""
# R_D_Spend,Administration,Marketing_Spend continuos variables as independent variables#

X1 = df.iloc[:,0:3]
X1
list(X1)

# transforming using standard scaler as they are continuos variables#

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X1 = SS.fit_transform(X1)
SS_X1

# converting into dataframe#

X2 = pd.DataFrame(SS_X1)
X2.columns = list(X1)
X2

"""  PROFIT COLUMN as this target variable """

X3 = df.iloc[:,4:5]
X3
SS_X2 = SS.fit_transform(X3)
SS_X2
X4 = pd.DataFrame(SS_X2)
X4.columns = list(X3)
X4

"""   STATE COLUMN as this is categorical column we are performing the label encoding process """

X5 = df.iloc[:,3:4]
X5
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
LE1 = LE.fit_transform(X5)
LE1
X6 = pd.DataFrame(LE1)
X6.columns = list(X5)
X6.columns
X6

""" X7 variable contains all the independent variables data after standardising the data """

X7 = pd.concat([X2,X6],axis = 1)
X7
#ALL THE DATA IN ON ONE FRAME#


""" FINDING THE CORRELATION X8 = pd.concat([X7,X4],axis = 1)
X8 BETWEEN ALL THE X VARIABLES AND TARGET VARIABLE (PROFIT)"""

X8 = pd.concat([X7,X4],axis = 1)
X8.corr()

# we have a larger relation with R&D Spend  with profit target variable so we will first fit that x variable 
# fit the linear regression model#

""" Defining X and Y variables """
X = pd.concat([X2,X6],axis = 1)
X

Y = X4
Y

"""  FITTING THE MODEL"""       # case 1
""" here X7 and X4 are the data frames where our transformed data is stored """

Y = X4["Profit"]
X = X7[["R_D_Spend"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))
print(" Root Mean Squared Error:",np.sqrt(mse).round(3))
from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2
print("r2:",R2.round(3))
#0.947


# case2 

""" VIF (variance influence factor is one of the metric which is used to calculate the relationship between  the two independent variables in order to see
there is a presence of multi collinearity, if exists it will effect the accuracy score of the model. so the vif factor ranges follows as below mentioned
VIF = 1/1-r2 ,VIF < 5 no multi collinearity
VIF : 5- 10  some multi collinearity issues will be present but can be accepted
VIF > 10 not at all acceptable """

Y = X7["R_D_Spend"]
X = X7[["Administration"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)      # as VIF here between the mentioned variables is < 5 these can be taken together

""" adding the administration column to the R_D_Spend """
 
Y = X4["Profit"]    
X = X7[["R_D_Spend","Administration"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))
print(" Root Mean Squared Error:",np.sqrt(mse).round(3))
from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2
print("r2:",R2.round(3))
#0.948

#case 3 

Y = X7["R_D_Spend"]
X = X7[["State"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)    #1.0110804010836976  can be considered

Y = X7["State"]
X = X7[["Administration"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)   #1.0001403759001628  can be considered


""" adding the State column to the R_D_Spend,administration  """

Y = X4["Profit"]     
X = X7[["R_D_Spend","Administration","State"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))
print(" Root Mean Squared Error:",np.sqrt(mse).round(3))
from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2
print("r2:",R2.round(3))
#0.951

"""here we are adding the x variables in order to increase the r2 value but the very next highly related variable is
Marketing Spend but before adding it we have to check for multi collinearity """

Y = X7["R_D_Spend"]
X = X7[["Marketing_Spend"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)   #2.103205816276043 can be considered

Y = X7["Administration"]
X = X7[["Marketing_Spend"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #1.0010349416824806 can be considered

Y = X7["State"]
X = X7[["Marketing_Spend"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #1.006069180313524  can be considered


Y = X4["Profit"]     
X = X7[["R_D_Spend","Administration","State","Marketing_Spend"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y_pred = LR.predict(X)
Y_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))
print(" Root Mean Squared Error:",np.sqrt(mse).round(3))
from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2
print("r2:",R2.round(3))

# Residual Analysis
#fit the model with seaborn,statsmodels package
import pandas as pd
df_residual = pd.read_csv("C:\\Users\\dell\\Downloads\\50_Startups.csv")
df_residual

#format the plot background as scatter plots for all variables
import seaborn as sns
sns.set_style(style="darkgrid")
sns.pairplot(X8)

#build a model
import statsmodels.formula.api as smf
model = smf.ols("Profit~R_D_Spend+Administration+State+Marketing_Spend",data=X8).fit()
model.summary()


import matplotlib.pyplot as plt
import statsmodels.api as sm

qqplot = sm.qqplot(model.resid,line = "q")
plt.title("Normal Q-Q plot of residuals")
plt.show()

import numpy as np
list(np.where((model.resid) > 10))


import pandas as pd
import statsmodels.formula.api as smf
# Create an empty list to store R-squared values
rsquared_values = []

# Define the models and calculate R-squared for each
models = [
    'Profit~R_D_Spend',
    'Profit~R_D_Spend+Administration',
    'Profit~R_D_Spend+Administration+State',
    'Profit~R_D_Spend+Administration+State+Marketing_Spend',
]

for model_formula in models:
    model = smf.ols(model_formula, data=X8).fit()
    rsquared = model.rsquared
    rsquared_values.append(rsquared)

# Create a DataFrame to display the R-squared values
model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
rsquared_df = pd.DataFrame({'Model': model_names, 'R-squared': rsquared_values})

print(rsquared_df)
