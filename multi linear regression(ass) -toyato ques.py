# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:07:59 2023

@author: dell
"""
# importing the data #

import numpy as np
import pandas as pd
df = pd.read_csv("C:\\Users\\dell\\Downloads\\ToyotaCorolla.csv",encoding = "latin")
df
df.info()
df.shape  #(1436, 38)
# List of column indexes to drop
column_indexes_to_drop = [0, 1, 4, 5, 7, 9, 10, 11, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]

# Drop the specified columns by index from the DataFrame

df = df.drop(df.columns[column_indexes_to_drop], axis=1)
# df now contains only the remaining columns
df
df.info()  #[1436 rows x 9 columns]


# EDA #
#EDA----->EXPLORATORY DATA ANALYSIS
#BOXPLOT AND OUTLIERS CALCULATION #
import seaborn as sns
import matplotlib.pyplot as plt
data = ['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']
for column in data:
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.boxplot(x=df[column])
    plt.title(" Horizontal Box Plot of column")
    plt.show()
#so basically we have seen the ouliers at once without doing everytime for each variable using seaborn#

"""removing the ouliers"""
# List of column names with continuous variables
continuous_columns = ['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']  
# Create a new DataFrame without outliers for each continuous column
data_without_outliers = df.copy()
for columns in continuous_columns:
    Q1 = data_without_outliers[column].quantile(0.25)
    Q3 = data_without_outliers[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker_Length = Q1 - 1.5 * IQR
    upper_whisker_Length = Q3 + 1.5 * IQR
    data_without_outliers = data_without_outliers[(data_without_outliers[column] >= lower_whisker_Length) & (data_without_outliers[column]<= upper_whisker_Length)]
# Print the cleaned data without outliers
print(data_without_outliers)#[1363 rows x 9 columns]
df = data_without_outliers
print(df)
df.shape#(1363, 9)
df.info() 
df.head()
df.isna().sum()

# constructing histograms,calculating skewness and kurtosis values#

df.hist()
df.skew()
df.kurt()
df.describe()

#standardising the variables we are using the standard scaler as all the variables are continuos#

X = df.iloc[:,1:9]
X.info()
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X
X1 = pd.DataFrame(SS_X)
X1
X1.columns = list(X)
X1.columns
X1
Y = df.iloc[:,0:1]
Y
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_Y = SS.fit_transform(Y)
SS_Y
Y1 = pd.DataFrame(SS_Y)
Y1
Y1.columns = list(Y)
Y1.columns
Y1
""" to identify the correlation in the form of a table between the all independent and dependent variable """
pd.set_option('display.max_columns', None)
df1 = pd.concat([X1,Y1],axis = 1)
df1.corr()
# case 1

Y = Y1["Price"]
X = X1[["Age_08_04"]]
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

""" checking for multi collineaity """

""" VIF (variance influence factor is one of the metric which is used to calculate the relationship between  the two independent variables in order to see
there is a presence of multi collinearity, if exists it will effect the accuracy score of the model. so the vif factor ranges follows as below mentioned
VIF = 1/1-r2 ,VIF < 5 no multi collinearity
VIF : 5- 10  some multi collinearity issues will be present but can be accepted
VIF > 10 not at all acceptable """)    
 
Y = X1["Age_08_04"]
X = X1[["KM"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  # 1.3435521293062282 can be considered

Y = X1["Age_08_04"]
X = X1[["Weight"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)   # 1.283924589019199 can be considered

Y = X1["KM"]
X = X1[["Weight"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)   # 1.0008185411887267 can be considered


Y = Y1["Price"]
X = X1[["Age_08_04","KM","Weight"]]
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

Y = X1["HP"]
X = X1[["Weight"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["KM"]
X = X1[["HP"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["Age_08_04"]
X = X1[["HP"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)



Y = Y1["Price"]
X = X1[["Age_08_04","KM","Weight","HP"]]
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

Y = X1["Quarterly_Tax"]
X = X1[["HP"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["Quarterly_Tax"]
X = X1[["KM"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["Quarterly_Tax"]
X = X1[["Weight"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["Quarterly_Tax"]
X = X1[["Age_08_04"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)


Y = Y1["Price"]
X = X1[["Age_08_04","KM","Weight","HP","Quarterly_Tax"]]
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


Y = X1["Doors"]
X = X1[["KM"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["Doors"]
X = X1[["Age_08_04"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["Doors"]
X = X1[["Weight"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["Doors"]
X = X1[["HP"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["Doors"]
X = X1[["Quarterly_Tax"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)


Y = Y1["Price"]
X = X1[["Age_08_04","KM","Weight","HP","Quarterly_Tax","Doors"]]
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

Y = X1["cc"]
X = X1[["Age_08_04"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["Doors"]
X = X1[["cc"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["cc"]
X = X1[["Quarterly_Tax"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["cc"]
X = X1[["HP"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["cc"]
X = X1[["KM"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = X1["cc"]
X = X1[["Weight"]]
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)

Y = Y1["Price"]
X = X1[["Age_08_04","KM","Weight","HP","Quarterly_Tax","Doors","cc"]]
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



""" as the Gears column consists of less correlation with all other independent variables multi collinearity doesnt exists
between any of them so we can consider Gears into model fitting"""

Y = Y1["Price"]
X = X1[["Age_08_04","KM","Weight","HP","Quarterly_Tax","Doors","cc","Gears"]]
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

""" This above model is the best model for the given dataset with low rmse as 0.369 and R2 score of 86%#"""

# Residual Analysis
#fit the model with seaborn,statsmodels package
import pandas as pd
df_residual = pd.read_csv("C:\\Users\\dell\\Downloads\\ToyotaCorolla.csv",encoding = "latin")
df_residual

#format the plot background as scatter plots for all variables
import seaborn as sns
sns.set_style(style="darkgrid")
sns.pairplot(df1)

#build a model
import statsmodels.formula.api as smf
model = smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=df1).fit()
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
    'Price~Age_08_04',
    'Price~Age_08_04+KM',
    'Price~Age_08_04+KM+Weight',
    'Price~Age_08_04+KM+Weight+HP',
    'Price~Age_08_04+KM+Weight+HP+Quarterly_Tax',
    'Price~Age_08_04+KM+Weight+HP+Quarterly_Tax+Doors',
    'Price~Age_08_04+KM+Weight+HP+Quarterly_Tax+Doors+cc',
    'Price~Age_08_04+KM+Weight+HP+Quarterly_Tax+Doors+cc+Gears',
]

for model_formula in models:
    model = smf.ols(model_formula, data=df1).fit()
    rsquared = model.rsquared
    rsquared_values.append(rsquared)

# Create a DataFrame to display the R-squared values
model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4','Model 5', 'Model 6', 'Model 7', 'Model 8']
rsquared_df = pd.DataFrame({'Model': model_names, 'R-squared': rsquared_values})

print(rsquared_df)

#Model 8 has best r-square value= 0.850697
