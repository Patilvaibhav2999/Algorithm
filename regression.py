# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 12:36:05 2021

@author: acer
"""


# Regression:-

# Types of ML :- 1) Supervised - When you have known the output column. or  data structuring & analysis teaching to the machine code of training & testing.
#             Classification Algorithm -Categorical data - 'yes or no' , ' 0 or 1'.
#             Regression Algorithm - Continues data such like a Salary , Premium
#                2)  Non-Supervised - When you don't known output column.  or data structuring & analysis  doesn't teaching to the machine code of training & testing.

#x=independent/actual varible, y=dependent/predict varible, m & c  as like a & b is coefficient.
# fornmula :- Y=mx+c,   Y=ma+b


import pandas as pd
import matplotlib.pyplot as plt

#Height
x=[[151],[174],[138],[186],[128],[136],[179],[163],[152],[131]] #Two dimensional value of numpy package.

#Weight
y=[63,81,56,91,47,57,76,72,62,48]


from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x, y)

y_pred=lr.predict([[175]])

plt.scatter(x, y, color='green',lw=5)
plt.plot(x, lr.predict(x), color='m', lw=4)

print("Weight is:", y_pred)
print("accuracy is", lr.score(x, y)*100) # Score is function used to show accuracy.



import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

dataset=pd.read_csv("Salary_Data.csv")
dataset.head()
dataset.shape
dataset.head()

sn.heatmap(dataset.corr(), annot=True) # "corr" is function for Co-Relation     and "annot" is to display CoRelation values.
plt.show()

x=dataset.loc[:, "YearsExperience"].values

x=x.reshape(-1, 1) #here we reshape  the 1 dimenasional array to 2nd arry
x.shape
y=dataset.iloc[:, 1].values # here 1 is index no of salary & 0 is index no of of yearexperince
y.shape

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.33, random_state=0)
print(x_train.shape)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)  #predict the values

df=pd.DataFrame({"actual":y_test, "pred":y_pred})

print(df)

print("accuracy:", lr.score(x,y)*100)


# Errors

# 1) Mean Absolute Error (MAE) :-absolute difference between the true value/ data and predicted value/ data.
# formaue: MAE = True values – Predicted values

# 2) Mean Squared Error (MSE) :-Square of difference between true value/data and predicted value/data.

#Formulae:   MSE = 1/N  (True value - predicted value)^2



# 3) Root Mean Squared Error (RMSE) :-This is standard deviation error..this is same as mean squared error &  which be occur when determining accuracy of dataset.

# 4) R Squared:-
#  It is also known as the coefficient of determination. R squared value lies between 0 and 1. the 0 is not fit in given data whereas 1 is fit perfectly in given data.
#Q. Why we use this error?
# Ans:- beacuse Accuracy in given data of dataset. How much accuracy will be obtained in given data to show by error concept.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 score:", r2_score(y_test, y_pred))
plt.scatter(x_train, y_train, color="r")

plt.title("Salary vs yr of Exp")
plt.xlabel("yr of exp")

# Coefficient
lr.coef_
# intercept
lr.intercept_
# Y=ax+b manually calculate the predicted value 
y_manual=(9345.94244312*5)+26816.19224403119    #x=5 here 5=year of expreience
print(y_manual)
plt.ylabel("Salary")
plt.plot(x_train, lr.predict(x_train), "-")
plt.show()

# MULTIPLE REGRESSIION :-
# To upload Advertising csv file
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
dataset=pd.read_csv("Advertising.csv")
dataset

sn.heatmap(dataset.corr(), annot=True)
plt.show()

dataset.shape

dataset.corr()

x=dataset.loc[:, ["TV", "radio", "newspaper"]]
print(x)

y=dataset["sales"]
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=0)
x_train.shape

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)
y_pred=lr.predict(x_test)

df=pd.DataFrame({"actual":y_test, "pred":y_pred})
print(df)

lr.score(x, y)*100
new=np.array([45,26,19])
y_pred_new=lr.predict([new])
y_pred_new
lr.coef_
lr.intercept_

# petrol consume
# displ, horspower & weight are independent and petrol are dependent.

# Linear Regression’, ‘Polynomial Regression’, ‘Logistic regression’ 



# Polynomial Regression 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Salary_Data.csv")
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.33, random_state=0)

from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=2)  # degree2 = is nothing but polynomial features always greater than 1.
                               #degrre=1 = is under fitting thats the poor result of accuracy of training & testing.
                               #degree=2= is base fitting that the middle result  of accuarcy of training & testing.
                               #degreee=4= is over fitting that the better result of accuarcy of training & testing.
                               
x_poly=poly.fit_transform(x)
x_train_poly=poly.fit_transform(x_train)

x_test_poly=poly.fit_transform(x_test)

from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(x_train_poly, y_train)

y_pred=reg.predict(x_test_poly)

reg.score(x_poly, y)*100


plt.plot(x_poly, y, color="red")

plt.plot(x_poly, reg.predict(x_poly), "-", color="blue") # by default the blue kr yenar.

plt.title("Training set")

plt.xlabel("Yr of Exp")

plt.ylabel("salary")

plt.show()

#degree=1 = liya to linear regression grapgh show/ under fitting and training & testing is poor accuracy.
# degree=2= base  fiiting = testing & training is middle accuracy result shiow.
# degree=4= over fitting =testing & training is excellent accuracy reuslt show.

#degree=1= all point don't be match, hence result of  accuracy is poor.
#degree=2= near about try to match all point , hence result of accuracy is good.
#degree=4= all point  be match, hence result of accuracy is best.





# Logistic Regression:-

# 1) All vales are 0 & 1., True-False
# 2) ALl the categorical data used it.
# 3) error of  data is nothing but confusin matrxi.

dataset=pd.read_csv("Social_Network_Ads.csv")
dataset.shape

x=dataset.iloc[:, 2:4].values

y=dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.20, random_state=0)

dataset.head(3)

from sklearn.preprocessing import scale

x_train=scale(x_train)

x_test=scale(x_test)

from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0)

classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

df=pd.DataFrame({"actual":y_test, "pred":y_pred})
print(df)

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score 
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred)*100)

#                                  perdicted value

#                     Not  purchased            purchased      Support

#                        52                       6            = 58
#                         3                       19           =22
# actaual value 



#--------------------------------------------------------------------------------------------------------------------------------------------------------

#                       1.Single Linear Regression Algorithm
# Single Linear Regression Algorithm:- it defined as only one independent and dependent varibles.

#    import libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
dataset=pd.read_csv('salary_data.csv')
print(dataset)

plt.scatter(dataset['YearsExperience'], dataset['Salary'])
dataset.head()

# input & output data
x=dataset.iloc[:, : -1].values #= array chahiye tabhi values used krna.
print(x)
y=dataset.iloc[:, -1:].values
print(y)
x[0:5]
y[0:8]

# splitting the dataset  training and testing dataset 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y, test_size=0.20, random_state=23)
x_train
x_test
x_train.shape 
x_test.shape 
y_train
y_test
y_train.shape 
y_test.shape 


#fitting simple linear regression to the triaining set 
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)
       
# predicting 
y_pred=lr.predict(x_test)
print(y_pred)

# actual answer
print(y_test)

# visulisaing part for training dataset

plt.scatter(x_train, y_train, color='r')
plt.plot(x_train, lr.predict(x_train), color='blue')


#  testing part visulaization 
plt.scatter(x_test, y_test, color='r')
plt.plot(x_train, lr.predict(x_train), color='m')
plt.xlabel('exp')
plt.ylabel('Salary')
plt.title('Salary vs Year of Exp')
plt.show()

print('accuracy:', lr.score(x, y)*100)
print('Salary is :', y_pred)





import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data=pd.read_csv('salary_data.csv')
print(data)

data.head()

data.tail()

data.shape 

data.describe()

print(data.info())

data.isnull().sum()

plt.scatter(data['YearsExperience'], data['Salary'])

x=data.iloc[:, : 1].values
print(x)

y=data.iloc[:, 1 :].values
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20, random_state=0)
print(x_train)

x_train.shape 

print(x_test)

x_test.shape 

print(y_train)

y_train.shape 

print(y_test)

y_test.shape 



from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)
print(y_pred)

# training data
plt.scatter(x_train, y_train, color='m', label='trainig data')

plt.plot(x_train, lr.predict(x_train), color='r')

#testing data
plt.scatter(x_test, y_test, color='green', label='testing data')

plt.plot(x_train, lr.predict(x_train), color='orange',lw=3, label='training')
plt.title('Year of Exp vs Salary')
plt.xlabel('Salary')
plt.ylabel('Year of Exp')
plt.legend()
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data1=pd.read_csv('petrol_consume.csv')
print(data1)


data1.head()

data1.iloc[:, 0:1]



x=data1.iloc[:, 0:1]
print(x)

y=data1.iloc[:, 4 :]
print(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=0)
print(x_train)

print(x_train.shape)

print(y_train)

print(y_train.shape)

print(x_test)

print(y_test.shape)

print(x_test)
print(x_test.shape)


print(y_test)

print(y_test.shape)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)
print(y_pred)


# trainig data
plt.scatter(x_train, y_train, color='red', label='training data')

plt.plot(x_train, lr.predict(x_train), color='m', label='training result')


# testing the data

plt.scatter(x_test, y_test, color='blue', label='testing data')

plt.plot(x_train, lr.predict(x_train), color='black', label='Regressor line')
plt.title('Petrol Ratio')
plt.xlabel('Petrol_Consumption')
plt.ylabel('Petrol_tax')
plt.legend(loc='upper right', shadow=True, fontsize=12)
plt.show()

print('Petrol_Consumption:', y_pred)
print('Accuracy :', lr.score(x, y)*100)




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
dataf=pd.read_csv('Support-Vector-Regression-Data.csv')
print(dataf)

dataf.shape

dataf.head()

dataf.tail()

dataf.describe()

dataf.info()

plt.scatter(dataf['x'], dataf['y'], color='black')


x=dataf.iloc[:, :1]
print(x)

y=dataf.iloc[:, 1 :]
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=0)
print(x_train)
x_train.shape 

print(x_test)
x_test.shape 

print(y_train)
y_train.shape 

print(y_test)
y_test.shape 


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)
print(y_pred)

# training the data
plt.scatter(x_train, y_train, color='red', label='traing data')

plt.plot(x_train, lr.predict(x_train), color='m', label='traing result', lw=6)

# testing the data

plt.scatter(x_test, y_test , color='green', label='testing data')

plt.plot(x_train, lr.predict(x_train), color='cyan', lw=3, label='testing result')

plt.legend(loc='lower right', shadow=True, fontsize=10)

plt.show()


print('unknown predict value is', y_pred)
print('Accuracy:', lr.score(x, y)*100)




#              2) Multiple Linear Regression Algorithm
#  Multiple Linear Regression Algorithm :- it defined as multiple independent variables and only one dependent variable.


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv('insurance.csv')
print(data)


data.describe()

data['region'].value_counts()

data=data.drop('region', axis=1)   #axis=1 that is horizontal column extract krega / axis=0 that is  vertical column krega.
data.head()

#slicing
x=data.iloc[:, : -1]
print(x)

y=data.iloc[:, -1 :]
print(y)

# x1 continues data
x1=data.drop(['sex', 'children', 'smoker'], axis=1)
print(x1)


x1.head()

# x2 categorical data

x2=data.drop(['age', 'bmi', 'expenses'],axis=1)
print(x2)

x2.head()



#Standardization of continues data
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x1=sc.fit_transform(x1)
print(x1)

x1=pd.DataFrame(x1)
print(x1)

x1=x1.rename(columns={0:'age', 1:'bmi', 2:'expenses'})
print(x1.head())



# LabelEncoding on categorical data

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x2['sex']=le.fit_transform(x2['sex'])
print(x2)

x2['smoker']=le.fit_transform(x2['smoker'])
print(x2.head())


# training , testing & spitting
X=pd.concat([x1,x2], axis=1)
X.head()

print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=2)
x_train

print(x_train.shape)

print(x_test.shape)

x_train.head()


print(y_train)
print(y_train.shape)

print(y_test.shape)

y_train.head()


# fit the model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

# prediction
y_pred=lr.predict(x_test)
print(y_pred)

# Actual output
y_test

y_pred1=lr.predict([[-1.438764, 0.93052, -0.960225,1,0,0]])
print(y_pred1)

plt.scatter(x_test['age'], y_test, color='red', label='age')

plt.scatter(x_test['bmi'], y_test , label='bmi')

plt.scatter(x_test['children'], y_test, label='children')
plt.legend(loc='lower right')
plt.show()
plt.plot(x_train, lr.predict(x_train), color='m', lw=4)






#----------------------------------------------------------------------------------------------------------

#r2_score=0.7978
#by separating continues(standard scaling) and catagorical(label_encoding)
#then spliting and hence apply model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv('insurance.csv')
data.head()

data['region'].value_counts()

data.describe()

data.corr()

data.head()

data=data.drop(['region'],axis=1)

data.head()

data['children'].value_counts()

x=data.iloc[:,:-1]
print(x)
y=data.iloc[:,-1:]

y.head()


x1=x.drop(['sex','children','smoker'],axis=1)
x2=x.drop(['age','bmi'],axis=1)
x1.head()
x2.head()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x1=scaler.fit_transform(x1)

x1=pd.DataFrame(x1)
x1=x1.rename(columns={0:'age',1:'bmi'})
x1.head()

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()

x2=x2.apply(label_encoder.fit_transform)
x2.head()


xx=pd.concat([x1,x2],axis=1)
xx.head()

from sklearn.model_selection import train_test_split as tts
xtrain,xtest,ytrain,ytest=tts(xx,y,test_size=0.2,random_state=0)
print(xtrain)
print(xtrain.shape) 
print(ytrain)
print(ytrain.shape)
print(xtest)
print(xtest.shape) 
print(ytest)
print(ytest.shape)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
model=lr.fit(xtrain,ytrain)
ypred=model.predict(xtest)
print(ypred)


ypred1=model.predict([[0.910875,-0.076355, 1,1,0]])
ypred1

plt.scatter(xtrain,ytrain, color='red',label=' training data')

plt.scatter(xtest, ytest, color='black' , label='testing data')
plt.plot(xtrain, lr.predict(xtrain), color='m')
plt.legend(loc='upper right')

from sklearn.metrics import r2_score
r2_score(ytest,ypred)

ytest.head()

ypred

plt.scatter(xtest['age'],ytest, color='cyan', label='training data')

plt.scatter(xtest['age'],ypred,c='green',label=' training result')

plt.scatter(xtest['bmi'],ytest, color='black', label='testing data')

plt.scatter(xtest['bmi'],ypred,c='red', label='testing result')

plt.plot(xtrain, lr.predict(xtrain), color='m', ls='--', alpha=0.18)
plt.legend(loc='upper right')
plt.show()

#---------------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
data=pd.read_csv('Salary_Data.csv')
print(data)

data.info()

data.describe()

x=data.iloc[:, : -1]
print(x)

y=data.iloc[:, -1 :]
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=0)
print(x_train)
print(x_train.shape)

print(y_train)
print(y_train.shape)

print(x_test.shape)

from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)
print(y_pred)

print('Salary is:', y_pred)
print('Accuracy is:', lr.score(x,y)*100)






plt.scatter(x_train, y_train, color='m',label='training')
plt.plot(x_train, lr.predict(x_train), color='red', lw=8, label='traing data')

plt.scatter(x_test, y_test, color='green', label='tetsting', lw=8)
plt.plot(x_train, lr.predict(x_train), color='black', label='REgression line')
plt.legend(loc='lower right')
plt.show()





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('Advertising.csv')
print(data)


data1=data.drop('Month', axis=1, inplace=True)
print(data1)


x=data.loc[:, ['TV', 'radio', 'newspaper'] ]
print(x)

y=data['sales']
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=0)
print(x_train)


print(x_train.shape)
print(y_train)
print(y_train.shape)

print(y_test)
print(x_test.shape)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)
print(y_pred)


print('Accuarcy is:', lr.score(x,y)*100)

print('sale is:', y_pred)

plt.scatter(x_test['TV'], y_test,color='green', label='TV')
plt.scatter(x_test['radio'], y_test, color='red', label='radio')
plt.scatter(x_test['newspaper'], y_test, color='m', label='nespaper')
plt.plot(x_train, lr.predict(x_train), color='black', lw=4, label='regression line', alpha=0.5)
plt.legend()
plt.show()





















#----------------------------------------------------------------------------------------------------


# Logistic Regression
#   Whenever we data analysis on categorical data to used it.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv('telco.csv')
print(data)

data.head()

data.info()


data.describe()

data=data.drop('customerID', axis=1)
print(data)

# data=data.drop('Customer ID', axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

""" This will use for only one column to label 
 data['gender']=le.fit_transform(data['gender'])

data['partner']=le.fit_transform(data['partner'])"""

# but apply method is used to label the column of whole datasetat a time.(simultanously)
data=data.apply(le.fit_transform)
print(data)

data.head()

data.info()

#x= input
#y = output

x=data.iloc[:, 0:-1]
print(x)

y=data.iloc[:, -1:]
print(y)

# conti =  continues data
x_conti=x[['tenure', 'MonthlyCharges', 'TotalCharges']]
x_conti

x_cat=x.drop(['tenure', 'MonthlyCharges', 'TotalCharges'], axis=1)
x_cat.head()


# Standardization for continues  data x_conti
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_conti=sc.fit_transform(x_conti)
x_conti


### Features_selection Method = chi2 test and selectbest
x_cat.head(3)

x_cat.info()

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

# from sklearn.feature_selection import chi2, SelectKBest

pred_model=SelectKBest(chi2, k=10)  # k= by default value is 10. select the feature = columns.
x_kbest=pred_model.fit_transform(x_cat, y)
x_kbest

x_kbest.shape 

pred_model.get_support()

xx_kbest=pd.DataFrame(x_kbest)
xx_kbest.head()

df=pd.concat([x_conti, xx_kbest], axis=1)
df.head()

print(df)

from sklearn.model_selection import train_test_split
x_train, y_train, x_test, y_test=train_test_split(df,y, test_size=0.20, random_state=0)
x_train.shape 

x_test.shape 

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train, y_train)


# prediction 

y_pred=model.predict(x_test)
y_pred

#Actual result

y_test.head()




# 14 oct 2021

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
data=pd.read_csv('telco.csv')
print(data)


data1=data.drop('customerID', axis=1)
print(data1)


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
m=data1.apply(le.fit_transform)
print(m)



x=m.iloc[:, : -1]
print(x)

y=m.iloc[:, -1 :]
print(y)

data.info()

cd=m[['tenure','MonthlyCharges' ,'SeniorCitizen' ]]
print(cd)

cgd=m.drop(['tenure','MonthlyCharges','SeniorCitizen'], axis=1)
print(cgd)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
z=sc.fit_transform(cd)
print(z)

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
pred_model=SelectKBest(chi2, k=10)
skb=pred_model.fit_transform(cgd, y)
print(skb)

pred_model.get_support()

askb=pd.DataFrame(skb)
print(askb)

df=pd.concat([cd, askb], axis=1)
print(df.head())

print(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(df,y, test_size=0.20, random_state=0)
x_train
x_train.shape 

y_train
y_train.shape 

x_test
x_test.shape 

print(y_test)
y_test.shape


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train, y_train)

# Prediction
y_pred=model.predict(x_test)
print(y_pred)

y_test.head()

#     Model Evaluation   #sklearn=scitet learn

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
print(accuracy_score(y_test, y_pred)*100)

print(confusion_matrix(y_test, y_pred))

print(precision_score(y_test, y_pred))

print(classification_report(y_test, y_pred))




















# Data name = banknotes.csv/file.
# Acuuracy is =52.36363636363637

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('banknotes.csv')
print(data)



x=data.iloc[:, : -1]
print(x)

y=data.iloc[:, -1 :]
print(y)



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
mm=sc.fit_transform(x)
print(mm)



from sklearn.preprocessing import Binarizer
bi=Binarizer(threshold=3.5)
df=bi.fit_transform(mm)
print(df)




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(df,y, test_size=0.20, random_state=2)
print(x_train)


print(x_train.shape)

print(y_test)
print(y_test.shape)

print(y_train.shape)

print(x_test.shape)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)
print(y_pred)


from sklearn.metrics import classification_report, accuracy_score, precision_score, confusion_matrix


print(classification_report(y_test, y_pred))


print(accuracy_score(y_test, y_pred)*100)


print(confusion_matrix(y_test, y_pred))
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV

parameter={'penalty':[11,12, 'elasticnet'], 'solver':['liblinear'], 'ccp_alpha':[0,0.1,0.2]}

lr=LogisticRegression()

lrr=GridSearchCV(lr, parameter, cv=2, verbose=1)
lrr.fit(x_train, y_train)



      
      

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv('insurance.csv')
print(data)
      
dataa=data.drop('region', axis=1 ,inplace=True)
print(dataa)

x=data.iloc[:, : -1] 
print(x)

y=data.iloc[:, -1 :] 
print(y)    
      

cd=data[['age', 'bmi']]
print(cd)

cgd=data[['sex', 'children', 'smoker']]
print(cgd)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
ll=sc.fit_transform(cd)
print(ll)

from sklearn.preprocessing import Binarizer
bi= Binarizer()
ff=bi.fit_transform(ll)
print(ff)

rr=pd.DataFrame(ff)
print(rr)

df=rr.rename(columns={0:'age', 1: 'bmi'})
print(df)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
oo=cgd.apply(le.fit_transform)
print(oo)


ii=pd.concat([df,oo], axis=1)
print(ii)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(ii, y, test_size=0.20, random_state=0)
x_train.shape
x_test.shape

y_train.shape
y_test.shape

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)
print(y_pred)

from sklearn.preprocessing import Binarizer
bii=Binarizer()
sss=bii.fit_transform(y_pred)
print(sss)



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_pred))















#-------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv('Mall_Customers.csv')
print(data)

datt=data.drop('CustomerID', axis=1)
print(datt)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
model=data.apply(le.fit_transform)
print(model)

X=model.loc[:,['Genre', 'Age', 'Annual Income (k$)']]
print(X)

y=model['Spending Score (1-100)']
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X,y, test_size=0.20, random_state=0)
print(x_train)

x_train.shape

x_test
x_test.shape

y_test.shape

y_train.shape

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)
print(y_pred)

print('accuracy is:', lr.score(X,y)*100)
print('Annual income is:', y_pred)

plt.scatter(x_test['Age'], y_test, color='green', label='data')
plt.plot(x_train, lr.predict(x_train), color='m', lw=4, label='regression line', alpha=0.5)
plt.legend(loc='lower right')


  #########  


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('Social_Network_Ads.csv')
print(data)

dataa=data.drop('Gender', axis=1, inplace=True)
print(dataa)

X=data.loc[:, ['User ID','Age','EstimatedSalary']]
print(X)

y=data['Purchased']
print(y)



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
model=X.apply(le.fit_transform)
print(model)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
pre=sc.fit_transform(model)
print(pre)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X,y, test_size=0.20, random_state=0)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)
print(y_pred)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(confusion_matrix(y_test, y_pred))
 
print( accuracy_score(y_test, y_pred)*100)

print( classification_report(y_test, y_pred))







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('wine.csv')
print(data)

x=data.iloc[:, 1 :14]
print(x)

y=data.iloc[:, 0:1]
print(y)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
jj=sc.fit_transform(y)
print(jj)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=0)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)
print(y_pred)

plt.scatter(x_test['alcohol'], y_pred, color='red', label='testing data')
plt.plot(x_train, lr.predict(x_train),color='m', label='regression line', alpha=0.7)
plt.legend()
plt.show()


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print( accuracy_score(x,y)*100)

print('Accuracy is:', lr.score(x,y)*100)
print('class is:', y_pred)



import pandas as pd
import numpy as np
data=pd.read_csv('mtcars.csv')
print(data)

xx=data.drop('cyl', axis=1)
print(xx)

x=xx.iloc[:, 0 : 6]
print(x)

y=data.iloc[:, 1 : 2]
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20 , random_state=2)
print(x_train.shape)

print(y_test.shape)

print(x_test.shape)

print(y_train.shape)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
model=lr.fit(x_train, y_train)
print(model)

y_pred=lr.predict(x_test)
print(y_pred)

print('accuracy is:', lr.score(x,y)*100)





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
data=pd.read_csv('iris.csv')
print(data)

data.head()

sn.pairplot(data, hue='species', size=1.5);


sn.heatmap(data.corr(), annot=True)
plt.show()




import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
data=pd.read_csv('C://Users/acer/Downloads//adult_dataset.csv')
print(data)

sn.heatmap(data.corr(), annot=True)
plt.show()




