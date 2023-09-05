# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MUKESH P
RegisterNumber:  212222240068
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

df.head()

![ML21](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393818/9647e6dd-b756-4c7c-9d2c-2495653fb7e1)

df.tail()

![ML22](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393818/c0a59b09-060d-49c8-ba2f-f8b40e51d0f5)

Array value of X

![ML23](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393818/aaaa787e-7ee5-45c6-9f33-d040965f1170)

Array value of Y

![ML24](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393818/8df4bd73-81c0-4159-8440-e06420a149ef)

Values of Y prediction

![ML25](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393818/5784ee5f-2362-4928-ba86-299ec5b9e723)

Array values of Y test

![ML26](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393818/aa8fcdab-ec11-4e84-8ead-5ebd3c8e4a1e)

Training Set Graph

![ML27](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393818/4a48a766-3400-4a59-99a8-f3c0dc8aa6e2)

Test Set Graph

![ML28](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393818/16cdee19-331e-4daf-a740-3d40e7fcd68b)

Values of MSE, MAE and RMSE

![ML29](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393818/a1ccf278-b3a8-4722-8b2d-34c2a905cfb6)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
