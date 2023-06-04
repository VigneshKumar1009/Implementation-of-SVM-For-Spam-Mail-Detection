# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset using chardet
2. Get dataset info and check for null values
3. Assign x and y values and split the dataset into training and testing sets
4. Import CountVectorizer and transform x_train,x_test as vectors
5. Import SVC and fit it to dataset
6. Find y predict and accuracy

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Vigneshkumar V
RegisterNumber: 212220220054  
*/
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy



## Output:
RESULT OUTPUT:

![image](https://github.com/VigneshKumar1009/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113573894/bb3e9968-f369-4ad8-b138-b7ba0fdfdbf5)

DATA.HEAD():

![image](https://github.com/VigneshKumar1009/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113573894/e3d893cd-b483-42b3-9684-6c81a8d25868)

DATA.INFO()

![image](https://github.com/VigneshKumar1009/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113573894/5574a704-abe7-48a5-8cbb-26acbad74727)

DATA.ISNULL().SUM():

![image](https://github.com/VigneshKumar1009/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113573894/0e6fcf8d-f00a-49ac-a89b-cfc1efc816b7)

Y_PREDICTION VALUE:

![image](https://github.com/VigneshKumar1009/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113573894/3dacab46-aa17-4403-b10a-0f0a9e579314)

ACCURACY VALUE:

![image](https://github.com/VigneshKumar1009/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113573894/5b3b6747-7d91-4e85-a443-5f48aa488b42)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
