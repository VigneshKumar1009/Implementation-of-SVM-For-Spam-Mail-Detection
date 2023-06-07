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

*/
```

## Output:

## RESULT OUTPUT:
![image](https://github.com/srimathi-25/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114581999/6e5f3037-b7a5-490c-bc75-efed4cd2994f)
## DATA.HEAD():
![image](https://github.com/srimathi-25/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114581999/52b8aafd-e3ce-49f0-ac1e-a41acda4e17c)
## DATA.INFO()
![image](https://github.com/srimathi-25/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114581999/f327e2ef-bd0f-442b-9b1e-87eeebc7d143)
## DATA.ISNULL().SUM():
![image](https://github.com/srimathi-25/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114581999/92fbf9da-f37d-4ba5-8e46-02c316046a9b)
## Y_PREDICTION VALUE:
![image](https://github.com/srimathi-25/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114581999/dd583d3a-8b77-4538-b307-60851c164ca3)
## ACCURACY VALUE:
![image](https://github.com/srimathi-25/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114581999/28f19c6f-2def-44fa-9c00-c37f9b7377ed)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
