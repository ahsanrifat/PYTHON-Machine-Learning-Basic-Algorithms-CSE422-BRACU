import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt

"""Task 01 --> Reading The File"""
print("Reading CSV File")
diabetes=pd.read_csv("diabetes.csv")
#print(diabetes)

"""Task 02 --> Encode Outcomes As They Are Not Numerical"""

df=pd.DataFrame(diabetes)
#print(df)
enc = LabelEncoder()
enc.fit(df['Outcome'])
df['Outcome'] = enc.transform(df['Outcome'])
#print(df)


""" Task 03 ---> Divide The Frame As Data And Out Come """
"""
print(df.head(0))
df1=df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]].copy()
df2=df[["Outcome"]].copy()
print(df1)
"""

"""Task 04 ---> Visuaizing throgh plotting"""
df.plot(kind='box',subplots=True,layout=(9,9))
plt.show()
df.hist()

print("Task05 ---> Extract basic statistical features")
binarized= preprocessing.binarize(diabetes)
print("Mean Of The Given Data")
#print(diabetes.mean(axis=0))
print("Standered Daviation of given data")
#print(diabetes.std(axis=0))