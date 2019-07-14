import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 




"""Reading Dataset"""
dataset=pd.read_csv('StockPriceData.csv')
print(dataset.shape)
print(dataset.head())
print(dataset.describe())



print("""-----Ploting in normal graph-----""")
dataset.plot(x='Open', y='Close', style='o')
plt.title('Open vs Close')
plt.xlabel('Open')
plt.ylabel('Close')
plt.show()




"""Making a new dataframe With Open and Close for Linear Regression"""
dataset_new=dataset[['Open', 'Close']].copy()
print(dataset_new.head(5))
"""Cleaning the dataset from all null values"""
dataset_new = dataset_new.dropna(how='any',axis=0)



"""Attributes(x) Label(y)"""
X = dataset_new.iloc[:, :-1].values
y = dataset_new.iloc[:, 1].values



"""Dividing train and test data"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



"""Train The Model"""
regressor = LinearRegression()
linreg=regressor.fit(X_train, y_train)
print("""-----Plotting the regression-----""")
plt.figure(figsize=(5,4))
plt.scatter(X_train, y_train, marker= 'o', s=10, alpha=0.8)
plt.plot(X_train, linreg.coef_ * X_train + linreg.intercept_, 'r-')
plt.title('Least-squares linear regression')
plt.xlabel('Feature value (x)')
plt.ylabel('Target value (y)')
plt.show()




print("""-----intercept val and coef val-----""")
print(regressor.intercept_)
print(regressor.coef_)



"""Predict on test datas"""
y_pred = regressor.predict(X_test)



print("""-----Accurecy of the prediction-----""")
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)



print("""---------Accuracy Testing--------""")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))