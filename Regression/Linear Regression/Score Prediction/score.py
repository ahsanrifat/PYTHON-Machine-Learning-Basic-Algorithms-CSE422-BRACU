import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn import metrics 

"""Reading Dataset"""
dataset=pd.read_csv('student_scores.csv')
print(dataset.shape)
print(dataset.head())
print(dataset.describe())
print("""-----Ploting in normal graph-----""")
dataset.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()
"""Attributes(x) Label(y)"""
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values 
"""Dividing train and test data"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
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