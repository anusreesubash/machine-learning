#import necessary libraries 
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# load Diabetes dataset from sklearn
# bmi-body mass index, tc-type of white blood cells, ldl-low density lipoprotiens, hdl-high density lipoprotien, tch-thyroid stimulating hormone,ltg-lamotrigine, glu-blood sugar level
columns="age sex bmi map tc ldl hdl tch ltg glu".split() 
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=columns)
y = diabetes.target #define the target variable(dependent variable) as y


# create training and testing variables
X_train, X_test, Y_train, Y_test = train_test_split(df, y, test_size=0.2)
# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, Y_train)
predictions = lm.predict(X_test)

# print(predictions[0:5])

# plot the model:
plt.scatter(Y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("predictions")
plt.show()
print('score:', model.score(X_test, Y_test))