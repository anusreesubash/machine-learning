import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('data/Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:])
x[:, 1:] = imputer.transform(x[:, 1:])

# print(x)
# print(y)

# Encoding categorical data using One-Hot encoder
# Encode Independent variable

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Encode Dependent variable
le = LabelEncoder()
y = np.array(le.fit_transform(y))

# print(x)
# print(y)

# Split into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# feature scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)


