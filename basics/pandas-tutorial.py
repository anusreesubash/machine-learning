import pandas as pd
import numpy as np

# create a series
s = pd.Series([1, 2, 3, 4, np.nan, 6])
print(s)

# print dates
date = pd.date_range('20200803', periods=10)
print(date)

# creating a dataframe
dataframe = pd.DataFrame(np.random.randn(10, 4), index = date, columns = ['A', 'B', 'C', 'D'])
print(dataframe)

# dataframe using different objects
df = pd.DataFrame({ 'A': [1, 2, 3, 4],
	'B':pd.Timestamp('20200803'),
	'C': pd.Series(1, index=list(range(4)), dtype='float32'),
	'D': np.array([5]*4, dtype='int32'),
	'E': pd.Categorical(['True', 'False', 'False', 'True'])

	}) 

 # to get the data types
typ = df.dtypes
print(typ)

df.head() # gives first five rows
df.tail() # gives last five rows
df.index() # gives indexing column
df.columns() # gives column heads
df.to_numpy() # covert data into numpy
df.describe() # gives count, mean, standar deviation, etc...
df.sort_index(axis=1, ascending=false)