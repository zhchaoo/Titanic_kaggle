import numpy as np
import pandas as pd

df = pd.read_csv('./data/train.csv',header=0)

##show the basic properties of the raw data
df.info()
##show statistics for properties:count,mean,std,min,max
print(df.describe())

# print(df.head(5))

















