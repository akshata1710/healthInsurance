import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("healthcare.csv")

#understanding the data
print(data.head())
print(data.shape)

print(data.describe())

print(data.columns)

print(data.nunique())

print(data.isnull().sum())

features=data.drop(['Provider Id','Provider Street Address','Provider State','Provider City','Provider Zip Code'],axis=1)

print(features.head())

#relationship analysis
corre=features.corr()
sns.heatmap(corre,xticklabels=corre.columns,yticklabels=corre.columns,annot=True)

plt.show()

sns.pairplot(features)
plt.show()
