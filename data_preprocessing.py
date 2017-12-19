'''first, create a simple data frame from a csv to better grasp the problem'''

import pandas as pd
from io import StringIO
csv_data = "A,B,C,D\n1.0,2.0,3.0,4.0\n6.0,7.0,,8.0\n10.0,11.0,12.0,\n"
df = pd.read_csv(StringIO(csv_data))
print(df)


'''dealing with missing values'''
#print(df.isnull().sum()) #use isnull to find elements where data is missing
#print(df.values)
#print(df.dropna()) #drops rows with missing data
#print(df.dropna(axis=1))
df.dropna(how='all') # drops rows where all cols are missing data
df.dropna(thresh=4) #drops rows where at least 4 cols are missing data
df.dropna(subset=['C']) #drops rows where data is missing in the cols indedicated by subset


'''interpolation techniques'''
#mean imputation: replace missing value with mean of entire feature col
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0) #other strategies include median, and most_common (mode)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)

'''handling categorical data'''

#create another dataframe to mess around with
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

'''manually define mapping for ordinal data'''
#we decide to map sizes to integers since size is ordinal data
size_mapping = {
    'XL' : 3,
    'L':2,
    'M':1
}
df['size'] = df['size'].map(size_mapping)
print(df)
inv_size_mapping = {v:k for k, v in size_mapping.items()} #in case we want to transform int values back to strings at some point in the future

'''enumerate unique categorical values to map categories to integers'''
#encoding class labels: since datais not ordinal, we can can simply enumerate them
import numpy as np
class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)

'''or simply use sklearn's LabelEncoder to achieve the same'''
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)

'''use sklearn's onehotencoder to utilize one hot encoding'''
from sklearn.preprocessing import OneHotEncoder
ohe = 