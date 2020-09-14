import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('accidents_india.csv')
df.head()
df.columns[df.isna().any()]
df.Sex_Of_Driver = df.Sex_Of_Driver.fillna(df.Sex_Of_Driver.mean())
df.Vehicle_Type = df.Vehicle_Type.fillna(df.Vehicle_Type.mean())
df.Speed_limit = df.Speed_limit.fillna(df.Speed_limit.mean())
df.Road_Type = df.Road_Type.fillna(df.Road_Type.mean())
df.Number_of_Pasengers = df.Number_of_Pasengers.fillna(df.Speed_limit.mean())

df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)
df.columns[df.isna().any()]

pd.unique(df['Accident_Severity'])
df.dropna(inplace = True)

df.columns[df.isna().any()]
#LabelEncoding
c = LabelEncoder()
df['Day'] = c.fit_transform(df['Day_of_Week'])
df.drop('Day_of_Week', axis=1, inplace=True)
l = LabelEncoder()
df['Light'] = l.fit_transform(df['Light_Conditions'])
df.drop('Light_Conditions', axis=1, inplace=True)
s = LabelEncoder()
df['Severity'] = s.fit_transform(df['Accident_Severity'])
df.drop('Accident_Severity', axis=1, inplace=True)
df.head() 
from sklearn.model_selection import train_test_split
x = df.drop(['Pedestrian_Crossing', 'Special_Conditions_at_Site', 'Severity'], axis=1)
y = df['Severity']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.76)


from sklearn.tree import DecisionTreeClassifier
reg = DecisionTreeClassifier(criterion='gini')
reg.fit(x_train, y_train)


reg.score(x_test, y_test)
pd.unique(y)

import numpy as np
import pickle
inputt=[int(x) for x in "2 10 201 10 10 8 3".split(' ')]
final=[np.array(inputt)]
b = reg.predict(final)
pickle.dump(reg,open('test1.pkl','wb'))
test=pickle.load(open('test1.pkl','rb'))

