# Customer_Churn_Prediction_-_Sales_Dashboard_Project

churn_mapping = {0: 'No', 1: 'Yes'}
from IPython.core.debugger import skip_doctest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost
import joblib

df = pd.read_csv("/content/Customer Churn.csv")
df.isnull().sum()
df.drop_duplicates(inplace=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.select_dtypes(include='object'):
  df[col] = le.fit_transform(df[col])

  import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Churn', data=df)
plt.show()

sns.histplot(df['MonthlyCharges'])
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.show()

X = df.drop('Churn', axis=1)
y = df['Churn']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

pred_lr = lr.predict(X_test)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators = 200,
    random_state = 42
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators = 200,
    random_state = 42
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

pred_xgb = xgb.predict(X_test)

from sklearn.metrics import accuracy_score

print("LR:", accuracy_score(y_test,pred_lr))
print("DT:", accuracy_score(y_test,pred_dt))
print("RF:", accuracy_score(y_test,pred_rf))
print("XGB:", accuracy_score(y_test,pred_xgb))

import joblib

joblib.dump(rf,'customer_churn_model.pkl')

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)

df['Cluster'] = kmeans.fit_predict(X)

sns.scatterplot(
    x='MonthlyCharges',
    y='TotalCharges',
    hue='Cluster',

    data=df
)
plt.show()

sales = df.groupby('tenure')['TotalCharges'].sum()

sales.plot(kind='line')
plt.title("Sales Trend")
plt.show()

sns.barplot(
    x='Contract',
    y='TotalCharges',
    data=df
)
plt.show()





