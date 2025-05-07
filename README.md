# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
```
```


df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/79149bbb-6f55-4962-bbc8-3ce7bf6e9696)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/9662779d-a5ea-4f7b-a1ba-565688ce990a)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/cf7d445f-cf75-4557-9d14-512873372c3e)
```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals

```
![image](https://github.com/user-attachments/assets/186ef359-2363-40b7-afcc-59c4a6cb4a1e)
```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()


```
![image](https://github.com/user-attachments/assets/08d10462-4bb3-4390-8a48-ee5a368495ac)

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)

```
![image](https://github.com/user-attachments/assets/53fa6a47-1e21-4b41-b08c-049e90217919)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/be1ba289-6afe-4a0a-9be9-63cd389b3dd7)
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
```
![image](https://github.com/user-attachments/assets/7addb2c4-9fc4-4009-b4c2-5b2afec1d8f2)
```
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/a55114c0-2d01-4d1e-a1bd-39e642d0613b)

```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("/content/bmi.csv")
df4.head()

```
![image](https://github.com/user-attachments/assets/36eb06ee-5f29-45d1-8e17-ec808a739b38)

```
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/user-attachments/assets/4d2416f4-8f86-4b4f-80cf-753a9e32d606)
```
import pandas as pd
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![image](https://github.com/user-attachments/assets/dce96fb9-6c75-41f8-b6c9-6fc9d7d1197d)

```
df
```
![image](https://github.com/user-attachments/assets/1df01f8a-5849-4441-b55b-19a575e6932b)

```
df.info()
```
![image](https://github.com/user-attachments/assets/e97adee1-0306-451a-92a4-fdbc9cacc69f)

```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/05657f48-bcaa-4679-9b09-f1c2191c3ecb)
```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/d5128368-79e7-4f3e-9ae1-9b1d85ed7413)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]


```
![image](https://github.com/user-attachments/assets/407d4555-82e5-4a62-8c5a-df0f004dad11)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/59bf0322-8112-4249-aaec-bc65a6decbab)
```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/d3d9b6ad-5fd6-48fa-9332-94661e0bab72)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]

```
![image](https://github.com/user-attachments/assets/5724662f-87e5-49ee-a6d3-c31d456840e1)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/e9e7a865-9f8a-478c-88a8-436624fd8a84)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```

![image](https://github.com/user-attachments/assets/fef500a4-eeb5-468c-9f2f-76ba1c7cb362)

```
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
```

![image](https://github.com/user-attachments/assets/92553ae8-7d6d-4bb4-8c55-371563b5ae9f)
```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```

![image](https://github.com/user-attachments/assets/28648db3-1956-402b-b15d-f902083ac3c7)

```
# @title
!pip install skfeature-chappers
```

![image](https://github.com/user-attachments/assets/26e4ba0f-016e-42af-b4da-edf95aaf4dba)

```
# @title
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# @title
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')

# @title
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/22d16557-b19b-4e38-bd96-68d9c1c317a0)

```
# @title
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/2deeb6c3-f2b2-4218-aaff-838143ac436c)

```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)

```

![image](https://github.com/user-attachments/assets/da06e6ff-b945-47d4-949a-a54ad2a59450)
```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```


![image](https://github.com/user-attachments/assets/ae7926f4-3b81-49ed-8f09-04c4cd19869d)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/baad27ba-6295-4042-a1dc-1e9f3ce58690)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select = 6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)

```

![image](https://github.com/user-attachments/assets/6cc11804-e529-4cc6-86c8-e870ce515d69)

![image](https://github.com/user-attachments/assets/8bd1a81c-2e5d-4167-995b-9b3688afde41)
```
selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)

```

![image](https://github.com/user-attachments/assets/dae73b6b-d954-47a9-a528-a08accd63a9f)

```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using Fisher Score selected features: {accuracy}")

```

![image](https://github.com/user-attachments/assets/391f2baf-c6f1-4982-9126-55074fc66f23)
# RESULT:
Thus,Feature selection and Feature scaling has been used on the given dataset.
