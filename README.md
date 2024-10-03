## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/b9cb0898-783e-460a-be8d-883bac83537f)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/53adf05b-a633-4158-ae6b-3111237aaa8b)

### df['bo2']=e1.fit_transform(df[["ord_2"]])
![image](https://github.com/user-attachments/assets/6dd4ac25-754d-4e59-bde0-bac8e66c84dc)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/a44a9fd1-13c9-49ac-aa9b-40fade264292)

```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)  # Change 'sparse' to 'sparse_output'
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2 = pd.concat([df2, enc], axis=1)
df2
```
![image](https://github.com/user-attachments/assets/d3b783ca-fc55-4383-bbcf-c4ded293b7a5)

### pip install --upgrade category_encoders
![image](https://github.com/user-attachments/assets/f894ac0a-cd58-49c8-bafa-7d3005ad9b90)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
```
![image](https://github.com/user-attachments/assets/e0031e7a-cc82-49f2-8067-67eb3c2ee056)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/656afbd2-d89b-400c-9b63-f20b79496e5d)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/33242455-ad6a-458a-9235-b6244b9019a9)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/eaceb39b-9e53-43c5-92fe-563fdd2793e9)

### np.log(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/a7e9b0da-b281-4ef3-b3a0-1b4715502a84)

### np.reciprocal(df["Moderate Positive Skew"])
![image](https://github.com/user-attachments/assets/0f1be1ac-dfdb-43ff-ac1c-2c774127152a)

### np.sqrt(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/34ec5e5b-561e-4059-b699-b758fe6ff9cc)

### np.square(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/ae015d8e-a80b-4e15-b6f3-d83b93144cfc)

### df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/925a6858-c59a-4668-948b-5817dff43772)

### df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
![image](https://github.com/user-attachments/assets/c61ceee3-c714-4941-94ca-f6ff032d0dd7)

### df.skew()
![image](https://github.com/user-attachments/assets/c04271e7-1a8a-4c81-bb32-e65dab56308c)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/87885251-fcb1-4db7-981e-5b59c497a78d)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/3a20027c-98bc-4067-a199-e01e74e6ee04)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/fd3be682-835a-4d37-b3c6-b196e4466604)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/c0093f96-050a-4c9a-87ab-722cc35935f3)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew_2"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew_2"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/7ec75127-7737-4e0f-827a-f45f047f4327)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/a4d1876a-827f-4d65-b758-b7925690b507)

```
sm.qqplot(df["Highly Negative Skew_1"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/fed3424d-31a4-4839-bfe1-ff8cc7b64951)

```
dt=pd.read_csv("titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt["Age"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/f868a90b-5832-4102-a0e0-a1b754576852)

```
sm.qqplot(dt['Age_1'],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/12d2d30a-d082-409d-bf87-fc666f1bbcf9)

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
