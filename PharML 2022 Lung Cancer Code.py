#!/usr/bin/env python
# coding: utf-8

# In[251]:


import pandas as pd


# In[202]:


pip install lifelines


# In[507]:


df = pd.read_csv('/pharml/data/flatiron_training.csv')


# In[508]:


x = 0
for i in df.event_died:
    if i == 1:
        x = x + 1
print(x)        


# In[509]:


df.head(10)


# In[514]:


df.isna().sum()


# In[518]:


for i in df.columns:
    print(i)


# In[519]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# df[].apply(LabelEncoder().fit_transform)
df_raw = df
#df[:, [0,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22]] = le.fit_transform(df[:,[0,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22]])


# In[520]:


df_temp = df_raw.astype("str").apply(LabelEncoder().fit_transform)
df_final = df_temp.where(~df_raw.isna(), df_raw)


# In[521]:


df_final


# In[525]:


# df.head()
num_col = [col for col in df.columns if df[col].dtypes != 'O']


# In[526]:


cols = df.columns


# In[527]:


num_col


# In[528]:


from sklearn.impute import KNNImputer


# In[529]:


knn = KNNImputer(n_neighbors = 5, add_indicator = True)


# In[530]:


knn.fit(df_final)


# In[531]:


df_final = knn.transform(df_final)


# In[532]:


df_encoded = pd.DataFrame(df_final)
df_encoded.iloc[:,0:25]


# In[533]:


df_encoded.drop(df_encoded.iloc[:,25:39], inplace = True, axis = 1)


# In[534]:


df_encoded.columns = cols


# In[535]:


df_encoded


# In[536]:


df['event_died'].value_counts()


# In[537]:


df.lab_creatinine.unique()


# In[538]:


df_encoded.dtypes


# In[539]:


df_encoded.biomarker_pdl1.unique()


# In[540]:


df_encoded['biomarker_pdl1'].value_counts()


# In[541]:


for i in range(31334):
    # df_encoded.biomarker_pdl1:
    if df_encoded.biomarker_pdl1[i] >= 0 and df_encoded.biomarker_pdl1[i]<0.5:
        df_encoded.biomarker_pdl1[i] = 0
    elif df_encoded.biomarker_pdl1[i] >= 0.5:
        df_encoded.biomarker_pdl1[i] = 1
        
            


# In[542]:


df_encoded.biomarker_pdl1.unique()


# In[543]:


df_encoded['biomarker_pdl1'].value_counts()


# In[544]:


for i in range(31334):
    # df_encoded.biomarker_pdl1:
    if df_encoded.biomarker_ros1[i] >= 0 and df_encoded.biomarker_ros1[i]<0.5:
        df_encoded.biomarker_ros1[i] = 0
    elif df_encoded.biomarker_ros1[i] >= 0.5:
        df_encoded.biomarker_ros1[i] = 1
        
            


# In[545]:


df_encoded.biomarker_ros1.unique()


# In[546]:


df_encoded['biomarker_ros1'].value_counts()


# In[547]:


for i in range(31334):
    # df_encoded.biomarker_pdl1:
    if df_encoded.biomarker_braf[i] >= 0 and df_encoded.biomarker_braf[i]<0.5:
        df_encoded.biomarker_braf[i] = 0
    elif df_encoded.biomarker_braf[i] >= 0.5:
        df_encoded.biomarker_braf[i] = 1
        
            


# In[ ]:





# In[548]:


df_encoded['biomarker_braf'].value_counts()


# In[549]:


df_encoded.biomarker_braf.unique()


# In[550]:


for i in range(31334):
    # df_encoded.biomarker_pdl1:
    if df_encoded.biomarker_egfr[i] >= 0 and df_encoded.biomarker_egfr[i]<0.5:
        df_encoded.biomarker_egfr[i] = 0
    elif df_encoded.biomarker_egfr[i] >= 0.5:
        df_encoded.biomarker_egfr[i] = 1
        
            


# In[551]:


df_encoded['biomarker_egfr'].value_counts()


# In[552]:


df_encoded.biomarker_egfr.unique()


# In[553]:


df_encoded.lab_creatinine.unique()


# In[554]:


for i in range(31334):
    # df_encoded.biomarker_pdl1:
    if df_encoded.lab_creatinine[i]>= 0 and df_encoded.lab_creatinine[i]<1:
        df_encoded.lab_creatinine[i] = 0
    elif df_encoded.lab_creatinine[i] >= 1:
        df_encoded.lab_creatinine[i] = 2
        
            


# In[555]:


df_encoded['lab_creatinine'].value_counts()


# In[556]:


df_encoded.lab_creatinine.unique()


# In[557]:


for i in range(31334):
    # df_encoded.biomarker_pdl1:
    if df_encoded.biomarker_alk[i] >= 0 and df_encoded.biomarker_alk[i]<0.5:
        df_encoded.biomarker_alk[i] = 0
    elif df_encoded.biomarker_alk[i] >= 0.5:
        df_encoded.biomarker_alk[i] = 1
        
            


# In[558]:


df_encoded.biomarker_alk.unique()


# In[559]:


for i in range(31334):
    # df_encoded.biomarker_pdl1:
    if df_encoded.lab_ldh[i] >= 0 and df_encoded.lab_ldh[i]<1:
        df_encoded.lab_ldh[i] = 0
    elif df_encoded.lab_ldh[i] >= 1:
        df_encoded.lab_ldh[i] = 2
        
            


# In[560]:


df_encoded.lab_ldh.unique() 


# In[561]:


for i in range(31334):
    # df_encoded.biomarker_pdl1:
    if df_encoded.lab_alanineaminotransferase[i] >= 0 and df_encoded.lab_alanineaminotransferase[i]<1:
        df_encoded.lab_alanineaminotransferase[i] = 0
    elif df_encoded.lab_alanineaminotransferase[i] >= 1:
        df_encoded.lab_alanineaminotransferase[i] = 2
        
            


# In[562]:


df_encoded.lab_alanineaminotransferase.unique()


# In[563]:


for i in range(31334):
    # df_encoded.biomarker_pdl1:
    if df_encoded.biomarker_kras[i] >= 0 and df_encoded.biomarker_kras[i]<0.5:
        df_encoded.biomarker_kras[i] = 0
    elif df_encoded.biomarker_kras[i] >= 0.5:
        df_encoded.biomarker_kras[i] = 1
        
            


# In[564]:


df_encoded['biomarker_kras'].value_counts()


# In[565]:


df_encoded.biomarker_kras.unique()


# In[566]:


df.index_lot1_year.unique()


# In[567]:


df_encoded.head()


# In[568]:


df_drop = df_encoded.drop('daysto_event',axis = 1)


# In[569]:


df_drop.head()


# In[570]:


y_drop = df_drop['event_died']


# In[571]:


X_drop = df_drop.drop('event_died', axis = 1)


# In[572]:


X_drop.head()


# In[573]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)


# In[574]:


classifier.fit(X_drop, y_drop)


# In[575]:


y_test_drop = classifier.predict(df_encoded_test)


# In[488]:


print(y_test_drop)
for i in y_test_drop:
    print(i)


# In[ ]:


event_diedlist = []
for i in y_test_drop:
    event_diedlist.append(i)


# In[ ]:


print(len(event_diedlist))


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df_encoded.iloc[:, 2:26]
y = df_encoded.iloc[:, 1]


# In[577]:


X


# In[578]:


print(y)


# In[579]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[59]:


pip install catboost


# In[580]:


from catboost import CatBoostRegressor

#model = CatBoostRegressor(iterations=50)
model = CatBoostRegressor()


# In[ ]:





# In[581]:


# model.fit(X_train, y_train)


# In[582]:


from sklearn.model_selection import GridSearchCV


# In[583]:


CBC = CatBoostRegressor()
parameters = {'depth'         : [4,5,6,7,8,9, 10],
              'learning_rate' : [0.01,0.02,0.03,0.04],
              'iterations'    : [50,90,100, 150, 170, 200, 300,350]
                 }


# In[584]:


Grid_CBC = GridSearchCV(estimator=CBC, param_grid = parameters, cv = 2, n_jobs=-1)
Grid_CBC.fit(X_train, y_train)


# In[585]:


print(Grid_CBC.best_params_)


# In[592]:


model = CatBoostRegressor(depth = 6, iterations = 400, learning_rate = 0.04)


# In[594]:


model.fit(X_train, y_train)


# In[595]:


predict_test = model.predict(X_test)


# In[596]:


from sklearn.metrics import mean_squared_error
print("MSE:",mean_squared_error(y_test, predict_test))


# In[597]:


from lifelines.utils import concordance_index
print(f'Concordance index: {concordance_index(y_test, predict_test)}')


# In[92]:


df.diagnosis_nsclc_year.unique()


# In[598]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn


# In[600]:


test_df = df[df["biomarker_alk"] == 2]
test_df


# In[602]:


df_train


# In[97]:


df['diagnosis_nsclc_year'].dropna(axis = 0)


# In[603]:


df_test = pd.read_csv('/pharml/data/flatiron_test.csv')
df_test


# In[643]:


pid = df_test['patientid']


# In[604]:


df_temp_test = df_test.astype("str").apply(LabelEncoder().fit_transform)
df_final_test = df_temp_test.where(~df_test.isna(), df_test)


# In[605]:


df_final_test


# In[606]:


df_final_test.isna().sum()


# In[607]:


df_final_test.histology.unique()   #dtypes


# In[608]:


knn.fit(df_final_test)


# In[609]:


df_final_test1 = knn.transform(df_final_test)


# In[610]:


df_encoded_test = pd.DataFrame(df_final_test1)


# In[611]:


df_encoded_test.iloc[:,0:25]


# In[612]:


df_encoded_test.drop(df_encoded_test.iloc[:,23:37], inplace = True, axis = 1)


# In[613]:


df_encoded_test.head(10)


# In[614]:


cols_test = df_final_test.columns


# In[615]:


df_encoded_test.columns = cols_test


# In[616]:


df_encoded_test.head()


# In[617]:


df_encoded_test['event_died'] = event_diedlist


# In[618]:


df_encoded_test.head()


# In[619]:


for i in df_test.columns:
    print(i)


# In[620]:


for i in df.columns:
    print(i)


# In[621]:


df_encoded_test.head()


# In[622]:


df_encoded_test.biomarker_pdl1.unique()


# In[623]:


for i in range(8953):
    # df_encoded.biomarker_pdl1:
    if df_encoded_test.biomarker_pdl1[i] >= 0 and df_encoded_test.biomarker_pdl1[i]<0.5:
        df_encoded_test.biomarker_pdl1[i] = 0
    elif df_encoded_test.biomarker_pdl1[i] >= 0.5:
        df_encoded_test.biomarker_pdl1[i] = 1
        
            


# In[624]:


df_encoded_test.biomarker_pdl1.unique()


# In[625]:


df_encoded_test['biomarker_pdl1'].value_counts()


# In[626]:


for i in range(8953):
    # df_encoded.biomarker_pdl1:
    if df_encoded_test.biomarker_ros1[i] >= 0 and df_encoded_test.biomarker_ros1[i]<0.5:
        df_encoded_test.biomarker_ros1[i] = 0
    elif df_encoded_test.biomarker_ros1[i] >= 0.5:
        df_encoded_test.biomarker_ros1[i] = 1
        
            


# In[627]:


df_encoded_test['biomarker_ros1'].value_counts()


# In[628]:


df_encoded_test['biomarker_braf'].value_counts()


# In[629]:


for i in range(8953):
    # df_encoded.biomarker_pdl1:
    if df_encoded_test.biomarker_braf[i] >= 0 and df_encoded_test.biomarker_braf[i]<0.5:
        df_encoded_test.biomarker_braf[i] = 0
    elif df_encoded_test.biomarker_braf[i] >= 0.5:
        df_encoded_test.biomarker_braf[i] = 1
        
            


# In[630]:


df_encoded_test['biomarker_braf'].value_counts()


# In[631]:


for i in range(8953):
    # df_encoded.biomarker_pdl1:
    if df_encoded_test.biomarker_egfr[i] >= 0 and df_encoded_test.biomarker_egfr[i]<0.5:
        df_encoded_test.biomarker_egfr[i] = 0
    elif df_encoded_test.biomarker_egfr[i] >= 0.5:
        df_encoded_test.biomarker_egfr[i] = 1
        
            


# In[632]:


df_encoded_test['biomarker_egfr'].value_counts()


# In[633]:


df_encoded_test['lab_creatinine'].value_counts()


# In[634]:


for i in range(8953):
    # df_encoded.biomarker_pdl1:
    if df_encoded_test.lab_creatinine[i]>= 0 and df_encoded_test.lab_creatinine[i]<1:
        df_encoded_test.lab_creatinine[i] = 0
    elif df_encoded_test.lab_creatinine[i] >= 1:
        df_encoded_test.lab_creatinine[i] = 2
        
            


# In[635]:


df_encoded_test['lab_creatinine'].value_counts()


# In[636]:


for i in range(8953):
    # df_encoded.biomarker_pdl1:
    if df_encoded_test.biomarker_alk[i] >= 0 and df_encoded_test.biomarker_alk[i]<0.5:
        df_encoded_test.biomarker_alk[i] = 0
    elif df_encoded_test.biomarker_alk[i] >= 0.5:
        df_encoded_test.biomarker_alk[i] = 1
        
            


# In[637]:


for i in range(8953):
    # df_encoded.biomarker_pdl1:
    if df_encoded_test.lab_ldh[i] >= 0 and df_encoded_test.lab_ldh[i]<1:
        df_encoded_test.lab_ldh[i] = 0
    elif df_encoded_test.lab_ldh[i] >= 1:
        df_encoded_test.lab_ldh[i] = 2
        
            


# In[638]:


for i in range(8953):
    # df_encoded.biomarker_pdl1:
    if df_encoded_test.lab_alanineaminotransferase[i] >= 0 and df_encoded_test.lab_alanineaminotransferase[i]<1:
        df_encoded_test.lab_alanineaminotransferase[i] = 0
    elif df_encoded_test.lab_alanineaminotransferase[i] >= 1:
        df_encoded_test.lab_alanineaminotransferase[i] = 2
        
            


# In[639]:


for i in range(8953):
    # df_encoded.biomarker_pdl1:
    if df_encoded_test.biomarker_kras[i] >= 0 and df_encoded_test.biomarker_kras[i]<0.5:
        df_encoded_test.biomarker_kras[i] = 0
    elif df_encoded_test.biomarker_kras[i] >= 0.5:
        df_encoded_test.biomarker_kras[i] = 1
        
            


# In[640]:


df_encoded_test.head()


# In[641]:


X_testing = df_encoded_test


# In[642]:


predict_test = model.predict(X_testing)


# In[505]:


print(predict_test.shape)


# In[648]:


dayslist = []
for i in predict_test:
    dayslist.append(i)


# In[646]:


df_finalsub = pd.DataFrame()


# In[647]:


df_finalsub['patientid'] = pid


# In[649]:


df_finalsub['prediction'] = dayslist


# In[650]:


df_finalsub.head()


# In[652]:


df_finalsub.to_csv('file.csv')

