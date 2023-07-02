# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""Sumber Dataset: https://www.kaggle.com/datasets/laotse/credit-risk-dataset"""

data = pd.read_csv('/content/drive/My Drive/dataset/CreditRisk.csv')
data.head()

data = data.rename(columns={'person_age': 'Umur','person_income':'Pendapatan','person_home_ownership':'KepemilikanRumah',
                            'person_emp_length':'LamaKerja','loan_intent':'TujuanPeminjaman','loan_grade':'TingkatanPinjaman',
                            'loan_amnt':'JumlahPinjaman','loan_int_rate':'SukuBunga','loan_status':'StatusPinjaman','loan_percent_income':'%Pendapatan',
                            'cb_person_default_on_file':'HistoriPeminjaman','cb_person_cred_hist_length':'JumlahHistoriPeminjaman'})

data.head()
data.shape
data.info()
data.describe()
data.isnull().sum()

print('Total Duplicated Values in dataframe are {0}'.format(data[data.duplicated()].shape[0]))

"""### Drop data duplicated"""

data.drop_duplicates(inplace=True)

data.shape

"""### Cek Perbandingan Kelas"""

data['StatusPinjaman'].value_counts()

viz=data.groupby('StatusPinjaman').size()
viz.plot(kind='bar', figsize=(4,3))

"""### cek Missing Value"""

data.isnull().sum()

"""### Percobaan 1: Delete Missing Value"""

data = data.dropna()

data.isnull().sum()

data.shape

"""### Cek Korelasi Feature"""

plt.figure(figsize=(12,8))
plt.title('Correlation Between Variables')

sns.heatmap(data.corr(), annot=True)

"""### Cek Persebaran Data Kategorikal"""

data['Umur'].nunique()

data['KepemilikanRumah'].value_counts()

data['TujuanPeminjaman'].value_counts()

data['TingkatanPinjaman'].value_counts()

data['JumlahHistoriPeminjaman'].value_counts()

"""### Cek Persebaran Data Numerik"""

data1 = data.copy()

data1 = data1.drop(columns=['KepemilikanRumah','TujuanPeminjaman','TingkatanPinjaman','StatusPinjaman','JumlahHistoriPeminjaman'])

data1.head(10)

data1.hist(bins=10, figsize=(10, 6))
plt.xlabel('Nilai')
plt.ylabel('Frekuensi')
plt.title('Histogram Data')
plt.show()

sns.boxplot(data=data['Umur'])

sns.boxplot(data=data['Pendapatan'])

sns.boxplot(data=data['JumlahPinjaman'])

sns.boxplot(data=data['LamaKerja'])

sns.boxplot(data=data['SukuBunga'])

"""## Feature Engineering

Seleksi Fitur - Drop Fitur, etc

### Drop Feature
"""

data = data.drop('HistoriPeminjaman', axis=1)

data.head()

data = data.drop('%Pendapatan', axis=1)

data.head()

"""### Drop data outliers"""

data = data[data['Umur'] <= 120]

data['Umur'].value_counts()

data = data[data['Pendapatan'] <= 5000000000]

sns.boxplot(data=data['Pendapatan'])

data = data[data['LamaKerja'] <= 120]

"""### Cek Data Selesai FE"""

data.head()

data.shape

"""## Encoding Data"""

from sklearn.preprocessing import LabelEncoder

le = [LabelEncoder(), LabelEncoder(), LabelEncoder()]

data['KepemilikanRumah'] = le[0].fit_transform(data['KepemilikanRumah'])
data['TujuanPeminjaman'] = le[1].fit_transform(data['TujuanPeminjaman'])
data['TingkatanPinjaman'] = le[2].fit_transform(data['TingkatanPinjaman'])

data.head()

"""## Pre Modelling"""

x = data.drop(columns='StatusPinjaman')
y = data['StatusPinjaman']

x.shape, y.shape

from sklearn.model_selection import train_test_split, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x,y ,train_size=0.7)

"""## Modelling

### Random Forest Algorithm
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]}

rf_model = RandomForestClassifier()

grid_search = GridSearchCV(rf_model, param_grid, cv=5)
grid_search.fit(x_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

best_model = grid_search.best_estimator_
test_accuracy = best_model.score(x_test, y_test)
print("Test accuracy: ", test_accuracy)


import pickle

with open('modelRF.pkl','wb') as f:
    pickle.dump(best_model,f)

with open('modelRF.pkl','wb') as f:
    pickle.dump(best_model,f)

filename = 'crc_trained.sav'
pickle.dump(best_model,open(filename,'wb'))

"""Predict"""

input_data = (25, 145000000, 2, 1,2,3,500000,11.14,2)

id_np_array = np.asarray(input_data)
id_reshaped = id_np_array.reshape(1,-1)

prediction = best_model.predict(id_reshaped)
print(prediction)

if(prediction[0]==0):
    print("Credit Status: Tidak Bermasalah")
else:
    print("Credit Status: Bermasalah")

"""Umur	Pendapatan	KepemilikanRumah	LamaKerja	TujuanPeminjaman TingkatanPinjaman	JumlahPinjaman	SukuBunga	StatusPinjaman	JumlahHistoriPeminjaman

## HASIL AKURASI ALGORITMA

Random Forest: 92.6%
"""