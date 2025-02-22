# Tahap 1 : Proses EDA
# 1. import pandas dan import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('train_house.csv')
# Data Understanding
print(data.head())
print(data.tail())

print(data.info())
print(data.describe())

# 2. mengatasi missing_value dengan cara drop variabel NaN untuk mengeliminasi data NaN
# Menghapus kolom yang memiliki nilai NaN
data.dropna(axis=1, inplace=True) #inplace =langsung pada dataframe tanpa buat salinan baru
print(data.info())
print(data.describe())

# 3. Pisahkan data numerical dan categorical sebelum masuk ke tahap visualisasi
# Pisahkan kolom numerik dan kategorikal

numerical_features = data.select_dtypes(include=[np.number])
print("data numerik :")
print(numerical_features.info())
print(numerical_features.describe())

categorical_features = data.select_dtypes(include=[object])
print("data kategorikal :")
print(categorical_features.info())
print(categorical_features.describe())

# Tahap 2 1D Visualisasi
"""
# 4. Buat Visualisasi 1D bar chart untuk data kategorikal dengan variabel Utilities, HouseStyle, ExterQual
# Menghitung frekuensi masing-masing kategori dalam variabel Utilities
utilities_count = data['Utilities'].value_counts()
print(utilities_count)
unique_utilities = data['Utilities'].unique()
print(unique_utilities)
# membuat bar chart Utilities
plt.figure(figsize=(10,6))
utilities_count.plot(kind='bar', color='skyblue')
plt.title('Distribusi Variabel Utilities')
plt.xlabel('Utilities')
plt.ylabel('Frekuensi')
plt.xticks(rotation=45)
plt.show()
"""
"""
# Menghitung frekuensi masing-masing kategori dalam variabel HouseStyle
housestyle_count = data['HouseStyle'].value_counts()
print(housestyle_count)
unique_housestyle = data['HouseStyle'].unique()
print(unique_housestyle)
# membuat bar chart HouseStyle
plt.figure(figsize=(10,6))
housestyle_count.plot(kind='bar', color='skyblue')
plt.title('Distribusi Variabel HouseStyle')
plt.xlabel('HouseStyle')
plt.ylabel('Frekuensi')
plt.xticks(rotation=45)
plt.show()
""" 
"""
# Menghitung frekuensi masing-masing kategori dalam variabel ExterQual
exterqual_count = data['ExterQual'].value_counts()
print(exterqual_count)
unique_exterqual = data['ExterQual'].unique()
print(unique_exterqual)
# membuat bar chart HouseStyle
plt.figure(figsize=(10,6))
exterqual_count.plot(kind='bar', color='skyblue')
plt.title('Distribusi Variabel ExterQual')
plt.xlabel('ExterQual')
plt.ylabel('Frekuensi')
plt.xticks(rotation=45)
plt.show()
"""

# 5. Buat Visualisasi 1D histogram dari data numerikal dengan variabel SalePrice, GrLivArea, PoolArea
# Membuat histogram untuk variabel SalePrice
"""
plt.figure(figsize=(10,6))
plt.hist(data['SalePrice'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribusi Variabel SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frekuensi')
plt.grid(True)
# menambahkan garis merah putus-putus untuk mean
mean_value = np.mean(data['SalePrice'])
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.legend()
plt.show()
"""
"""
# membuat log 10
# Membuat histogram untuk variabel SalePrice
data['LogSalePrice'] = np.log10(data['SalePrice']+1)
plt.figure(figsize=(10,6))
plt.hist(data['LogSalePrice'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribusi Variabel LogSalePrice')
plt.xlabel('Log(SalePrice)')
plt.ylabel('Frekuensi')
plt.grid(True)
# menambahkan garis merah putus-putus untuk mean
mean_log_value = np.mean(data['LogSalePrice'])
plt.axvline(mean_log_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_log_value:.2f}')
plt.legend()
plt.show()
"""
"""
# Membuat histogram untuk variabel GrLivArea
plt.figure(figsize=(10,6))
plt.hist(data['GrLivArea'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribusi Variabel GrLivArea')
plt.xlabel('GrLivArea')
plt.ylabel('Frekuensi')
plt.grid(True)
# menambahkan garis merah putus-putus untuk mean
mean_value = np.mean(data['GrLivArea'])
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.legend()
plt.show()
"""
"""
# membuat log 10
# Membuat histogram untuk variabel GrLivArea
data['LogGrLivArea'] = np.log10(data['GrLivArea']+1)
plt.figure(figsize=(10,6))
plt.hist(data['LogGrLivArea'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribusi Variabel LogGrLivArea')
plt.xlabel('Log(GrLivArea)')
plt.ylabel('Frekuensi')
plt.grid(True)
# menambahkan garis merah putus-putus untuk mean
mean_log_value = np.mean(data['LogGrLivArea'])
plt.axvline(mean_log_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_log_value:.2f}')
plt.legend()
plt.show()
"""
"""
# Membuat histogram untuk variabel PoolArea
plt.figure(figsize=(10,6))
plt.hist(data['PoolArea'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribusi Variabel PoolArea')
plt.xlabel('PoolArea')
plt.ylabel('Frekuensi')
plt.grid(True)
# menambahkan garis merah putus-putus untuk mean
mean_value = np.mean(data['PoolArea'])
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.legend()
plt.show()
"""
"""
# membuat log 10
# Membuat histogram untuk variabel PoolArea
data['LogPoolArea'] = np.log10(data['PoolArea']+1)
plt.figure(figsize=(10,6))
plt.hist(data['LogPoolArea'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribusi Variabel LogPoolArea')
plt.xlabel('Log(PoolArea)')
plt.ylabel('Frekuensi')
plt.grid(True)
# menambahkan garis merah putus-putus untuk mean
mean_log_value = np.mean(data['LogPoolArea'])
plt.axvline(mean_log_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_log_value:.2f}')
plt.legend()
plt.show()
"""
# Tahap 3 2D Visualisasi
# GrLivArea dan SalePrice
"""
# Membuat plot
plt.figure(figsize=(10, 6))
plt.scatter(data['GrLivArea'], data['SalePrice'], alpha=0.5)
plt.title('GrLivArea vs. SalePrice')
plt.xlabel('GrLivArea (sq ft)')
plt.ylabel('SalePrice (USD)')
plt.legend()
plt.grid(True)
plt.show()
"""
"""
# GarageArea dan SalePrice
# Membuat plot
plt.figure(figsize=(10, 6))
plt.scatter(data['GarageArea'], data['SalePrice'], alpha=0.5)
plt.title('GarageArea vs. SalePrice')
plt.xlabel('GarageArea (sq ft)')
plt.ylabel('SalePrice (USD)')
plt.legend()
plt.grid(True)
plt.show()
"""
correlationSGr = data['SalePrice'].corr(data['GrLivArea'])
print (f'Korelasi antara SalePrice dan GrLivArea : {correlationSGr:.2f}')
correlationSGa = data['SalePrice'].corr(data['GarageArea'])
print (f'Korelasi antara SalePrice dan GarageArea : {correlationSGa:.2f}')