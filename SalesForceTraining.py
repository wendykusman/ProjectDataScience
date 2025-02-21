# Case 1 = Sales Force Training
# 1. Buatlah DataFrame 1 kolom dengan 25 baris bernama ‘TransactionAmount’
import numpy as np
data = [100,150,50,100,130,120,100,85,70,150,150,120,50,100,100,140,90,150,50,90,120,100,110,75,65]

TransactionAmount0 = np.array(data)
print('data sebelum diurutkan :')
print(TransactionAmount0)

TransactionAmount = np.sort(data)
print('data setelah diurutkan :')
print(TransactionAmount)

# 2. Buatlah analisis measure of central tendency dari data ‘TransactionAmount’
from scipy import stats

# menghitung mean, median, dan mode
mean = np.mean(TransactionAmount)
median = np.median(TransactionAmount)
mode = stats.mode(TransactionAmount)

print(f'Mean : {mean}')
print(f'Median : {median}')
print(f'Mode : {mode}')

# 3. Lakukan juga analisis measure of variability dari data ‘TransactionAmount’
range_value = np.max(TransactionAmount) - np.min(TransactionAmount)
print(f'Range :{range_value}')
q1 = np.percentile(TransactionAmount,25)
print(f'Q1 :{q1}')
q2 = np.percentile(TransactionAmount,50)
print(f'Q2 :{q2}')
q3 = np.percentile(TransactionAmount,75)
print(f'Q3 :{q3}')
variansi = np.var(TransactionAmount, ddof=1)
print(f'Variance : {variansi}')
std_dev = np.std(TransactionAmount, ddof=1)
print(f'Standard Deviation :{std_dev}')

# 4.Tentukan H0 & H1 untuk persiapan dilakukan T-Test
# Hipotesis Nol (H0): Tidak ada peningkatan penjualan rata-rata per transaksi setelah pelatihan. 
#                     penjualan rata-rata tetap $100 per transaksi.
# Hipotesis Alternatif (H1): Ada peningkatan penjualan rata-rata per transaksi setelah pelatihan. 
#                     penjualan rata-rata lebih dari $100 per transaksi.

#5. Lakukan proses T-Test pada data rata-rata jumlah transaksi (dengan alpha = 5%)
mean_sebelumnya = 100
alpha = 0.05
# one sample t-test
t_stat, p_value = stats.ttest_1samp(TransactionAmount, mean_sebelumnya)

print(f'T-test : {t_stat}')
print(f'P-value :{p_value}')
# 6. Buatlah kesimpulan dari hasil pengujian sales force training!
if p_value < alpha:
    print('Tolak H0, ada bukti signifikan bahwa rata-rata penjualan berbeda dari 100.')
else:
    print('Gagal menolak H0, tidak ada bukti signifikan bahwa rata-rata penjualan berbeda dari 100')
