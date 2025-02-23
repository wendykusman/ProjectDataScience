# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 2. Preprocessing Data
# Baca data
data = pd.read_csv('GOOGL_2006-01-01_to_2018-01-01.csv')

# Konversi kolom 'Date' menjadi tipe datetime
data['Date'] = pd.to_datetime(data['Date'])

# Sort data berdasarkan tanggal
data.sort_values('Date', inplace=True)

# Plotting harga penutupan
plt.figure(figsize=(14, 8))
plt.plot(data['Date'], data['Close'], label='Nilai Aktual')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan (USD)')
plt.title('Harga Saham Google dari 2006 hingga 2018')
plt.legend()
plt.grid(True)
plt.show()

# Feature Engineering
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()
data['Volatility'] = data['High'] - data['Low']
data['Daily_Return'] = data['Close'].pct_change()

# Mengatasi missing values
data.dropna(inplace=True)

# Scaling target 'Close'
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(data[['Close']])

# Memisahkan fitur dan target
X = data[['MA5', 'MA20', 'Volatility', 'Daily_Return']]
y = data['Close']

# Scaling/Normalizing Features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Membuat dataset untuk LSTM
def create_dataset(X, y, time_step=1):
    Xs, ys = [], []
    for i in range(len(X)-time_step):
        Xs.append(X[i:(i+time_step)])
        ys.append(y[i+time_step])
    return np.array(Xs), np.array(ys)

time_step = 60
X_lstm, y_lstm = create_dataset(X_scaled, y_scaled, time_step)

# Membagi dataset menjadi training dan testing
train_size = int(len(X_lstm) * 0.8)
test_size = len(X_lstm) - train_size
X_train, X_test = X_lstm[0:train_size], X_lstm[train_size:len(X_lstm)]
y_train, y_test = y_lstm[0:train_size], y_lstm[train_size:len(y_lstm)]

# 3. Membangun dan Melatih model LsTM
# Membangun model LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Menggunakan Early Stopping untuk menghindari overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Melatih model dengan data validasi
history = model.fit(X_train, y_train, batch_size=1, epochs=1, validation_data=(X_test, y_test), callbacks=[early_stop])

# 4. Evaluasi dan Prediksi
# Prediksi menggunakan data test
predictions = model.predict(X_test)

# Mengembalikan skala prediksi ke bentuk asli
predictions = target_scaler.inverse_transform(predictions)

# Plotting hasil prediksi vs nilai aktual
plt.figure(figsize=(14, 8))
plt.plot(data['Date'], data['Close'], label='Nilai Aktual')
plt.plot(data['Date'].iloc[-len(predictions):], predictions, label='Prediksi', color='red')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan (USD)')
plt.title('Prediksi Harga Saham Google menggunakan LSTM')
plt.legend()
plt.grid(True)
plt.show()

# Membuat prediksi 30 hari ke depan dengan penyesuaian sequence
future_steps = 30
future_predictions = []

last_sequence = X_test[-1]  # Menggunakan sequence terakhir dari data test

for _ in range(future_steps):
    # Prediksi berikutnya menggunakan last_sequence
    next_pred = model.predict(last_sequence.reshape(1, time_step, X_train.shape[2]))
    future_predictions.append(next_pred[0, 0])
    
    # Update last_sequence dengan prediksi terbaru
    next_pred_full = np.zeros((1, X_train.shape[2]))  # Membuat array prediksi baru dengan ukuran yang sesuai
    next_pred_full[0, 0] = next_pred  # Menambahkan prediksi terbaru sebagai fitur pertama
    
    # Menyusun kembali sequence
    last_sequence = np.concatenate([last_sequence[1:], next_pred_full], axis=0)

# Mengembalikan skala prediksi masa depan ke bentuk asli
future_predictions = target_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Menggabungkan prediksi masa depan dengan data asli untuk plotting
future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)
future_df = pd.DataFrame({'Date': future_dates, 'Predictions': future_predictions.ravel()})

# Plotting hasil prediksi lanjutan
plt.figure(figsize=(14, 8))
plt.plot(data['Date'], data['Close'], label='Nilai Aktual')
plt.plot(data['Date'].iloc[-len(predictions):], predictions, label='Prediksi', color='red')
plt.plot(future_df['Date'], future_df['Predictions'], label='Prediksi Masa Depan', color='green')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan (USD)')
plt.title('Prediksi Harga Saham Google menggunakan LSTM')
plt.legend()
plt.grid(True)
plt.show()
