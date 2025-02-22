# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("fitness_class_2212.csv")
print(data.head())
print(data.tail())
print(data.info())
print(data.describe())

# Drop rows with missing values
data_cleaned = data.dropna().copy()
# Replace '-' in the 'category' column with 'Others'
data_cleaned.loc[data_cleaned['category'] == '-', 'category'] = 'Others'

# Menghitung frekuensi masing-masing kategori dalam variabel Category
category_count = data_cleaned['category'].value_counts()
"""
# membuat bar chart Category
plt.figure(figsize=(10,6))
category_count.plot(kind='bar', color='skyblue')
plt.title('Distribusi Variabel Category')
plt.xlabel('category')
plt.ylabel('frekuensi')
plt.xticks(rotation=45)
plt.show()
"""
"""
# membuat bar chart dari distribusi kehadiran
plt.figure(figsize=(10,6))
sns.countplot(data=data_cleaned, x='attended')
plt.title('Distribusi Kehadiran')
plt.xlabel('Attended')
plt.ylabel('Count')
plt.show()
"""

# Melakukan one hot encoding untuk mengubah variabel kategorikal menjadi numerical
# Mengatur opsi tampilan
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

labelencoder = LabelEncoder()

data_cleaned.loc[:,'category'] = labelencoder.fit_transform(data_cleaned['category'])
data_cleaned.loc[:,'days_before'] = labelencoder.fit_transform(data_cleaned['days_before'])
data_cleaned.loc[:,'day_of_week'] = labelencoder.fit_transform(data_cleaned['day_of_week'])
data_cleaned.loc[:,'time'] = labelencoder.fit_transform(data_cleaned['time'])

# Ubah tipe data kolom numerik menjadi float sebelum standarisasi
data_cleaned[['months_as_member', 'weight']] = data_cleaned[['months_as_member', 'weight']].astype(float)

scaler = StandardScaler()
data_cleaned.loc[:, ['months_as_member', 'weight']] = scaler.fit_transform(data_cleaned[['months_as_member', 'weight']])

# Definisikan fitur dan variabel target
X = data_cleaned.drop(['attended', 'booking_id'], axis=1)
y = data_cleaned['attended']

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Daftar algoritma
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}
"""
# Melatih dan menguji setiap model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
"""

# Definisikan parameter grid untuk GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.05],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0]
}

# Inisiasi GridSearchCV
grid_search = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Tampilkan parameter terbaik
print("Best parameters found: ", grid_search.best_params_)

# Gunakan parameter terbaik untuk model akhir
best_model = grid_search.best_estimator_

# Melakukan prediksi pada data uji
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
cr = classification_report(y_test, y_pred)
print("Classification Report:")
print(cr)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Analisis feature importance
feature_importance = best_model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)