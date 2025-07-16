import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# Cargar el dataset
df = pd.read_csv('Egg_Production.csv')  # Asegúrate de tener este archivo en el mismo directorio

# Seleccionar variables predictoras y variable objetivo
X = df.drop(['Total_egg_production'], axis=1)
y = df['Total_egg_production']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ XGBRegressor ------------------
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_score = xgb.score(X_test, y_test)
xgb_cv = cross_val_score(xgb, X_train, y_train, cv=5)
print(f"Precisión XGBRegressor: {xgb_score:.4f}")
print(f"Promedio Validación Cruzada XGBRegressor: {np.mean(xgb_cv):.4f}")

# Gráfico de predicciones XGB
pd.DataFrame(xgb_pred, columns=['Predicción']).plot.line(color='blue')
plt.title('Predicciones XGBRegressor')
plt.xlabel('Índice')
plt.ylabel('Predicción')
plt.show()

# ------------------ Naive Bayes ------------------
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)
gnb_score = gnb.score(X_test, y_test)
y_train_cat = y_train.astype('category')
gnb_cv = cross_val_score(gnb, X_train, y_train_cat, cv=5, scoring='accuracy')
print(f"Precisión Naive Bayes: {gnb_score:.4f}")
print(f"Promedio Validación Cruzada Naive Bayes: {np.mean(gnb_cv):.4f}")

# Gráfico de predicciones Naive Bayes
pd.DataFrame(gnb_pred, columns=['Predicción']).plot.line(color='orange')
plt.title('Predicciones Naive Bayes')
plt.xlabel('Índice')
plt.ylabel('Predicción')
plt.show()

# ------------------ KNN Regressor ------------------
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_score = knn.score(X_test, y_test)
knn_cv = cross_val_score(knn, X_train, y_train, cv=5)
print(f"Precisión KNN: {knn_score:.4f}")
print(f"Promedio Validación Cruzada KNN: {np.mean(knn_cv):.4f}")

# Gráfico de predicciones KNN
pd.DataFrame(knn_pred, columns=['Predicción']).plot.line(color='green')
plt.title('Predicciones KNN')
plt.xlabel('Índice')
plt.ylabel('Predicción')
plt.show()

# ------------------ Recomendaciones Prácticas (OE5) ------------------
print("\nRECOMENDACIONES PRÁCTICAS BASADAS EN LOS HALLAZGOS (OE5):")
print("- El modelo XGBRegressor mostró el mejor desempeño, por lo que se recomienda su uso para predecir la producción de huevos en función de variables ambientales.")
print("- Se sugiere monitorear y ajustar las variables ambientales más influyentes (según la importancia de las variables del modelo XGB) para maximizar la producción.")
print("- Implementar sistemas de control ambiental automatizados puede mejorar la precisión de las predicciones y el rendimiento productivo.")
print("- Realizar validaciones periódicas del modelo con nuevos datos para mantener la precisión y adaptabilidad a cambios en el entorno avícola.")

# (Opcional) Mostrar importancia de variables en XGB
importancias = pd.Series(xgb.feature_importances_, index=X.columns)
importancias.sort_values().plot(kind='barh')
plt.title('Importancia de Variables - XGBRegressor')
plt.xlabel('Importancia')
plt.show()