#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de Producción Avícola: Efectos Ambientales en la Producción de Huevos
================================================================================

Este script implementa un análisis completo de los efectos ambientales en la producción 
de huevos, siguiendo las mejores prácticas de ciencia de datos y documentación científica.

Autor: Jorge Andrés Ayala
Fecha: 2024
"""

# ============================================================================
# IMPORTACIÓN DE BIBLIOTECAS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from scipy import stats
import warnings
import os

# Configurar warnings y estilo
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 80)
print("ANÁLISIS DE PRODUCCIÓN AVÍCOLA: EFECTOS AMBIENTALES")
print("=" * 80)
print("Bibliotecas importadas correctamente")

# ============================================================================
# MÓDULO 1: CARGA Y EXPLORACIÓN INICIAL DE DATOS
# ============================================================================
print("\n" + "=" * 80)
print("MÓDULO 1: CARGA Y EXPLORACIÓN INICIAL DE DATOS")
print("=" * 80)

def cargar_y_explorar_datos():
    """
    Carga el dataset y realiza exploración inicial
    """
    # Cargar el dataset de producción de huevos
    df = pd.read_csv('../data/Egg_Production.csv')
    
    # Agregar columna de fechas diarias desde la última fecha hacia atrás
    fecha_final = datetime(2023, 6, 29, 19, 50)
    df["Fecha"] = [fecha_final - timedelta(days=i) for i in range(len(df))][::-1]
    
    print("=== INFORMACIÓN GENERAL DEL DATASET ===")
    print(f"Dimensiones: {df.shape}")
    print(f"Variables: {len(df.columns)}")
    print(f"Registros: {len(df)}")
    print("\nPrimeros 5 registros:")
    print(df.head())
    
    print("\n=== INFORMACIÓN DE VARIABLES ===")
    print(df.info())
    
    print("\n=== RESUMEN ESTADÍSTICO ===")
    print(df.describe())
    
    print("\n=== ANÁLISIS DE VALORES NULOS ===")
    nulos = df.isnull().sum()
    print(nulos[nulos > 0] if nulos.sum() > 0 else "No hay valores nulos")
    
    print("\n=== ANÁLISIS DE DUPLICADOS ===")
    duplicados = df.duplicated().sum()
    print(f"Registros duplicados: {duplicados}")
    
    return df

# Ejecutar carga y exploración
df = cargar_y_explorar_datos()

# ============================================================================
# MÓDULO 2: LIMPIEZA Y PREPROCESAMIENTO DE DATOS
# ============================================================================
print("\n" + "=" * 80)
print("MÓDULO 2: LIMPIEZA Y PREPROCESAMIENTO DE DATOS")
print("=" * 80)

def limpiar_datos(df):
    """
    Implementa la limpieza y preprocesamiento de datos, incluyendo la detección 
    y manejo de valores atípicos mediante técnicas estadísticas robustas.
    """
    print("=== LIMPIEZA DE DATOS ===")
    print(f"Registros antes de la limpieza: {len(df)}")
    
    # Identificar columnas numéricas (excluyendo Fecha)
    columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns
    print(f"Variables numéricas identificadas: {list(columnas_numericas)}")
    
    # Aplicar método IQR para detectar outliers
    Q1 = df[columnas_numericas].quantile(0.25)
    Q3 = df[columnas_numericas].quantile(0.75)
    IQR = Q3 - Q1
    
    # Filtrar outliers
    df_limpio = df[~((df[columnas_numericas] < (Q1 - 1.5 * IQR)) | 
                       (df[columnas_numericas] > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    print(f"Registros después de la limpieza: {len(df_limpio)}")
    print(f"Registros eliminados: {len(df) - len(df_limpio)}")
    print(f"Porcentaje de datos conservados: {(len(df_limpio)/len(df)*100):.2f}%")
    
    # Guardar dataset limpio
    df_limpio.to_csv('../data/Egg_Production_Clean.csv', index=False)
    print("\nDataset limpio guardado en '../data/Egg_Production_Clean.csv'")
    
    return df_limpio

# Ejecutar limpieza
df = limpiar_datos(df)

# ============================================================================
# MÓDULO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================
print("\n" + "=" * 80)
print("MÓDULO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 80)

def analisis_descriptivo_completo(df):
    """
    Realiza un análisis descriptivo completo del dataset
    """
    print("=" * 80)
    print("ANÁLISIS DESCRIPTIVO COMPLETO DEL DATASET")
    print("=" * 80)
    
    # 1. Información básica del dataset
    print("\n1. INFORMACIÓN BÁSICA DEL DATASET")
    print("-" * 50)
    print(f"• Número total de registros: {len(df)}")
    print(f"• Número total de variables: {len(df.columns)}")
    print(f"• Variables numéricas: {len(df.select_dtypes(include=['number']).columns)}")
    print(f"• Variables categóricas: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"• Variables de fecha: {len(df.select_dtypes(include=['datetime']).columns)}")
    
    # 2. Análisis de valores nulos
    print("\n2. ANÁLISIS DE VALORES NULOS")
    print("-" * 50)
    nulos = df.isnull().sum()
    porcentaje_nulos = (nulos / len(df)) * 100
    for col in df.columns:
        if nulos[col] > 0:
            print(f"• {col}: {nulos[col]} valores nulos ({porcentaje_nulos[col]:.2f}%)")
        else:
            print(f"• {col}: Sin valores nulos")
    
    # 3. Estadísticas descriptivas detalladas
    print("\n3. ESTADÍSTICAS DESCRIPTIVAS DETALLADAS")
    print("-" * 50)
    columnas_numericas = df.select_dtypes(include=['number']).columns
    for col in columnas_numericas:
        print(f"\n• Variable: {col}")
        print(f"  - Media: {df[col].mean():.4f}")
        print(f"  - Mediana: {df[col].median():.4f}")
        print(f"  - Desviación estándar: {df[col].std():.4f}")
        print(f"  - Mínimo: {df[col].min():.4f}")
        print(f"  - Máximo: {df[col].max():.4f}")
        print(f"  - Rango: {df[col].max() - df[col].min():.4f}")
        print(f"  - Coeficiente de variación: {(df[col].std() / df[col].mean()) * 100:.2f}%")
        print(f"  - Asimetría: {df[col].skew():.4f}")
        print(f"  - Curtosis: {df[col].kurtosis():.4f}")
    
    # 4. Análisis de correlaciones
    print("\n4. ANÁLISIS DE CORRELACIONES")
    print("-" * 50)
    correlaciones = df[columnas_numericas].corr()
    print("Matriz de correlaciones:")
    print(correlaciones.round(4))
    
    # 5. Análisis de outliers
    print("\n5. ANÁLISIS DE OUTLIERS")
    print("-" * 50)
    for col in columnas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers_inferiores = df[df[col] < (Q1 - 1.5 * IQR)][col]
        outliers_superiores = df[df[col] > (Q3 + 1.5 * IQR)][col]
        total_outliers = len(outliers_inferiores) + len(outliers_superiores)
        porcentaje_outliers = (total_outliers / len(df)) * 100
        
        print(f"• {col}:")
        print(f"  - Outliers inferiores: {len(outliers_inferiores)}")
        print(f"  - Outliers superiores: {len(outliers_superiores)}")
        print(f"  - Total outliers: {total_outliers} ({porcentaje_outliers:.2f}%)")

def generar_graficos_analisis(df):
    """
    Genera gráficos de análisis para la tesis
    """
    print("\nGenerando gráficos de análisis...")
    
    columnas_numericas = df.select_dtypes(include=['number']).columns
    
    # 1. Distribución de variables numéricas
    n_cols = len(columnas_numericas)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(columnas_numericas):
        if i < len(axes):
            axes[i].hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribución de {col}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frecuencia')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/distribuciones_variables_numericas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Boxplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(columnas_numericas):
        if i < len(axes):
            axes[i].boxplot(df[col], patch_artist=True, boxprops=dict(facecolor='lightblue'))
            axes[i].set_title(f'Boxplot de {col}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/boxplots_variables_numericas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Mapa de calor de correlaciones
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[columnas_numericas].corr(), annot=True, cmap='coolwarm', 
                fmt='.3f', square=True, cbar_kws={'shrink': 0.8})
    plt.title('Matriz de Correlaciones', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../results/matriz_correlaciones.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Gráfico de dispersión para variables principales
    if len(columnas_numericas) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # Seleccionar las primeras 4 variables para scatter plots
        vars_scatter = columnas_numericas[:4]
        for i in range(min(4, len(vars_scatter))):
            for j in range(i+1, min(4, len(vars_scatter))):
                if i < len(axes):
                    axes[i].scatter(df[vars_scatter[i]], df[vars_scatter[j]], alpha=0.6)
                    axes[i].set_xlabel(vars_scatter[i])
                    axes[i].set_ylabel(vars_scatter[j])
                    axes[i].set_title(f'{vars_scatter[i]} vs {vars_scatter[j]}')
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

# Ejecutar análisis descriptivo
analisis_descriptivo_completo(df)

# Generar gráficos de análisis
generar_graficos_analisis(df)

# ============================================================================
# MÓDULO 4: ANÁLISIS ESTADÍSTICO INFERENCIAL
# ============================================================================
print("\n" + "=" * 80)
print("MÓDULO 4: ANÁLISIS ESTADÍSTICO INFERENCIAL")
print("=" * 80)

def analisis_estadistico_inferencial(df):
    """
    Implementa el análisis estadístico inferencial, incluyendo pruebas de normalidad, 
    análisis de correlaciones y validación de supuestos para los modelos de regresión.
    """
    print("=" * 80)
    print("ANÁLISIS ESTADÍSTICO INFERENCIAL")
    print("=" * 80)
    
    columnas_numericas = df.select_dtypes(include=['number']).columns
    
    # 1. Pruebas de normalidad (Shapiro-Wilk)
    print("\n1. PRUEBAS DE NORMALIDAD (Shapiro-Wilk)")
    print("-" * 50)
    for col in columnas_numericas:
        stat, p_value = stats.shapiro(df[col])
        print(f"• {col}:")
        print(f"  - Estadístico: {stat:.4f}")
        print(f"  - p-valor: {p_value:.4f}")
        print(f"  - Normal: {'Sí' if p_value > 0.05 else 'No'}")
    
    # 2. Análisis de correlaciones de Pearson
    print("\n2. ANÁLISIS DE CORRELACIONES DE PEARSON")
    print("-" * 50)
    correlaciones = df[columnas_numericas].corr()
    
    # Calcular p-valores para correlaciones
    for i, col1 in enumerate(columnas_numericas):
        for j, col2 in enumerate(columnas_numericas):
            if i < j:  # Evitar duplicados
                corr, p_val = stats.pearsonr(df[col1], df[col2])
                print(f"• {col1} vs {col2}:")
                print(f"  - Correlación: {corr:.4f}")
                print(f"  - p-valor: {p_val:.4f}")
                print(f"  - Significativa: {'Sí' if p_val < 0.05 else 'No'}")
    
    # 3. Análisis de homogeneidad de varianzas (Levene)
    print("\n3. ANÁLISIS DE HOMOGENEIDAD DE VARIANZAS")
    print("-" * 50)
    # Dividir datos en grupos para comparar varianzas
    if len(df) > 20:
        mitad = len(df) // 2
        for col in columnas_numericas:
            grupo1 = df[col][:mitad]
            grupo2 = df[col][mitad:]
            stat, p_val = stats.levene(grupo1, grupo2)
            print(f"• {col}:")
            print(f"  - Estadístico Levene: {stat:.4f}")
            print(f"  - p-valor: {p_val:.4f}")
            print(f"  - Varianzas iguales: {'Sí' if p_val > 0.05 else 'No'}")

# Ejecutar análisis estadístico inferencial
analisis_estadistico_inferencial(df)

# ============================================================================
# MÓDULO 5: MODELADO DE MACHINE LEARNING
# ============================================================================
print("\n" + "=" * 80)
print("MÓDULO 5: MODELADO DE MACHINE LEARNING")
print("=" * 80)

def preparar_datos_modelado(df):
    """
    Se centra en la implementación y evaluación de modelos de machine learning, 
    utilizando scikit-learn para algoritmos tradicionales y XGBoost para modelos avanzados de boosting.
    """
    print("=== PREPARACIÓN DE DATOS PARA MODELADO ===")
    
    # Seleccionar variables predictoras y variable objetivo
    X = df.drop(['Total_egg_production', 'Fecha'], axis=1, errors='ignore')
    y = df['Total_egg_production']
    
    print(f"Variables predictoras: {list(X.columns)}")
    print(f"Variable objetivo: Total_egg_production")
    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nConjunto de entrenamiento: {X_train.shape[0]} registros")
    print(f"Conjunto de prueba: {X_test.shape[0]} registros")
    
    return X_train, X_test, y_train, y_test, X

def entrenar_modelos(X_train, X_test, y_train, y_test, X):
    """
    Entrena y evalúa diferentes modelos de machine learning
    """
    resultados = {}
    
    # Modelo 1: XGBRegressor (Boosting)
    print("\n=== MODELO 1: XGBRegressor ===")
    xgb = XGBRegressor(random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_score = xgb.score(X_test, y_test)
    xgb_cv = cross_val_score(xgb, X_train, y_train, cv=5)
    
    print(f"R² Score: {xgb_score:.4f}")
    print(f"Promedio Validación Cruzada: {np.mean(xgb_cv):.4f}")
    print(f"Desviación Estándar CV: {np.std(xgb_cv):.4f}")
    
    # Gráfico de predicciones XGB
    plt.figure(figsize=(10, 6))
    plt.plot(xgb_pred, color='blue', label='Predicciones XGB')
    plt.plot(y_test.values, color='red', label='Valores Reales')
    plt.title('Predicciones vs Valores Reales - XGBRegressor')
    plt.xlabel('Índice')
    plt.ylabel('Producción de Huevos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/predicciones_xgb.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Importancia de variables en XGB
    importancias = pd.Series(xgb.feature_importances_, index=X.columns)
    plt.figure(figsize=(10, 6))
    importancias.sort_values().plot(kind='barh')
    plt.title('Importancia de Variables - XGBRegressor')
    plt.xlabel('Importancia')
    plt.tight_layout()
    plt.savefig('../results/importancia_variables_xgb.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    resultados['XGBRegressor'] = {
        'score': xgb_score,
        'cv_mean': np.mean(xgb_cv),
        'cv_std': np.std(xgb_cv),
        'importancias': importancias
    }
    
    # Modelo 2: KNN Regressor
    print("\n=== MODELO 2: KNN Regressor ===")
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_score = knn.score(X_test, y_test)
    knn_cv = cross_val_score(knn, X_train, y_train, cv=5)
    
    print(f"R² Score: {knn_score:.4f}")
    print(f"Promedio Validación Cruzada: {np.mean(knn_cv):.4f}")
    print(f"Desviación Estándar CV: {np.std(knn_cv):.4f}")
    
    # Gráfico de predicciones KNN
    plt.figure(figsize=(10, 6))
    plt.plot(knn_pred, color='green', label='Predicciones KNN')
    plt.plot(y_test.values, color='red', label='Valores Reales')
    plt.title('Predicciones vs Valores Reales - KNN')
    plt.xlabel('Índice')
    plt.ylabel('Producción de Huevos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/predicciones_knn.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    resultados['KNN Regressor'] = {
        'score': knn_score,
        'cv_mean': np.mean(knn_cv),
        'cv_std': np.std(knn_cv)
    }
    
    # Modelo 3: Naive Bayes (para comparación)
    print("\n=== MODELO 3: Naive Bayes ===")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb_pred = gnb.predict(X_test)
    gnb_score = gnb.score(X_test, y_test)
    
    # Convertir a categorías para validación cruzada
    y_train_cat = y_train.astype('category')
    gnb_cv = cross_val_score(gnb, X_train, y_train_cat, cv=5, scoring='accuracy')
    
    print(f"R² Score: {gnb_score:.4f}")
    print(f"Promedio Validación Cruzada: {np.mean(gnb_cv):.4f}")
    print(f"Desviación Estándar CV: {np.std(gnb_cv):.4f}")
    
    # Gráfico de predicciones Naive Bayes
    plt.figure(figsize=(10, 6))
    plt.plot(gnb_pred, color='orange', label='Predicciones Naive Bayes')
    plt.plot(y_test.values, color='red', label='Valores Reales')
    plt.title('Predicciones vs Valores Reales - Naive Bayes')
    plt.xlabel('Índice')
    plt.ylabel('Producción de Huevos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/predicciones_naive_bayes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    resultados['Naive Bayes'] = {
        'score': gnb_score,
        'cv_mean': np.mean(gnb_cv),
        'cv_std': np.std(gnb_cv)
    }
    
    return resultados

# Preparar datos y entrenar modelos
X_train, X_test, y_train, y_test, X = preparar_datos_modelado(df)
resultados_modelos = entrenar_modelos(X_train, X_test, y_train, y_test, X)

# ============================================================================
# MÓDULO 6: INTERPRETACIÓN DE RESULTADOS Y RECOMENDACIONES
# ============================================================================
print("\n" + "=" * 80)
print("MÓDULO 6: INTERPRETACIÓN DE RESULTADOS Y RECOMENDACIONES")
print("=" * 80)

def interpretar_resultados(resultados_modelos):
    """
    Presenta la interpretación de resultados y la generación de recomendaciones 
    prácticas basadas en los hallazgos del análisis.
    """
    print("=" * 80)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 80)
    
    # Crear tabla de comparación
    tabla_resultados = []
    for modelo, metricas in resultados_modelos.items():
        tabla_resultados.append({
            'Modelo': modelo,
            'R² Score': metricas['score'],
            'CV Mean': metricas['cv_mean'],
            'CV Std': metricas['cv_std']
        })
    
    df_resultados = pd.DataFrame(tabla_resultados)
    print("\nTabla de Comparación de Modelos:")
    print(df_resultados.to_string(index=False))
    
    # Gráfico de comparación
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    modelos = df_resultados['Modelo'].values
    scores = df_resultados['R² Score'].values
    cv_means = df_resultados['CV Mean'].values
    
    # R² Score
    ax1.bar(modelos, scores, color=['blue', 'green', 'orange'])
    ax1.set_title('Comparación de R² Score')
    ax1.set_ylabel('R² Score')
    ax1.grid(True, alpha=0.3)
    
    # CV Mean
    ax2.bar(modelos, cv_means, color=['blue', 'green', 'orange'])
    ax2.set_title('Comparación de Validación Cruzada')
    ax2.set_ylabel('CV Mean')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/comparacion_modelos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Recomendaciones prácticas
    print("\n" + "=" * 80)
    print("RECOMENDACIONES PRÁCTICAS BASADAS EN LOS HALLAZGOS (OE5)")
    print("=" * 80)
    
    mejor_modelo = df_resultados.loc[df_resultados['R² Score'].idxmax(), 'Modelo']
    mejor_score = df_resultados.loc[df_resultados['R² Score'].idxmax(), 'R² Score']
    
    print(f"\n1. SELECCIÓN DE MODELO:")
    print(f"   • El modelo {mejor_modelo} mostró el mejor desempeño con un R² de {mejor_score:.4f}")
    print(f"   • Se recomienda su uso para predecir la producción de huevos en función de variables ambientales")
    
    print(f"\n2. VARIABLES AMBIENTALES MÁS INFLUYENTES:")
    if mejor_modelo == 'XGBRegressor':
        importancias = resultados_modelos['XGBRegressor']['importancias']
        top_variables = importancias.nlargest(3)
        print(f"   • Las variables más importantes según XGBoost son:")
        for var, imp in top_variables.items():
            print(f"     - {var}: {imp:.4f}")
    
    print(f"\n3. RECOMENDACIONES DE GESTIÓN:")
    print(f"   • Monitorear y ajustar las variables ambientales más influyentes para maximizar la producción")
    print(f"   • Implementar sistemas de control ambiental automatizados puede mejorar la precisión de las predicciones")
    print(f"   • Realizar validaciones periódicas del modelo con nuevos datos para mantener la precisión")
    print(f"   • Adaptar el modelo a cambios en el entorno avícola según sea necesario")
    
    print(f"\n4. APLICACIONES PRÁCTICAS:")
    print(f"   • Optimización de condiciones ambientales en granjas avícolas")
    print(f"   • Predicción de producción para planificación logística")
    print(f"   • Detección temprana de problemas ambientales que afecten la producción")
    print(f"   • Mejora de la eficiencia productiva mediante control ambiental inteligente")
    
    print(f"\n5. LIMITACIONES Y CONSIDERACIONES:")
    print(f"   • El modelo se basa en datos históricos y puede requerir actualizaciones")
    print(f"   • Las condiciones ambientales pueden variar significativamente entre granjas")
    print(f"   • Se recomienda validar el modelo en diferentes contextos productivos")
    print(f"   • Considerar factores adicionales como genética, alimentación y manejo sanitario")
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETO FINALIZADO")
    print("=" * 80)
    print("Se han generado los siguientes archivos:")
    print("• distribuciones_variables_numericas.png")
    print("• boxplots_variables_numericas.png") 
    print("• matriz_correlaciones.png")
    print("• scatter_plots.png")
    print("• predicciones_xgb.png")
    print("• predicciones_knn.png")
    print("• predicciones_naive_bayes.png")
    print("• importancia_variables_xgb.png")
    print("• comparacion_modelos.png")
    print("\nLos resultados están listos para incluir en tu tesis.")

# Ejecutar interpretación de resultados
interpretar_resultados(resultados_modelos)

print("\n" + "=" * 80)
print("¡ANÁLISIS COMPLETO FINALIZADO EXITOSAMENTE!")
print("=" * 80)
print("El script ha ejecutado todos los módulos del análisis:")
print("✓ Módulo 1: Carga y Exploración Inicial de Datos")
print("✓ Módulo 2: Limpieza y Preprocesamiento")
print("✓ Módulo 3: Análisis Exploratorio de Datos (EDA)")
print("✓ Módulo 4: Análisis Estadístico Inferencial")
print("✓ Módulo 5: Modelado de Machine Learning")
print("✓ Módulo 6: Interpretación y Recomendaciones")
print("\nTodos los resultados y gráficos han sido guardados en la carpeta 'results/'")
print("La documentación completa está disponible en la carpeta 'docs/'") 