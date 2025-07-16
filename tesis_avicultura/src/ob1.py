import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import seaborn as sns

# Cargar el dataset
df = pd.read_csv('Egg_Production.csv')  # Asegúrate de tener este archivo en el mismo directorio

# Agregar columna de fechas diarias desde la última fecha hacia atrás
fecha_final = datetime(2023, 6, 29, 19, 50)
df["Fecha"] = [fecha_final - timedelta(days=i) for i in range(len(df))][::-1]

# Mostrar información general
print("Información general del DataFrame:")
print(df.info())
print("\nPrimeros registros:")
print(df.head())
print("\nResumen estadístico:")
print(df.describe())
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Contar duplicados
duplicados = df.duplicated().sum()
print(f"\nNúmero de registros duplicados: {duplicados}")

# NOTA: No eliminamos duplicados, porque ahora con la columna Fecha no hay duplicados exactos.

# Eliminar outliers usando el método IQR para cada columna numérica (excepto Fecha)
columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns
Q1 = df[columnas_numericas].quantile(0.25)
Q3 = df[columnas_numericas].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[columnas_numericas] < (Q1 - 1.5 * IQR)) | (df[columnas_numericas] > (Q3 + 1.5 * IQR))).any(axis=1)]

print(f"\nDimensiones finales después de limpiar outliers: {df.shape}")

# ============================================================================
# FUNCIONES PARA ANÁLISIS DESCRIPTIVO DETALLADO
# ============================================================================

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

def generar_tabla_analisis(df):
    """
    Genera una tabla formateada con el análisis del dataset para la tesis
    """
    print("\n" + "=" * 80)
    print("TABLA DE ANÁLISIS PARA LA TESIS")
    print("=" * 80)
    
    columnas_numericas = df.select_dtypes(include=['number']).columns
    
    # Crear tabla de estadísticas descriptivas
    tabla_stats = []
    for col in columnas_numericas:
        stats = {
            'Variable': col,
            'Media': f"{df[col].mean():.4f}",
            'Mediana': f"{df[col].median():.4f}",
            'Desv. Est.': f"{df[col].std():.4f}",
            'Mínimo': f"{df[col].min():.4f}",
            'Máximo': f"{df[col].max():.4f}",
            'CV (%)': f"{(df[col].std() / df[col].mean()) * 100:.2f}",
            'Asimetría': f"{df[col].skew():.4f}",
            'Curtosis': f"{df[col].kurtosis():.4f}"
        }
        tabla_stats.append(stats)
    
    # Imprimir tabla formateada
    df_stats = pd.DataFrame(tabla_stats)
    print("\nTabla 1: Estadísticas Descriptivas de las Variables Numéricas")
    print("-" * 100)
    print(df_stats.to_string(index=False))
    
    # Tabla de correlaciones
    print("\n\nTabla 2: Matriz de Correlaciones")
    print("-" * 50)
    correlaciones = df[columnas_numericas].corr()
    print(correlaciones.round(4).to_string())
    
    # Tabla de outliers
    print("\n\nTabla 3: Análisis de Outliers")
    print("-" * 60)
    tabla_outliers = []
    for col in columnas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers_inferiores = len(df[df[col] < (Q1 - 1.5 * IQR)])
        outliers_superiores = len(df[df[col] > (Q3 + 1.5 * IQR)])
        total_outliers = outliers_inferiores + outliers_superiores
        porcentaje = (total_outliers / len(df)) * 100
        
        tabla_outliers.append({
            'Variable': col,
            'Outliers Inferiores': outliers_inferiores,
            'Outliers Superiores': outliers_superiores,
            'Total Outliers': total_outliers,
            'Porcentaje (%)': f"{porcentaje:.2f}"
        })
    
    df_outliers = pd.DataFrame(tabla_outliers)
    print(df_outliers.to_string(index=False))

def analisis_temporal(df):
    """
    Análisis temporal del dataset
    """
    print("\n" + "=" * 80)
    print("ANÁLISIS TEMPORAL DEL DATASET")
    print("=" * 80)
    
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df['Año'] = df['Fecha'].dt.year
        df['Mes'] = df['Fecha'].dt.month
        df['Día'] = df['Fecha'].dt.day
        
        print(f"• Período de análisis: {df['Fecha'].min().strftime('%Y-%m-%d')} a {df['Fecha'].max().strftime('%Y-%m-%d')}")
        print(f"• Duración total: {(df['Fecha'].max() - df['Fecha'].min()).days} días")
        
        # Análisis por año
        if len(df['Año'].unique()) > 1:
            print("\n• Análisis por año:")
            for año in sorted(df['Año'].unique()):
                datos_año = df[df['Año'] == año]
                print(f"  - {año}: {len(datos_año)} registros")
        
        # Análisis por mes
        print("\n• Análisis por mes:")
        for mes in sorted(df['Mes'].unique()):
            datos_mes = df[df['Mes'] == mes]
            print(f"  - Mes {mes}: {len(datos_mes)} registros")

def generar_graficos_analisis(df):
    """
    Genera gráficos de análisis para la tesis
    """
    print("\nGenerando gráficos de análisis...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    
    # 1. Distribución de variables numéricas
    columnas_numericas = df.select_dtypes(include=['number']).columns
    n_cols = len(columnas_numericas)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(columnas_numericas):
        if i < len(axes):
            axes[i].hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribución de {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frecuencia')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distribuciones_variables_numericas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Boxplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(columnas_numericas):
        if i < len(axes):
            axes[i].boxplot(df[col], patch_artist=True, boxprops=dict(facecolor='lightblue'))
            axes[i].set_title(f'Boxplot de {col}')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('boxplots_variables_numericas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Mapa de calor de correlaciones
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[columnas_numericas].corr(), annot=True, cmap='coolwarm', 
                fmt='.3f', square=True, cbar_kws={'shrink': 0.8})
    plt.title('Matriz de Correlaciones', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('matriz_correlaciones.png', dpi=300, bbox_inches='tight')
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
        plt.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# EJECUTAR ANÁLISIS COMPLETO
# ============================================================================

print("INICIANDO ANÁLISIS COMPLETO DEL DATASET")
print("=" * 80)

# Ejecutar análisis descriptivo
analisis_descriptivo_completo(df)

# Generar tabla de análisis para la tesis
generar_tabla_analisis(df)

# Análisis temporal
analisis_temporal(df)

# Generar gráficos
generar_graficos_analisis(df)

print("\n" + "=" * 80)
print("ANÁLISIS COMPLETO FINALIZADO")
print("=" * 80)
print("Se han generado los siguientes archivos:")
print("• distribuciones_variables_numericas.png")
print("• boxplots_variables_numericas.png") 
print("• matriz_correlaciones.png")
print("• scatter_plots.png")
print("\nLos resultados están listos para incluir en tu tesis.")
