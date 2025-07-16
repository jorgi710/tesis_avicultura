# Instrucciones para Ejecutar el Análisis de Avicultura

## 🚀 Ejecución Rápida

### Opción 1: Ejecutar desde la carpeta principal
```bash
cd tesis_avicultura
python ejecutar_analisis.py
```

### Opción 2: Ejecutar el script principal directamente
```bash
cd tesis_avicultura/src
python analisis_completo_avicultura.py
```

## 📋 Requisitos Previos

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verificar que los datos estén en su lugar:**
   - `data/Egg_Production.csv` debe existir

## 📁 Estructura del Proyecto

```
tesis_avicultura/
├── 📊 data/                    # Datos originales y procesados
│   ├── Egg_Production.csv      # Datos originales
│   └── Egg_Production_Clean.csv # Datos limpios (generado)
├── 🔧 src/                     # Código fuente
│   ├── analisis_completo_avicultura.py  # Script principal
│   ├── ob1.py                 # Análisis descriptivo original
│   └── obj4.py                # Modelado predictivo original
├── 📈 results/                 # Resultados y gráficos
│   └── [archivos .png generados]
├── 📚 docs/                    # Documentación
│   └── metodologia.md         # Metodología detallada
├── 📓 notebooks/               # Cuadernos (opcional)
├── 📋 requirements.txt         # Dependencias
├── 📖 README.md               # Documentación principal
├── 🚀 ejecutar_analisis.py    # Script de ejecución
└── 📝 INSTRUCCIONES.md        # Este archivo
```

## 🎯 Qué hace el análisis

El script ejecuta **6 módulos** completos:

1. **📊 Carga y Exploración Inicial de Datos**
   - Carga del dataset
   - Exploración inicial
   - Análisis de estructura

2. **🧹 Limpieza y Preprocesamiento**
   - Detección de outliers
   - Limpieza de datos
   - Preparación para modelado

3. **📈 Análisis Exploratorio de Datos (EDA)**
   - Estadísticas descriptivas
   - Generación de gráficos
   - Análisis de correlaciones

4. **📊 Análisis Estadístico Inferencial**
   - Pruebas de normalidad
   - Análisis de correlaciones
   - Validación de supuestos

5. **🤖 Modelado de Machine Learning**
   - XGBRegressor (Boosting)
   - KNN Regressor
   - Naive Bayes
   - Validación cruzada

6. **💡 Interpretación y Recomendaciones**
   - Comparación de modelos
   - Análisis de importancia
   - Recomendaciones prácticas

## 📊 Archivos Generados

### Gráficos de Análisis:
- `distribuciones_variables_numericas.png`
- `boxplots_variables_numericas.png`
- `matriz_correlaciones.png`
- `scatter_plots.png`

### Resultados de Modelado:
- `predicciones_xgb.png`
- `predicciones_knn.png`
- `predicciones_naive_bayes.png`
- `importancia_variables_xgb.png`
- `comparacion_modelos.png`

## 🔧 Solución de Problemas

### Error: "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError: '../data/Egg_Production.csv'"
Verifica que el archivo `Egg_Production.csv` esté en la carpeta `data/`

### Error: "No module named 'xgboost'"
```bash
pip install xgboost
```

## 📝 Para la Tesis

El análisis está diseñado para incluirse directamente en tu tesis:

1. **Código fuente**: Documentado en español
2. **Metodología**: Explicada en `docs/metodologia.md`
3. **Resultados**: Gráficos listos para incluir
4. **Recomendaciones**: Basadas en hallazgos científicos

## 🎓 Citas para la Tesis

Puedes citar este trabajo como:
> "Análisis de Producción Avícola: Efectos Ambientales en la Producción de Huevos. 
> Implementación de modelos de machine learning para optimización productiva. 
> Ayala, J.A. (2024)."

¡El análisis está listo para tu tesis! 🎯 