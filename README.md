# Análisis de Producción Avícola: Efectos Ambientales en la Producción de Huevos

## Descripción del Proyecto

Este proyecto implementa un análisis completo de los efectos ambientales en la producción de huevos, siguiendo las mejores prácticas de ciencia de datos y documentación científica. El análisis incluye exploración de datos, limpieza, modelado de machine learning y generación de recomendaciones prácticas.

## Estructura del Proyecto

```
tesis_avicultura/
├── data/                    # Datos originales y procesados
│   ├── Egg_Production.csv
│   └── Egg_Production_Clean.csv
├── src/                     # Código fuente modular
│   ├── ob1.py              # Análisis descriptivo
│   └── obj4.py             # Modelado predictivo
├── notebooks/               # Cuadernos de análisis
│   └── aviculture_analysis.ipynb
├── results/                 # Resultados y visualizaciones
│   ├── distribuciones_variables_numericas.png
│   ├── boxplots_variables_numericas.png
│   ├── matriz_correlaciones.png
│   └── ...
├── docs/                    # Documentación
├── requirements.txt         # Dependencias del proyecto
└── README.md               # Este archivo
```

## Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/username/aviculture-analysis
   cd aviculture-analysis
   ```

2. **Crear entorno virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar el análisis:**
   ```bash
   jupyter notebook notebooks/aviculture_analysis.ipynb
   ```

## Módulos del Análisis

### 1. Carga y Exploración Inicial de Datos
- Carga del dataset de producción de huevos
- Exploración inicial con histogramas y diagramas de caja
- Análisis de valores nulos y duplicados

### 2. Limpieza y Preprocesamiento
- Detección y eliminación de outliers usando método IQR
- Validación de integridad de datos
- Preparación para modelado

### 3. Análisis Exploratorio de Datos (EDA)
- Distribuciones de variables numéricas
- Análisis de correlaciones
- Visualizaciones exploratorias

### 4. Análisis Estadístico Inferencial
- Pruebas de normalidad (Shapiro-Wilk)
- Análisis de correlaciones de Pearson
- Validación de supuestos para modelos

### 5. Modelado de Machine Learning
- **XGBRegressor**: Modelo de boosting avanzado
- **KNN Regressor**: Algoritmo de vecinos cercanos
- **Naive Bayes**: Modelo probabilístico
- Validación cruzada y evaluación de rendimiento

### 6. Interpretación y Recomendaciones
- Comparación de modelos
- Análisis de importancia de variables
- Recomendaciones prácticas para gestión avícola

## Resultados Principales

- **Mejor modelo**: XGBRegressor con R² superior
- **Variables más influyentes**: Identificadas mediante análisis de importancia
- **Recomendaciones**: Optimización ambiental para maximizar producción

## Archivos Generados

- `distribuciones_variables_numericas.png`: Distribuciones de variables
- `boxplots_variables_numericas.png`: Análisis de outliers
- `matriz_correlaciones.png`: Correlaciones entre variables
- `predicciones_*.png`: Comparación de predicciones por modelo
- `importancia_variables_xgb.png`: Importancia de variables en XGBoost
- `comparacion_modelos.png`: Comparación de rendimiento de modelos

## Citas y Referencias

Este proyecto sigue las mejores prácticas recomendadas por:
- Cookiecutter Data Science (Nüst et al., 2018)
- Perkel (2018) - Guía sobre buenas prácticas en programación científica

## Autor

Jorge Andrés Ayala - Análisis de Producción Avícola

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles. 
