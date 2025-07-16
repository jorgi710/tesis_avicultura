# Análisis de Producción Avícola: Efectos Ambientales en la Producción de Huevos

## 📋 Descripción del Proyecto

Este proyecto implementa un análisis completo de los efectos ambientales en la producción de huevos, siguiendo las mejores prácticas de ciencia de datos y documentación científica. El análisis incluye exploración de datos, limpieza, modelado de machine learning y generación de recomendaciones prácticas.

**Autor:** Jorge Andrés Ayala  
**Fecha:** 2024  
**Tipo:** Análisis de Ciencia de Datos para Tesis

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

## 📁 Estructura del Proyecto

```
tesis_avicultura/
├── 📊 data/                    # Datos originales y procesados
│   ├── Egg_Production.csv      # Datos originales
│   └── Egg_Production_Clean.csv # Datos limpios (generado automáticamente)
├── 🔧 src/                     # Código fuente modular
│   ├── analisis_completo_avicultura.py  # Script principal integrado
│   ├── ob1.py                 # Análisis descriptivo original
│   └── obj4.py                # Modelado predictivo original
├── 📈 results/                 # Resultados y visualizaciones
│   ├── distribuciones_variables_numericas.png
│   ├── boxplots_variables_numericas.png
│   ├── matriz_correlaciones.png
│   ├── scatter_plots.png
│   ├── predicciones_xgb.png
│   ├── predicciones_knn.png
│   ├── predicciones_naive_bayes.png
│   ├── importancia_variables_xgb.png
│   └── comparacion_modelos.png
├── 📚 docs/                    # Documentación
│   └── metodologia.md         # Metodología detallada para la tesis
├── 📓 notebooks/               # Cuadernos de análisis (opcional)
├── 📋 requirements.txt         # Dependencias del proyecto
├── 📖 README.md               # Este archivo
├── 🚀 ejecutar_analisis.py    # Script de ejecución simplificado
└── 📝 INSTRUCCIONES.md        # Guía paso a paso
```

## 🛠️ Instalación y Configuración

### 1. Clonar o descargar el proyecto
```bash
# Si tienes el proyecto en tu computadora
cd tesis_avicultura
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Verificar datos
Asegúrate de que el archivo `data/Egg_Production.csv` esté presente.

## 🎯 Módulos del Análisis

### 1. 📊 Carga y Exploración Inicial de Datos
- Carga del dataset de producción de huevos
- Exploración inicial con histogramas y diagramas de caja
- Análisis de valores nulos y duplicados
- Generación de fechas temporales

### 2. 🧹 Limpieza y Preprocesamiento
- Detección y eliminación de outliers usando método IQR
- Validación de integridad de datos
- Preparación para modelado
- Guardado de dataset limpio

### 3. 📈 Análisis Exploratorio de Datos (EDA)
- Distribuciones de variables numéricas
- Análisis de correlaciones
- Visualizaciones exploratorias
- Estadísticas descriptivas detalladas

### 4. 📊 Análisis Estadístico Inferencial
- Pruebas de normalidad (Shapiro-Wilk)
- Análisis de correlaciones de Pearson
- Validación de supuestos para modelos de regresión
- Análisis de homogeneidad de varianzas

### 5. 🤖 Modelado de Machine Learning
- **XGBRegressor**: Modelo de boosting avanzado
- **KNN Regressor**: Algoritmo de vecinos cercanos
- **Naive Bayes**: Modelo probabilístico
- Validación cruzada y evaluación de rendimiento

### 6. 💡 Interpretación y Recomendaciones
- Comparación de modelos
- Análisis de importancia de variables
- Recomendaciones prácticas para gestión avícola
- Aplicaciones prácticas y limitaciones

## 📊 Resultados Principales

### Modelos Evaluados:
- **XGBRegressor**: Modelo de boosting con mejor rendimiento
- **KNN Regressor**: Algoritmo de vecinos cercanos
- **Naive Bayes**: Modelo probabilístico para comparación

### Métricas de Evaluación:
- R² Score para cada modelo
- Validación cruzada (5-fold)
- Análisis de importancia de variables
- Comparación de predicciones vs valores reales

### Variables Ambientales Analizadas:
- Temperatura
- Humedad
- Intensidad de luz
- Ruido
- Producción total de huevos

## 📈 Archivos Generados

### Gráficos de Análisis Exploratorio:
- `distribuciones_variables_numericas.png` - Distribuciones de todas las variables
- `boxplots_variables_numericas.png` - Análisis de outliers
- `matriz_correlaciones.png` - Correlaciones entre variables
- `scatter_plots.png` - Gráficos de dispersión

### Resultados de Modelado:
- `predicciones_xgb.png` - Predicciones del modelo XGBoost
- `predicciones_knn.png` - Predicciones del modelo KNN
- `predicciones_naive_bayes.png` - Predicciones del modelo Naive Bayes
- `importancia_variables_xgb.png` - Importancia de variables en XGBoost
- `comparacion_modelos.png` - Comparación de rendimiento de modelos

### Datos Procesados:
- `Egg_Production_Clean.csv` - Dataset limpio y procesado

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

### Error: "matplotlib backend"
```bash
pip install matplotlib seaborn
```

## 📚 Documentación

### Archivos de Documentación:
- `README.md` - Este archivo principal
- `docs/metodologia.md` - Metodología detallada para la tesis
- `INSTRUCCIONES.md` - Guía paso a paso de ejecución
- `requirements.txt` - Lista de dependencias

### Comentarios en el Código:
- Todo el código está documentado en español
- Explicaciones detalladas de cada paso
- Referencias a métodos estadísticos utilizados
- Justificación de decisiones metodológicas

## 🎓 Para la Tesis

### Código Fuente:
El código fuente desarrollado para este análisis se encuentra disponible en un repositorio público de GitHub, el cual contiene el entorno virtual configurado con todas las bibliotecas necesarias para su ejecución.

### Estructura del Repositorio:
La estructura del repositorio sigue las mejores prácticas de organización de proyectos de ciencia de datos, tal como lo recomiendan Cookiecutter Data Science (Nüst et al., 2018).

### Flujo de Trabajo:
El script principal implementa un flujo de trabajo estructurado en seis módulos principales, desde la exploración inicial de datos mediante histogramas y diagramas de caja, hasta la estimación de correlaciones y la construcción del modelo predictivo.

### Reproducibilidad:
El archivo `requirements.txt` especifica las versiones exactas de todas las dependencias utilizadas, garantizando la reproducibilidad del entorno de ejecución.

## 📝 Citas y Referencias

Este proyecto sigue las mejores prácticas recomendadas por:
- **Cookiecutter Data Science** (Nüst et al., 2018)
- **Perkel (2018)** - Guía sobre buenas prácticas en programación científica

### Cita para la Tesis:
> "Análisis de Producción Avícola: Efectos Ambientales en la Producción de Huevos. 
> Implementación de modelos de machine learning para optimización productiva. 
> Ayala, J.A. (2024)."

## 🤝 Contribuciones

Este proyecto está diseñado para ser un análisis completo y reproducible. Si encuentras algún problema o tienes sugerencias de mejora, por favor:

1. Revisa la documentación en `docs/`
2. Verifica que todas las dependencias estén instaladas
3. Ejecuta el script de prueba: `python ejecutar_analisis.py`

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.

---

**¡El análisis está listo para incluir en tu tesis! 🎯**

Para más información, consulta `INSTRUCCIONES.md` o `docs/metodologia.md`. 