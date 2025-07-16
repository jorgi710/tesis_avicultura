#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de ejecución para el análisis de producción avícola
==========================================================

Este script ejecuta el análisis completo de efectos ambientales en la producción de huevos.

Autor: Jorge Andrés Ayala
Fecha: 2024
"""

import sys
import os

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """
    Función principal que ejecuta el análisis completo
    """
    print("=" * 80)
    print("INICIANDO ANÁLISIS DE PRODUCCIÓN AVÍCOLA")
    print("=" * 80)
    
    try:
        # Importar y ejecutar el análisis completo
        import analisis_completo_avicultura
        
        print("\n" + "=" * 80)
        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        print("📁 Archivos generados en la carpeta 'results/':")
        print("   • distribuciones_variables_numericas.png")
        print("   • boxplots_variables_numericas.png")
        print("   • matriz_correlaciones.png")
        print("   • scatter_plots.png")
        print("   • predicciones_xgb.png")
        print("   • predicciones_knn.png")
        print("   • predicciones_naive_bayes.png")
        print("   • importancia_variables_xgb.png")
        print("   • comparacion_modelos.png")
        
        print("\n📊 Datos procesados guardados en 'data/Egg_Production_Clean.csv'")
        print("📚 Documentación disponible en 'docs/'")
        print("🔧 Código fuente en 'src/'")
        
        print("\n🎯 El análisis está listo para incluir en tu tesis!")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {str(e)}")
        print("Verifica que todas las dependencias estén instaladas:")
        print("pip install -r requirements.txt")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 