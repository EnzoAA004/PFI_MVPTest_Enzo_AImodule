# Arquitectura inicial

## Flujo general

```text
RM lumbar sagital
    ↓
Carga y validación
    ↓
Preprocesamiento
    ↓
Modelo de segmentación
    ↓
Postprocesamiento
    ↓
Métricas y mediciones
    ↓
Visualización
    ↓
Salida estructurada editable
```

## Capas del proyecto

### 1. Capa experimental

Ubicada en `notebooks/`. Sirve para ejecutar el proyecto en Colab, probar modelos, visualizar resultados y documentar experimentos.

### 2. Capa de código reutilizable

Ubicada en `src/lumbar_mri/`. Contiene funciones, modelos y utilidades importables desde notebooks, scripts y apps.

### 3. Capa de producto

Ubicada en `app/`. Inicialmente se implementará con Streamlit como demo simple.

## Módulos actuales

- `config.py`: rutas y configuración general.
- `data/`: carga y preprocesamiento.
- `models/`: modelos de segmentación.
- `training/`: métricas y entrenamiento futuro.
- `measurements/`: mediciones geométricas desde máscaras.
- `visualization/`: overlays y gráficos.
- `reporting/`: exportación estructurada.

## Decisión clave

El notebook de Colab no debe contener toda la lógica del sistema. Debe importar funciones desde `src/lumbar_mri/` para mantener trazabilidad, reutilización y limpieza del código.
