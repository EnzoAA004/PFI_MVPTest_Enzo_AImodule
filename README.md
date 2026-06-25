# PFI MVP Test Enzo

Prototipo académico para análisis asistido de resonancias magnéticas lumbares sagitales.

El objetivo del proyecto es construir un flujo reproducible que permita:

- cargar y preprocesar RM lumbares sagitales;
- segmentar estructuras anatómicas relevantes;
- calcular mediciones geométricas simples derivadas de máscaras;
- visualizar resultados con superposición de máscaras;
- exportar una salida estructurada, editable y revisable por un profesional.

## Alcance

Este repositorio corresponde a un MVP técnico para el Proyecto Final de Ingeniería. El sistema no emite diagnósticos, no clasifica patologías, no recomienda tratamientos y no reemplaza el criterio profesional.

## Dataset

La fuente principal prevista para el MVP es SPIDER, un dataset público de RM lumbar con máscaras de referencia para vértebras, discos intervertebrales y canal espinal.

Los datos no deben subirse al repositorio. Se recomienda almacenarlos en Google Drive u otra ubicación externa y configurar las rutas desde Colab o variables de entorno.

## Flujo de trabajo recomendado

1. Desarrollar código localmente o con Codex sobre este repositorio.
2. Subir cambios a GitHub con `git push`.
3. Ejecutar notebooks en Google Colab clonando o actualizando este repositorio.
4. Mantener datasets, checkpoints y resultados pesados fuera de GitHub.

## Estructura inicial

```text
src/lumbar_mri/      Código reutilizable del proyecto
notebooks/           Notebooks de ejecución y experimentación en Colab
app/                 Interfaz futura del producto
tests/               Tests unitarios básicos
docs/                Documentación técnica y de producto
outputs/             Resultados locales ignorados por Git
models/              Checkpoints locales ignorados por Git
data/                Datasets locales ignorados por Git
```

## Instalación en modo desarrollo

```bash
pip install -e .
```

## Tests

```bash
pytest
```

## Nota importante

Toda medición generada por el sistema debe interpretarse como una medición geométrica derivada de una máscara segmentada. La interpretación clínica queda fuera del alcance del MVP y debe permanecer bajo revisión profesional.
