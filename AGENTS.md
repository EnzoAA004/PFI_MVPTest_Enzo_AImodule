# AGENTS.md

Este repositorio corresponde a un Proyecto Final de Ingeniería sobre análisis asistido de resonancias magnéticas lumbares.

## Objetivo del proyecto

Desarrollar un prototipo académico que permita:

- cargar y preprocesar RM lumbares en los planos definidos para el MVP;
- segmentar estructuras anatómicas relevantes;
- calcular mediciones geométricas simples derivadas de máscaras;
- visualizar máscaras superpuestas;
- exportar resultados estructurados y editables para revisión profesional.

## Arquitectura acordada

La arquitectura del producto será híbrida:

- Backend principal: Spring Boot / Java.
- Motor de IA y procesamiento: Python, inicialmente en Google Colab.
- Frontend futuro: React.
- Persistencia futura: PostgreSQL y almacenamiento de archivos.

No usar FastAPI como backend principal. Si se requiere integración inicial entre Spring Boot y Python, priorizar intercambio por archivos, JSON o worker Python simple antes que microservicios complejos.

## Restricciones importantes

- El sistema no debe emitir diagnósticos.
- El sistema no debe recomendar tratamientos.
- El sistema no debe reemplazar el criterio profesional.
- Las mediciones deben presentarse como valores geométricos derivados de máscaras.
- La salida debe ser revisable y editable.
- No usar datos sensibles ni identificables de pacientes.
- No subir datasets, checkpoints ni resultados pesados al repositorio.
- Priorizar herramientas open source.
- Mantener el proyecto como prototipo académico reproducible.
- El núcleo inicial del MVP es sagital; el plano axial debe tratarse como módulo complementario o spike técnico.

## Convenciones técnicas para Python / Colab

- Usar Python como lenguaje principal del motor de IA.
- Priorizar PyTorch, MONAI, nnU-Net o U-Net como baseline si es viable.
- Usar pydicom, NiBabel o SimpleITK para lectura de imágenes médicas según formato.
- Mantener funciones reutilizables dentro de `src/lumbar_mri/`.
- Mantener notebooks como capa de experimentación, evidencia y ejecución.
- Cada notebook debe tener secciones claras: objetivo, configuración, carga de datos, procesamiento, resultados, evidencia y conclusiones.
- Agregar tests para métricas, mediciones y exportación cuando sea posible.
- Documentar decisiones técnicas en `docs/decisiones_tecnicas.md`.
- Evitar sobreingeniería: primero pipeline completo, luego optimización.

## Flujo con Colab

Colab debe clonar o actualizar el repositorio desde GitHub. Los datos, checkpoints y outputs deben estar en Google Drive u otra ubicación externa, no dentro del repositorio.

Cada avance debe dejar una evidencia verificable: notebook, figura, tabla CSV, JSON exportado, métrica, decisión técnica o limitación documentada.

## Validación

Cuando se modifique código funcional, intentar ejecutar:

```bash
pytest
```

Si se agregan funciones de inferencia, métricas o mediciones, agregar tests mínimos con arrays sintéticos.

## Estilo

- Código claro y simple.
- Funciones pequeñas y reutilizables.
- Nombres explícitos.
- Documentar supuestos.
- Separar código de experimentos.
- No mezclar lógica de backend Spring Boot dentro de notebooks de IA.
