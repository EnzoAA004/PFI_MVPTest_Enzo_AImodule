# Arquitectura técnica del producto

## Decisión general

La arquitectura del producto será híbrida. El desarrollo inicial del motor de inteligencia artificial se realizará en Python y Google Colab, mientras que la capa de producto final se integrará con un backend principal en Spring Boot.

## Capas principales

### 1. Backend principal

Tecnología prevista: Spring Boot / Java.

Responsabilidades:

- APIs del producto.
- Gestión de casos.
- Estados de revisión.
- Persistencia.
- Trazabilidad.
- Salida estructurada editable.
- Orquestación del motor Python.
- Usuarios y permisos si corresponde.

No se usará FastAPI como backend principal porque el equipo prefiere y domina Spring Boot.

### 2. Motor de IA y procesamiento

Tecnología prevista: Python, inicialmente en Google Colab.

Responsabilidades:

- Lectura de imágenes médicas.
- Preprocesamiento.
- Segmentación.
- Métricas.
- Mediciones geométricas.
- Exportación de resultados técnicos en CSV/JSON.

La integración futura con Spring Boot podrá resolverse mediante archivos, JSON, endpoint interno o worker Python. Al inicio se prioriza no sobrediseñar microservicios.

### 3. Frontend futuro

Tecnología prevista: React.

Responsabilidades:

- Visualizar imagen.
- Visualizar máscaras o contornos.
- Mostrar mediciones.
- Mostrar estados de revisión.
- Permitir edición de salida estructurada.

Para visualización médica se puede tomar como referencia OHIF / Cornerstone3D, pero no se desarrollará un visor DICOM completo desde cero en la primera etapa.

### 4. Persistencia futura

Tecnología prevista: PostgreSQL + almacenamiento de archivos.

Responsabilidades:

- Casos.
- Resultados.
- Estados de revisión.
- Trazabilidad.
- Metadatos.
- Rutas de imágenes, máscaras, figuras y salidas exportables.

## Flujo técnico inicial en Colab

```text
Dataset público / Drive
    ↓
Lectura de imagen médica
    ↓
Inspección de metadatos
    ↓
Preprocesamiento
    ↓
Segmentación sagital
    ↓
Métricas y mediciones
    ↓
Visualización y evidencia
    ↓
CSV / JSON exportable
    ↓
Consumo futuro por Spring Boot
```

## Flujo funcional del producto

```text
Carga de estudio
    ↓
Segmentación automática
    ↓
Visualización imagen + máscara
    ↓
Mediciones geométricas trazables
    ↓
Revisión profesional
    ↓
Salida estructurada editable
    ↓
Persistencia y auditoría
```

## Plano sagital y módulo axial

El núcleo inicial del MVP sigue siendo sagital por disponibilidad de dataset y alcance académico. El plano axial se considera un módulo complementario importante, derivado del user research, pero debe evaluarse mediante un spike técnico antes de incorporarlo al MVP ampliado.

## Decisión clave

Los notebooks de Colab deben construir evidencia técnica verificable, pero no deben mezclar lógica de producto de Spring Boot. El motor Python debe exportar resultados claros y trazables para que luego puedan ser consumidos por el backend.
