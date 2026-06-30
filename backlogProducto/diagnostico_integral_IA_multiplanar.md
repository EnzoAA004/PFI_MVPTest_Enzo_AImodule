# Diagnóstico integral — IA multiplanar RM lumbar

**Proyecto:** Plataforma de apoyo al análisis estructural de RM lumbar mediante segmentación anatómica asistida por IA  
**Fecha:** 2026-06-30  
**Autores:** Enzo Asplanatti y Francisco Fabrello  
**Estado general:** etapa de investigación y validación de datos cerrada para sagital y axial 2D; pendiente integración/agente/3D.

---

## 1. Resumen ejecutivo

El proyecto ya cuenta con dos líneas de segmentación 2D validadas experimentalmente:

1. **Sagital sobre SPIDER**, con desempeño sólido en segmentación multiclase agrupada.
2. **Axial T2 sobre Al-Kafri/Sudirman**, rescatado técnicamente mediante pairing oficial y labels finales.

El avance más importante reciente fue la recuperación del dataset axial Al-Kafri/Sudirman: se reconstruyó el emparejamiento oficial, se identificaron los labels finales correctos y se entrenó un modelo axial T2 final limpio.

**Diagnóstico clave:**

- Ya existen modelos 2D separados viables para sagital y axial.
- Todavía no existe una reconstrucción 3D multiplanar real.
- Para 3D real se requiere aplicar ambos modelos sobre estudios del mismo paciente con geometría DICOM compatible.
- El siguiente paso es consolidar el pipeline de inferencia y el agente/orquestador.

---

## 2. Módulo sagital — SPIDER

### Estado

**Avanzado / bastante cerrado.**

Dataset: **SPIDER**  
Plano: **sagital**  
Tarea: segmentación anatómica lumbar  
Modelo: U-Net 2D

### Resultados principales

| Experimento | Resultado |
|---|---:|
| Baseline binario mejorado — Val Dice | 0.8875 |
| Baseline binario mejorado — Holdout Dice | 0.8816 |
| Multiclase agrupado — Val Dice macro sin fondo | 0.8392 |
| Multiclase agrupado — Holdout Dice macro sin fondo | 0.8311 |
| Multiclase agrupado — Holdout IoU macro sin fondo | 0.7309 |

### Clases sagitales agrupadas

- Fondo.
- Estructuras óseas / vértebras agrupadas.
- Canal espinal.
- Discos intervertebrales.

### Pendiente para cerrar sagital

- Entrenamiento final limpio estilo E10.
- Checkpoint final sagital.
- Reporte final con métricas por clase.
- Figuras cualitativas finales.
- Wrapper de inferencia sagital reusable.

---

## 3. Módulo axial T2 — Al-Kafri/Sudirman

### Estado

**Rescatado y viable.**

Se logró:

1. Inventariar el dataset.
2. Reconstruir el pairing oficial con `Slices Numbers.csv`, `T1_Subfolders.csv` y `T2_Subfolders.csv`.
3. Identificar que los labels entrenables no eran los intermedios, sino los finales.
4. Usar `05_Final_Ground_Truth_Data/Label_Images/L1_XXXX_DY.png`.
5. Curar 610 pares T2.
6. Entrenar baseline E9.
7. Entrenar modelo final limpio E10.

### Resultados E10 — modelo axial T2 final limpio

| Métrica | Validación | Test |
|---|---:|---:|
| Dice macro sin fondo | 0.7054 | 0.6587 |
| IoU macro sin fondo | 0.6364 | 0.5628 |
| Dice macro excluyendo raw_0 | 0.8817 | 0.8167 |
| IoU macro excluyendo raw_0 | 0.7955 | 0.7001 |

### Distribución de datos axial T2

| Split | Pares | Casos |
|---|---:|---:|
| Train | 427 | 128 |
| Val | 81 | 27 |
| Test | 102 | 29 |
| Total | 610 | 184 |

### Métricas por clase en test

| Clase | Dice | IoU |
|---|---:|---:|
| background_250 | 0.9930 | 0.9860 |
| raw_0 | 0.0264 | 0.0134 |
| raw_50 | 0.9348 | 0.8775 |
| raw_100 | 0.8485 | 0.7369 |
| raw_150 | 0.7997 | 0.6663 |
| raw_200 | 0.6840 | 0.5197 |

### Interpretación

El modelo axial T2 aprendió correctamente las clases principales. La clase `raw_0` presenta bajo desempeño y debe ser analizada como clase minoritaria, clase de borde, artefacto o categoría anatómica poco representada. Por eso se reportan métricas con y sin `raw_0`.

### Pendiente para cerrar axial

- Mapear formalmente `raw_50`, `raw_100`, `raw_150`, `raw_200` y `raw_0` a nombres anatómicos.
- Decidir tratamiento final de `raw_0`.
- Generar tabla oficial de clases para el informe.
- Exportar checkpoint final y wrapper de inferencia axial.

---

## 4. Uso sagital + axial en simultáneo

### Lo que ya es viable

Se puede diseñar un sistema con dos pipelines:

1. Si la serie es sagital → ejecutar modelo sagital SPIDER.
2. Si la serie es axial T2 → ejecutar modelo axial Al-Kafri/Sudirman.
3. Generar máscaras, overlays y métricas de calidad.
4. Devolver resultados al profesional para revisión.

### Lo que todavía no está resuelto

No se puede fusionar directamente SPIDER + Al-Kafri para construir un 3D anatómico real porque son datasets distintos y pacientes distintos.

Para 3D real se necesita un mismo estudio DICOM que contenga series sagitales y axiales del mismo paciente, con metadata geométrica compatible.

---

## 5. Modelación 3D — estado y requisitos

### Estado actual

Modelos 2D funcionales, pero reconstrucción 3D no implementada.

### Pipeline 3D necesario

- Leer series DICOM completas.
- Agrupar series por plano.
- Ordenar slices.
- Extraer metadata:
  - `ImagePositionPatient`
  - `ImageOrientationPatient`
  - `PixelSpacing`
  - `SliceThickness`
  - `SpacingBetweenSlices`
- Convertir máscaras 2D a coordenadas físicas 3D.
- Resamplear a volumen común.
- Fusionar máscaras sagitales y axiales.
- Resolver conflictos entre planos.
- Generar volumen 3D.
- Generar malla con marching cubes.
- Exportar STL/OBJ/GLB.
- Visualizar y validar.

---

## 6. Agente IA — estado y propuesta

El agente no es un nuevo segmentador: es el orquestador del flujo.

### Funciones esperadas

- Recibir una RM lumbar.
- Leer metadata DICOM.
- Detectar series disponibles.
- Identificar plano.
- Seleccionar modelo:
  - modelo sagital.
  - modelo axial T2.
- Ejecutar inferencia.
- Postprocesar máscaras.
- Calcular calidad:
  - área esperada.
  - cantidad de componentes.
  - continuidad anatómica.
  - confianza.
  - outliers.
- Generar overlays.
- Generar reporte.
- Permitir revisión humana.
- Preparar reconstrucción 3D si hay ambas series.

### Estado

Pendiente. Hoy las piezas existen como notebooks experimentales, pero falta convertirlas en pipeline reusable.

---

## 7. Checklist integral

### Hecho

- [x] SPIDER explorado.
- [x] SPIDER preprocesado.
- [x] SPIDER segmentación sagital binaria.
- [x] SPIDER segmentación sagital multiclase.
- [x] SPIDER holdout validado.
- [x] Al-Kafri inventariado.
- [x] Al-Kafri pairing oficial rescatado.
- [x] Al-Kafri labels finales encontrados.
- [x] Al-Kafri T2 curado.
- [x] Al-Kafri axial T2 entrenado.
- [x] Al-Kafri axial T2 evaluado.
- [x] E10 axial T2 final limpio completado.

### Pendiente crítico

- [ ] E11: mapeo de clases axiales y reporte final.
- [ ] E12: entrenamiento sagital final limpio.
- [ ] E13: pipeline común de inferencia multiplanar.
- [ ] E14: agente/orquestador IA.
- [ ] E15: spike de reconstrucción 3D con geometría DICOM.
- [ ] Integración con MVP/backend.
- [ ] UI para overlays y revisión profesional.

---

## 8. Qué se puede mejorar

### Prioritario

1. Mapear clases axiales a anatomía real.
2. Reentrenar axial excluyendo o tratando distinto `raw_0`.
3. Hacer entrenamiento sagital final limpio.
4. Crear pipeline de inferencia común.
5. Crear evaluación final comparativa.

### Opcional

1. Más épocas axial.
2. Augmentations.
3. Dice + Focal Loss.
4. Attention U-Net o U-Net++.
5. Comparar T1 vs T2 axial.

### Avanzado

1. 2.5D axial usando cortes vecinos.
2. Reconstrucción 3D con geometría DICOM.
3. Fusión sagital + axial por coordenadas físicas.
4. Agente con control de calidad.
5. Validación profesional.

---

## 9. Roadmap recomendado

### Notebook 19 — E11 axial class mapping and final report

Objetivo:

- Analizar valores raw.
- Confirmar clases anatómicas.
- Calcular distribución por clase.
- Reportar métricas con/sin `raw_0`.
- Dejar tabla final de clases.

### Notebook 20 — E12 sagittal final training clean

Objetivo:

- Usar pipeline SPIDER multiclase.
- Entrenar/evaluar de forma limpia.
- Guardar checkpoint final sagital.
- Generar reporte comparable al axial.

### Notebook 21 — E13 multiplanar inference pipeline

Objetivo:

- `infer_sagittal()`
- `infer_axial_t2()`
- `detect_plane()`
- `generate_overlay()`
- Salida común: imagen, máscara, overlay, métricas y quality flags.

### Notebook 22 — E14 AI agent orchestrator Colab

Objetivo:

- Simular agente.
- Recibir serie DICOM.
- Decidir pipeline.
- Ejecutar modelo.
- Evaluar calidad.
- Generar reporte.

### Notebook 23 — E15 multiplanar 3D reconstruction spike

Objetivo:

- Leer geometría DICOM.
- Construir volumen.
- Proyectar máscaras.
- Generar malla 3D preliminar.

---

## 10. Texto resumido para tesis

El proyecto cuenta actualmente con dos líneas de segmentación validadas experimentalmente. En el plano sagital, se entrenó y evaluó un modelo multiclase sobre SPIDER, alcanzando un Dice macro sin fondo cercano a 0.83 en holdout. En el plano axial, se reconstruyó el emparejamiento oficial del dataset Al-Kafri/Sudirman, se identificaron los labels finales entrenables y se entrenó un modelo axial T2 multiclase sobre 610 pares curados. El modelo axial final alcanzó un Dice macro sin fondo de 0.6587 en test y un Dice macro de 0.8167 al excluir la clase minoritaria `raw_0`. Estos resultados habilitan el diseño de un sistema multiplanar, aunque la reconstrucción 3D real requiere una etapa adicional de integración geométrica DICOM sobre estudios que contengan series sagitales y axiales del mismo paciente.

---

## 11. Diagnóstico final

**Lo hecho:** modelos 2D sagital y axial viables.  
**Lo que falta:** convertir experimentos en pipeline final, agente y 3D.  
**Lo más crítico:** no confundir modelos sagital + axial con 3D integrado.  
**Lo más valioso:** ya se demostró que el enfoque multiplanar es técnicamente posible.
