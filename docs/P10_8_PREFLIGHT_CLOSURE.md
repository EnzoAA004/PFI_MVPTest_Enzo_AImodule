# P10.8 — Cierre formal del preflight de expansión clínica

**Estado:** `P10_8_PREFLIGHT_CLOSED_NO_TRAINING`

## Decisión

El preflight P10.8 queda formalmente cerrado sin abrir Notebook 77 de entrenamiento.

La evidencia de los Notebooks 69–76 permite avanzar con protocolos de medición y revisión profesional, pero no autoriza un nuevo entrenamiento ni una nueva afirmación de automatización clínica.

## Gates cerrados

- Notebook 69: inventario global, hashes de checkpoints y guardia de reentrenamiento.
- Notebook 70: requerimientos clínicos de Susana Torres.
- Notebook 71: auditoría de fuentes primarias Al-Kafri/Sudirman.
- Notebook 72: auditoría de máscaras y semántica D3/D4/D5.
- Notebook 73: compuerta de viabilidad.
- Notebook 74: protocolo geométrico de altura discal y diámetro AP.
- Notebook 75: protocolo multiframe de hernia, sin reentrenamiento.
- Notebook 76: auditoría T1/T2 y sagital–axial, con corrección de disponibilidad SPIDER.

## Hallazgos de cierre

### Entrenamiento P10.8

- `trainingAuthorized = false`
- `trainingPrerequisitesMet = false`
- no se generaron checkpoints P10.8;
- no corresponde ejecutar un Notebook 77 de entrenamiento con la evidencia actual.

### Candidatos bloqueados con el dataset actual

- `facet_hypertrophy`
- `ligamentum_flavum_hypertrophy`
- `annular_tear`
- `nerve_root_compression`
- `epidural_fat`

Motivo: no existe asociación validada entre las máscaras auditadas y una taxonomía clínica utilizable para esas tareas.

### Tareas ya entrenadas que no deben reabrirse

- P10.6: central, foraminal y subarticular.
- P10.7: Pfirrmann, Modic, cambios de platillos, espondilolistesis, hernia, estrechamiento discal y bulging.

### Protocolos P10.8 habilitados para productización controlada

1. **Altura discal**
   - plano sagital;
   - seis landmarks seleccionados o revisados por profesional;
   - salida descriptiva en milímetros;
   - sin umbrales clínicos congelados.

2. **Diámetro AP**
   - plano axial;
   - dos landmarks seleccionados o revisados por profesional;
   - salida descriptiva en milímetros;
   - sin clasificación de severidad.

3. **Revisión multiframe de hernia**
   - original primero;
   - sagital/parasagital y axial;
   - cortes adyacentes relevantes;
   - overlays bajo demanda;
   - sin reentrenamiento;
   - sin barras de probabilidad para la vista de producto.

### T1/T2 y multiplanar

SPIDER, después de corregir el parseo booleano del manifest:

- total: 1518 filas;
- T1 disponible: 1370;
- T2 disponible: 1467;
- T1+T2: 1319;
- solo T1: 51;
- solo T2: 148;
- ninguna: 0.

Al-Kafri FAST preflight:

- 1363 series representativas auditadas;
- 204 estudios;
- 195 estudios con T1+T2;
- 198 estudios con sagital+axial;
- 0 pares sagital↔axial que superen el gate conservador de geometría/Frame of Reference.

Por lo tanto:

- `automaticSagittalAxialAlignmentValidated = false`
- `fullProductMultiplanarValidation = false`
- `crossCohortPairingAllowed = false`

## Alcance de producto congelado

P10.8 no incorpora un nuevo modelo. Sus entregables para producto son contratos y flujos de apoyo:

- mediciones geométricas descriptivas con revisión profesional;
- protocolo de revisión multiframe;
- separación T1/T2;
- guardas explícitas para no presentar alineación multiplanar automática como validada.

Los hallazgos degenerativos continúan describiéndose como hallazgos de apoyo y no como diagnóstico clínico autónomo.

## Fuente de trazabilidad

Rama de preflight: `enzo/p10-8-clinical-expansion-preflight`

Commit observado al cierre: `0cd72d1fda51fcb755f4cf555791f0534a90404d`

La rama de consolidación no importa de forma masiva el historial/notebooks de P10.8 para evitar duplicados y artefactos experimentales; conserva este cierre como fuente normativa de alcance.

## Siguiente etapa

`PRODUCT_CONSOLIDATION_AND_E2E`

Orden:

1. consolidar AI Module;
2. consolidar Backend;
3. ejecutar regresión P10.6;
4. cerrar runtime P10.7 con checkpoint real, paridad de preprocessing y localización de disco;
5. incorporar contratos P10.8 sin nuevo modelo;
6. ejecutar E2E AI↔Backend;
7. congelar contratos y fixtures;
8. merge a `main` solo con todos los gates verdes;
9. emitir handoff a Frontend.
