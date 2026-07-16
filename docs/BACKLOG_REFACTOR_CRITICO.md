# Refactor crítico del backlog — de "demostrar que corre" a "sistema funcional defendible"

Autor: Planner (revisión crítica solicitada por Enzo)
Fecha: 2026-07-16
Alcance: complementa el `backlog_maestro_producto_final_100.md`. No reemplaza los tickets existentes; los reencuadra y agrega las piezas faltantes.

---

## 1. Diagnóstico crítico del backlog actual

Lo que el backlog hace bien:
- Define un pipeline end-to-end honesto (ingesta → inferencia → overlays → revisión → trazabilidad).
- La regla de `real_baseline` es rigurosa: no permite declarar inferencia real sin artifact válido, SHA, `strict=True`, outputs y trazabilidad.
- Separa contract-mode de real-mode, evitando el fallback silencioso.

El problema de fondo (lo que motivó esta revisión):
El backlog mide **que el sistema corra con pesos reales**, no **que el sistema sea bueno**. `effectiveInferenceMode=real_baseline` es una afirmación **técnica** (el modelo real ejecutó), no de **calidad** (el modelo acierta). Con el criterio actual, un modelo de 2 MB entrenado sobre un subconjunto mínimo puede marcar todos los tickets de IA como DONE aunque su exactitud sea inservible. Para una tesis eso es insuficiente: el tribunal pide un piso de desempeño (≥70%), y "funciona" debe significar "funciona con una calidad medida", no "no tira excepción".

Huecos concretos detectados:
1. **No hay quality gate.** No existe ningún ticket que defina la métrica, el umbral de aceptación (≥70%) ni el conjunto sobre el que se mide. Es el vacío más importante.
2. **No hay conjunto de test held-out.** Se mezcla "validación" (usada para early-stopping) con "evaluación final". Un 70% sobre el set que se usó para elegir el checkpoint no es evidencia de generalización.
3. **No hay plan de escalado de entrenamiento.** El backlog asume los modelos "ya entrenados". No hay ruta explícita de "modelo base chico → modelo escalado que alcanza el umbral".
4. **Trazabilidad de modelo débil.** Los `.pt` no documentan de qué notebook, dataset, split ni métricas provienen. Señal de alarma concreta: el checkpoint axial es un `state_dict` crudo sin config (`target_size` perdido), y en Drive hay dos candidatos axiales (`E9`, `E10`) sin un canónico claro.
5. **No hay medición de latencia.** Crítico para UX del frontend y para dimensionar la nube.
6. **No hay deploy de inferencia en la nube.** Solo Dockerfile/compose local. Falta la ruta a "en vivo".
7. **Encuadre de validación clínica difuso.** Se menciona "revisión profesional" pero no un protocolo de validación por médicos con registro de acuerdo.

---

## 2. Reencuadre del "100%": agregar un quality gate

Propuesta de definición corregida de "AI Module terminado":

> El AI Module está terminado cuando (a) ejecuta `real_baseline` con trazabilidad **y** (b) alcanza el umbral de desempeño acordado (≥70% en la métrica definida) sobre un **conjunto de test held-out** deidentificado, con reporte reproducible por clase.

Es decir: `real_baseline` (técnico) es **condición necesaria pero no suficiente**. El cierre real de la IA pasa a exigir el quality gate. Esto se inserta como un hito nuevo **M1.5 — Calidad y evaluación**, entre M1 (runtime real) y M2 (producto).

---

## 3. Nuevos epics y tickets

### Epic QUAL — Calidad y evaluación del modelo (NUEVO, P0)

**QUAL-001 — Definir métrica y umbral de aceptación**
- Prioridad: P0 | Dependencias: ninguna
- Descripción: Operacionalizar el "≥70%". Decidir métrica primaria (p. ej. Dice macro sin fondo para segmentación; y/o exactitud por clase) y umbral por estructura. Documentar qué significa "70%" y de dónde sale el requisito.
- Criterios de aceptación: métrica primaria y secundarias definidas; umbral por clase/global fijado; escrito en `docs/`.
- Tests/evidencia: documento revisado.

**QUAL-002 — Congelar conjunto de test held-out deidentificado**
- Prioridad: P0 | Dependencias: QUAL-001
- Descripción: Separar un test set que NO se usó para entrenar ni para early-stopping, congelarlo y versionar su manifiesto (IDs, no imágenes pesadas en git).
- Criterios de aceptación: split reproducible train/val/test sin fuga; lista de casos de test fija; deidentificación verificada.
- Tests/evidencia: script de split + hash del listado.

**QUAL-003 — Script de evaluación reproducible por clase**
- Prioridad: P0 | Dependencias: QUAL-002, AI-005/AI-006
- Descripción: Evaluar un checkpoint sobre el test set y producir métrica global + por clase + matriz de confusión / Dice por estructura.
- Criterios de aceptación: corrida reproducible; reporte con número por clase; sin fuga de test en el proceso.
- Tests/evidencia: `pytest` del evaluador con fixtures sintéticas + corrida real documentada.

**QUAL-004 — Medir baseline actual (modelo chico) contra el test set**
- Prioridad: P0 | Dependencias: QUAL-003
- Descripción: Correr los checkpoints actuales (E12 sagital, E10 axial) sobre el test set y reportar el número REAL de hoy. Es el punto de partida honesto para saber cuán lejos del 70% se está.
- Criterios de aceptación: número real por plano documentado; brecha contra umbral explícita.
- Tests/evidencia: reporte en `docs/` (sin maquillar).

**QUAL-005 — Plan y ejecución de escalado de entrenamiento**
- Prioridad: P0 | Dependencias: QUAL-004
- Descripción: Ruta de mejora hasta el umbral: dataset completo, aumento de capacidad (`base_channels`), data augmentation, regularización, más épocas. El entrenamiento sigue en Colab; el repo solo consume el `.pt` resultante.
- Criterios de aceptación: checkpoint que alcanza ≥ umbral en test held-out; manifest actualizado con SHA/métricas nuevas.
- Tests/evidencia: reporte de QUAL-003 sobre el nuevo checkpoint superando el umbral.

**QUAL-006 — Model card / lineage por artifact**
- Prioridad: P1 | Dependencias: QUAL-001
- Descripción: Ficha por `.pt`: notebook de origen, dataset, split, hiperparámetros (`num_classes`, `base_channels`, `target_size`), métricas en test, fecha, versión, SHA-256. Resuelve además la ambigüedad E9/E10 y el `target_size` axial perdido.
- Criterios de aceptación: model card por modelo en `models/final/`; campos `dataset`/`version` del manifest poblados con datos reales.
- Tests/evidencia: revisión + validación de manifest (AI-002).

### Epic PERF — Rendimiento y latencia (NUEVO, P1)

**PERF-001 — Medir latencia de inferencia local**
- Prioridad: P1 | Dependencias: AI-007/AI-008
- Descripción: Medir tiempo de respuesta por plano y multiplanar (carga + preproceso + forward + postproceso) en el entorno local.
- Criterios de aceptación: tiempos p50/p95 documentados por plano y multiplanar.
- Tests/evidencia: script de medición + tabla en `docs/`.

**PERF-002 — Presupuesto de latencia para UX del frontend**
- Prioridad: P2 | Dependencias: PERF-001
- Descripción: Definir qué latencia es aceptable y cómo la absorbe el frontend (async, spinner, estado de progreso).
- Criterios de aceptación: presupuesto acordado; contrato de estados de corrida (pending/running/done/error) para el front.

### Epic DEPLOY — Servir inferencia en la nube (NUEVO, P1)

**DEPLOY-001 — Estrategia de hosting de inferencia (serve, no train)**
- Prioridad: P1 | Dependencias: DEV-001, QUAL-005
- Descripción: Definir dónde y cómo se hostea el AI Module + backend para uso en vivo. **Guardarraíl explícito: para el MVP NO se monta entrenamiento en la nube.** El entrenamiento queda en Colab; la nube solo sirve inferencia con el `.pt` final.
- Criterios de aceptación: arquitectura de deploy documentada; separación clara train (Colab) / serve (cloud); requisitos de recursos (CPU/GPU, RAM) estimados desde PERF-001.

**DEPLOY-002 — Materialización de artifacts en cloud**
- Prioridad: P1 | Dependencias: DEPLOY-001, DEV-006
- Descripción: Los `.pt` (fuera de git) se descargan/materializan por config en el entorno cloud, con verificación SHA (AI-003).
- Criterios de aceptación: arranque en cloud materializa y valida artifacts; sin `.pt` en el repo.

**DEPLOY-003 — Prueba en vivo end-to-end**
- Prioridad: P1 | Dependencias: DEPLOY-002, FE-004
- Descripción: Subir una RM real deidentificada, correr el pipeline en cloud, medir tiempo real y visualizar el resultado en el frontend.
- Criterios de aceptación: corrida real en cloud con outputs visibles en el front; latencia real registrada; evidencia capturable para demo.

### Epic VAL-CLIN — Encuadre de validación clínica (refuerzo de VAL/DOC existentes)

**VAL-CLIN-001 — Protocolo de validación por médicos**
- Prioridad: P1 | Dependencias: QUAL-005, FE-009
- Descripción: Definir qué se muestra a los médicos, qué evalúan (utilidad del overlay/mediciones), y cómo se registra el acuerdo/desacuerdo. Encuadre honesto de limitaciones (no diagnóstico, revisión obligatoria).
- Criterios de aceptación: protocolo escrito; instrumento de registro; muestra mínima de casos; limitaciones documentadas.

---

## 4. Repriorización sugerida

- Insertar **M1.5 — Calidad y evaluación** (QUAL-001..005) como P0, **antes** de considerar cerrado el AI Module. Hoy el camino crítico salta de "carga real" (AI-005/006) a "producto" (M2) sin pasar por calidad; eso es el error a corregir.
- `real_baseline` (AI-007/008) deja de ser la línea de meta de la IA y pasa a ser un prerequisito de QUAL-003.
- QUAL-006 (model cards) es barato y debería hacerse ya, porque además destraba el `target_size` axial y ordena la trazabilidad para la defensa.

---

## 5. Riesgos críticos a vigilar

- **Sobreajuste por dataset chico:** un 70% sobre pocos casos no generaliza. El test held-out (QUAL-002) es lo que da validez al número.
- **Fuga de datos train↔test:** si el mismo paciente/estudio aparece en ambos, el 70% es ficticio. Split por paciente, no por slice.
- **Trazabilidad de modelos:** dos checkpoints axiales (E9/E10) y un `target_size` perdido son señales de que sin model cards la reproducibilidad se pierde.
- **Scope creep de "entrenar en la nube":** mantener entrenamiento en Colab; la nube solo sirve. Cambiar esto explota el alcance de una tesis.
- **Gestión de datos pesados:** SPIDER `images.zip` = 3.7 GB. Storage y materialización deben resolverse (DEV-006/DEPLOY-002), nunca en git.
