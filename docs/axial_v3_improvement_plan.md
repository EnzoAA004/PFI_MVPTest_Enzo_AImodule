# Plan reproducible de mejora axial v3

## 1. Resumen ejecutivo

Axial v3 es un plan de mejora del modelo de segmentacion axial T2 de RM lumbar enfocado en el problema de la etiqueta `raw_0`. El objetivo primario de desarrollo es superar en validation el `dice_macro_foreground` de axial-final-v2, que fue `0.7283182698789201`, sin acceder al split test ni usar resultados test para tomar decisiones de entrenamiento, arquitectura, calibracion o seleccion.

El plan se organiza en cuatro iteraciones:

- A. Diagnostico de `raw_0`.
- B. Mejoras de bajo costo conservando, cuando sea posible, `AxialUNet2D`.
- C. Incorporacion de contexto 2.5D solo si existe orden confiable entre slices.
- D. Evaluacion controlada de arquitecturas alternativas solo si B y C no justifican congelar axial v3.

Axial-final-v2 fallo el quality gate global porque `dice_macro_foreground` incluyendo `raw_0` quedo por debajo del umbral predefinido `0.70`. La caida esta dominada por `raw_0`: excluyendo esa clase, el Dice test fue aproximadamente `0.811`. Por lo tanto, v2 sigue siendo un resultado experimental valido, pero no debe presentarse como modelo clinicamente validado ni renombrarse como `best`.

## 2. Contexto de axial-final-v2

La evaluacion final v2 fue ejecutada una unica vez sobre 102 slices de test. El modelo quedo empaquetado como `axial_t2_alkafri_final_v2_candidate.pt`. No debe renombrarse como `best`.

Debe conservarse el warning metodologico:

> The held-out test partition was previously evaluated for the axial-full-v1 baseline. This v2 evaluation is comparative and should not be interpreted as a fully untouched external validation.

Una futura evaluacion de axial v3 sobre el mismo test tambien sera comparativa, no una validacion externa completamente intacta.

## 3. Metricas v2

| Split | Metrica | Valor |
|---|---:|---:|
| validation | dice_macro_foreground | 0.7283182698789201 |
| test | dice_macro_foreground incluyendo raw_0 | 0.679348283374592 |
| test | umbral predefinido | 0.70 |
| test | qualityGatePassed | false |
| test | dice_macro_excluding_raw0 | 0.8105661875685084 |
| test | iou_macro_foreground | 0.5699727598420694 |
| test | iou_macro_excluding_raw0 | 0.6915400877715925 |

| Clase | Dice test | IoU test | Precision | Recall | TP pixels | FP pixels | FN pixels | Presencia real | Presencia predicha | Prediccion falsa en ausencia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_0 | 0.15447666659892562 | 0.08370344812397683 | 0.08640693559265518 | 0.7279111338100103 | 11402 | 120555 | 4262 | 21/102 | 102/102 | 81/81 |
| raw_50 | 0.9316161514449703 | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d |
| raw_100 | 0.8399799336337151 | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d |
| raw_150 | 0.7954988747186796 | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d |
| raw_200 | 0.6751697904766687 | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d | n/d |

## 4. Problema de raw_0

El problema principal esta concentrado en la muy baja precision de `raw_0` y en su prediccion sistematica en slices donde no esta presente. En test, `raw_0` estuvo presente realmente en 21 de 102 slices, pero fue predicho en 102 de 102. En los 81 slices sin `raw_0`, el modelo predijo falsamente la clase en todos los casos.

No se asigna semantica anatomica a `raw_0`. Las etiquetas se mantienen como:

| Valor original | Nombre permitido |
|---:|---|
| 250 | background_250 |
| 0 | raw_0 |
| 50 | raw_50 |
| 100 | raw_100 |
| 150 | raw_150 |
| 200 | raw_200 |

## 5. Hipotesis de trabajo

- `raw_0` podria estar sobrerrepresentado por la funcion de perdida o por pesos efectivos de clase.
- `raw_0` podria tener anotaciones pequenas, dispersas, cercanas a bordes o inconsistentes.
- La ambiguedad semantica de `raw_0` podria inducir falsos positivos aun cuando el resto de clases se comporta mejor.
- La informacion de slices vecinos podria ayudar si existe continuidad anatomica y orden confiable.
- Un postprocesamiento calibrado con validation podria reducir falsos positivos sin degradar materialmente otras clases.

## 6. Restricciones metodologicas

- No acceder al split test durante el desarrollo de axial v3.
- No usar metricas, probabilidades, imagenes, predicciones ni resultados test para pesos, arquitectura, umbrales, checkpoints, postprocesamiento, early stopping ni quality gate.
- Todos los experimentos A, B, C y D usan solo train y validation.
- Mantener el split por paciente existente.
- No modificar ni borrar `test_evaluated_once.json`, `test_metrics.json`, `test_case_metrics.csv`, `test_metrics_per_class.csv`, `test_confusion_matrix.csv`, `test_predictions.png`, `axial_t2_alkafri_final_v2_candidate.pt`, su manifest, su model card, `final_artifact_verification.json` ni el checkpoint axial-final-v2 existente.
- No ejecutar entrenamientos largos ni evaluaciones finales desde Codex.
- Todo notebook nuevo debe quedar limpio: `execution_count = null`, `outputs = []`, IDs unicos, sin tokens hardcodeados y sin rutas de usuario innecesarias.

## 7. Iteraciones A, B, C y D

### Iteracion A: diagnostico de raw_0

Entrada: split train/validation, manifest curado, mascaras, imagenes, checkpoint v2 solo para inferencia sobre validation.

Salida: CSV por slice, CSV por paciente, JSON resumen, histogramas, boxplots, presencia por paciente, mosaicos de ejemplos positivos, negativos, cercanos al borde y extremos por area. Tambien distribuciones validation de probabilidad maxima `raw_0`, area predicha, falsePositivePixels por slice, relacion probabilidad/presencia y curva precision-recall de presencia.

La iteracion A no reentrena. Incluye auditoria explicita de contacto frecuente con bordes y una lista de casos representativos para revision humana pendiente.

### Iteracion B: mejoras de bajo costo

Entrada: hallazgos de A, train/validation, arquitectura base `AxialUNet2D`, configuracion serializable.

Salida: registro reproducible por experimento, metricas validation globales y por clase, checkpoints seleccionados por validation y reporte de configuracion completa.

Experimentos minimos: B0 reproduccion v2, B1 grilla de peso efectivo de `raw_0`, B2 limite de ratio de pesos de clase, B3 perdida asimetrica con convencion clara alpha=FP y beta=FN, B4 calibracion validation-only de umbral/margen `raw_0`, B5 cabeza opcional de presencia, B6 muestreo balanceado train-only.

Politica de seleccion basada solo en validation:

1. Mayor `dice_macro_foreground`.
2. Ante empate, menor `raw0PredictedInGtAbsentCases`.
3. Ante empate, mayor `raw0Precision`.
4. Ante empate, mayor `dice_macro_excluding_raw0`.

Guardrail configurable: no seleccionar una configuracion que mejore `raw_0` degradando fuertemente `raw_50`, `raw_100`, `raw_150` o `raw_200`. El valor inicial documentado es `maxOtherClassDiceDrop = 0.05`.

### Iteracion C: contexto 2.5D

Entrada: mejor configuracion B y manifest con orden confiable por paciente/estudio.

Salida: dataset 2.5D con canales anterior, central y posterior; salida del slice central; pruebas de extremos, orden y proteccion cross-patient; modelo compatible con 3 canales; comparacion validation contra B.

Antes de implementar se debe verificar si el orden proviene de `InstanceNumber`, `ImagePositionPatient`, `SliceLocation`, nombre de archivo, indice curado o metadata del manifest. No se asume orden lexicografico. Si no hay orden confiable, C queda `blocked`.

### Iteracion D: arquitecturas alternativas

Entrada: resultados validation de B y C, presupuesto de computo y justificacion tecnica.

Salida: comparacion controlada de `AxialUNet2D`, Attention U-Net liviana, U-Net++ reducida, encoder preentrenado compatible y modelo 3D liviano solo como propuesta futura o prototipo controlado.

Cada arquitectura debe reportar parametros, memoria aproximada, tiempo por epoca, inferencia, tamano de artifact, compatibilidad T4, complejidad de despliegue, beneficio validation, impacto en `raw_0` y en `raw_200`.

## 8. Entradas y salidas por iteracion

| Iteracion | Entradas | Salidas |
|---|---|---|
| A | train, validation, manifest, mascaras, imagenes, checkpoint v2 para validation | auditorias CSV/JSON, figuras, mosaicos, revision humana pendiente |
| B | train, validation, resultados A, config B | metricas validation, registry, configs, checkpoints livianos externos |
| C | train, validation, mejor B, orden confiable | dataset 2.5D, pruebas, metricas validation comparativas |
| D | resultados B/C, presupuesto | matriz de arquitectura, prototipos controlados, recomendacion |

## 9. Criterios de avance y descarte

De A a B: auditoria completa, distribucion de presencia conocida, anomalias documentadas y metricas de probabilidades en validation disponibles.

De B a C: B no produce mejora suficiente o estable, `raw_0` sigue con falsas detecciones sistematicas y existe orden confiable de slices.

De C a D: 2.5D implementado correctamente, sin mejora suficiente, presupuesto disponible y complejidad adicional justificada.

Estos son criterios de avance, no resultados.

## 10. Riesgos tecnicos

- Falta de orden confiable entre slices para C.
- Checkpoints grandes fuera del control de version.
- Incompatibilidad entre modelos de 1 canal y 3 canales.
- Aumento de complejidad que dificulte despliegue en el AI service.
- Mejoras locales de `raw_0` con degradacion de `raw_200` u otras clases.

## 11. Riesgos metodologicos

- Filtracion accidental de test observado hacia decisiones de v3.
- Sobreajuste a validation por grillas demasiado amplias.
- Reinterpretacion anatomica no validada de etiquetas raw.
- Presentar una futura evaluacion comparativa como validacion externa intacta.
- Modificar markers o artifacts historicos de v2.

## 12. Trazabilidad de experimentos

El registro inicial vive en `outputs/axial_v3/experiment_registry.csv`. Puede moverse a una ubicacion externa configurable si los artifacts crecen.

Campos minimos: `experimentId`, `iteration`, `runId`, `createdAtUtc`, `gitCommit`, `seed`, `configPath`, `splitSha256`, `trainingStatus`, `selectedEpoch`, `monitorMetric`, metricas validation, `artifactPath`, `artifactSha256` y `notes`.

Cada experimento debe tener configuracion JSON o YAML serializable. Todo artifact relevante debe incluir SHA-256 cuando corresponda. El repositorio no debe versionar checkpoints grandes.

## 13. Politica de acceso al test

Los notebooks 51 a 54 deben usar `require_train_val_only()` o loaders equivalentes que rechacen cualquier split fuera de train/validation. El validador estatico debe fallar ante invocaciones de evaluacion test, loaders test, lectura de `test_metrics.json`, uso de `AXIAL_FINAL_TEST_CONFIRMATION` o referencias que modifiquen markers v2.

Axial v3 no puede seleccionarse usando el test ya observado.

## 14. Estrategia futura de evaluacion

Cuando axial v3 quede congelado por escrito, puede definirse un protocolo de evaluacion comparativa sobre el mismo test. Ese protocolo debe declarar que el test ya fue observado en v1/v2 y no constituye una validacion externa completamente intacta. No se crea en este alcance un notebook de test v3 que se ejecute automaticamente.

## 15. Matriz de decision A-B-C-D

| Iteracion | Hipotesis | Costo | Riesgo | Datos necesarios | Criterio de exito | Criterio de descarte | Dependencia | Estado |
|---|---|---|---|---|---|---|---|---|
| A | El error de raw_0 es diagnosticable en train/validation | Bajo | Bajo | train, validation, checkpoint v2 para validation | Auditoria completa y casos para revision | Auditoria no reproduce senales utiles | Ninguna | ready |
| B | Ajustes de perdida, pesos, umbral o presencia reducen FP raw_0 | Medio | Medio | train, validation, A | Mejora validation sin degradacion material | Mejora inestable o degrada otras clases | A | planned |
| C | Contexto vecino reduce falsas detecciones | Medio | Medio-alto | orden confiable de slices | Mejora validation contra B | Sin orden confiable o sin mejora | B | planned |
| D | Otra arquitectura aporta beneficio justificado | Alto | Alto | resultados B/C, computo | Beneficio validation que justifique despliegue | Complejidad no justificada | C | planned |

Estados permitidos: `planned`, `ready`, `running`, `completed`, `blocked`, `discarded`.

## 16. Orden recomendado de ejecucion

1. Ejecutar Iteracion A.
2. Revisar manualmente `raw_0`.
3. Congelar hipotesis de B.
4. Ejecutar grilla B unicamente con train/validation.
5. Seleccionar una configuracion usando validation.
6. Evaluar viabilidad de C.
7. Ejecutar C si existe orden confiable.
8. Ejecutar D solo si esta justificado.
9. Congelar axial v3.
10. Definir por escrito el protocolo de una futura evaluacion comparativa.

## 17. Recomendacion final para el alcance del PFI

Para el PFI conviene priorizar A y B como alcance principal: diagnostico reproducible de `raw_0`, guardias tecnicas de no acceso a test, y una grilla acotada de mejoras de bajo costo. C debe quedar condicionado a orden confiable de slices. D debe mantenerse como evaluacion secundaria o futura, salvo que B/C no mejoren validation y exista presupuesto real de computo.

## 18. Estado de implementacion

| Bloque | Estado | Nota |
|---|---|---|
| Iteracion A | ready_to_run | Ejecutable con `run_iteration_a()`, manifest train/validation, mapping explicito y auditoria probabilistica validation-only si se configura checkpoint v2. |
| Iteracion B | framework_ready | Runner modular para preflight/smoke/train, B0-B6 expandibles, B0 con CE ponderada + Soft Dice foreground como v2. |
| Iteracion C | planned_or_blocked_pending_order_audit | Bloqueada si no existe `order_source` confiable y validado. |
| Iteracion D | planned | Metadata de arquitecturas; alternativas no implementadas siguen deshabilitadas. |

No se marca A como `completed` porque no se ejecuto sobre el dataset real en este commit. No se declara ninguna mejora de axial v3.

## 19. Correcciones posteriores al commit dba7dad

- El validador axial v3 ya no depende de `outputs/axial_v3/experiment_registry.csv`; valida el registry con un CSV temporal.
- Se agrego mapping explicito en `labels.py`: raw `250, 0, 50, 100, 150, 200` a indices `0..5`.
- Notebook 51 ahora orquesta un runner real de Iteracion A, sin ejecutar por defecto.
- Notebook 52 ahora orquesta un runner real B0-B6, sin ejecutar por defecto.
- B0 usa perdida baseline compatible con v2: CrossEntropy ponderada + Soft Dice multicategoria foreground.
- La calibracion de `raw_0` soporta `[C,H,W]` y `[N,C,H,W]`.
- El ranking validation-only preserva ceros como valores validos.
- El guardrail de otras clases falla cerrado ante metricas ausentes.
- La cabeza de presencia se integra mediante wrapper y usa bottleneck features.
- B6 cuenta con sampler balanceado train-only.
- C exige `order_source` confiable.

## 20. Protocolo de Iteracion A

Configurar variables de entorno: `PFI_REPO_ROOT`, `PFI_DATASET_ROOT`, `AXIAL_E9_CURATED_SPLIT_CSV`, `PFI_OUTPUT_ROOT`, `PFI_MASK_LABEL_MODE=raw` y opcionalmente `AXIAL_V2_CHECKPOINT_PATH` para auditoria probabilistica validation-only.

Ejecutar notebook 51 con `PFI_RUN_AXIAL_V3_AUDIT=1`. El runner valida storage, filtra train/validation, mapea mascaras crudas explicitamente, genera tablas estructurales, figuras basicas y reportes. Si se define checkpoint v2, carga `axial_t2_alkafri_v2.best_checkpoint.pt` solo para validation.

## 21. Protocolo de Iteracion B

Configurar `PFI_AXIAL_V3_EXPERIMENT_ID` para un experimento atomico. Usar `RUN_MODE=preflight`, `smoke`, `train` o `summarize`. Para ejecutar, definir ademas `PFI_RUN_AXIAL_V3_EXPERIMENT=1`.

B0 reproduce la configuracion train/validation v2 desde cero. B1-B6 se expanden de forma independiente; no se crea un producto cartesiano grande.

## 22. Politica de congelamiento de axial v3

Un modelo axial v3 solo puede congelarse despues de ejecutar train/validation real, registrar configuracion, seed, commit, split SHA-256, metricas validation y guardrail. Hasta entonces el mejor resultado solo puede llamarse `validation_candidate`.

## 23. Definicion de validation_candidate

`validation_candidate` es el run completo no-smoke que pasa guardrails y queda primero por politica validation-only:

1. mayor `dice_macro_foreground`;
2. menor `raw0PredictedInGtAbsentCases`;
3. mayor `raw0Precision`;
4. mayor `dice_macro_excluding_raw0`.

No implica aprobacion clinica ni exito test.

## 24. Que no constituye mejora valida

- Mejorar `raw_0` degradando materialmente `raw_50`, `raw_100`, `raw_150` o `raw_200`.
- Usar test historico para elegir thresholds, pesos, checkpoints, arquitectura o postprocesamiento.
- Usar un smoke run como candidato.
- Cambiar el quality gate para declarar exito.
- Inferir anatomia de etiquetas raw.
