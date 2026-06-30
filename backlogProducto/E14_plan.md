# E14 - Agente/orquestador IA

Estado: En progreso.

## Objetivo

Construir un prototipo de agente/orquestador IA sobre el pipeline E13. E14 no entrena modelos nuevos: toma las salidas de inferencia sagital y axial, aplica reglas de seleccion/validacion, prioriza revision profesional y genera reportes automaticos.

## Dependencias

- E10: modelo axial T2 final.
- E11: documentacion de clases axiales y decision sobre raw_0.
- E12: modelo sagital final consolidado.
- E13: pipeline comun de inferencia multiplanar con eje sagital dinamico.

## Entradas principales

- results/E13_multiplanar_inference_pipeline/E13_axial_examples_quality.csv
- results/E13_multiplanar_inference_pipeline/E13_axial_examples_metrics_by_class.csv
- results/E13_multiplanar_inference_pipeline/E13_sagittal_examples_quality.csv
- results/E13_multiplanar_inference_pipeline/E13_sagittal_examples_metrics_by_class.csv
- results/E13_multiplanar_inference_pipeline/E13_multiplanar_pipeline_report.json

## Funciones del agente

- Consolidar una worklist multiplanar axial/sagital.
- Seleccionar el modelo esperado por plano.
- Evaluar quality flags y confianza.
- Calcular prioridad de revision.
- Generar decision de estado: listo para revision, revisar con atencion, repetir/preprocesar.
- Generar reportes por caso.
- Generar reporte global JSON/Markdown.

## Salidas esperadas

- results/E14_ai_agent_orchestrator/E14_agent_worklist.csv
- results/E14_ai_agent_orchestrator/E14_agent_decisions.csv
- results/E14_ai_agent_orchestrator/E14_agent_metrics_summary.csv
- results/E14_ai_agent_orchestrator/E14_agent_report.json
- docs/E14_ai_agent_orchestrator_conclusion.md
- docs/E14_agent_case_reports/E14_case_*.md
- figures/E14_agent_priority_summary.png
- figures/E14_agent_confidence_summary.png

## Criterios de aceptacion

- El agente carga las salidas de E13.
- Genera una worklist mixta sagital/axial.
- Asigna modelo, prioridad y accion recomendada por caso.
- Diferencia casos estandar de casos con flags.
- Genera reporte global y reportes individuales.

## Decision metodologica

E14 representa la capa de orquestacion y control de calidad del sistema. No reemplaza la revision profesional: organiza los resultados de IA, explicita flags y prepara las salidas para validacion humana.

## Proximo paso posterior

E15: spike de reconstruccion 3D / integracion geometrica DICOM, o integracion backend del agente segun prioridad del proyecto.
