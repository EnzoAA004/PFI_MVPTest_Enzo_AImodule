# Arquitectura multirepo

La arquitectura final del producto se separa en tres repositorios para mantener responsabilidades claras, facilitar despliegues independientes y evitar mezclar investigacion de IA con logica de producto.

```text
Frontend React -> Spring Boot Backend -> Python FastAPI AI Module
```

## Por que hay 3 repos

- El frontend cambia al ritmo de experiencia de usuario, visualizacion y flujos de revision.
- El backend concentra reglas de producto, persistencia, seguridad, auditoria e integraciones.
- El AI Module evoluciona con modelos, preprocesamiento, inferencia, mediciones y evidencia tecnica.

Esta separacion reduce acoplamiento y permite desplegar el modulo IA con dependencias cientificas sin contaminar el backend Java ni el frontend.

## Responsabilidad del AI Module

Este repositorio expone un servicio Python/FastAPI para tareas tecnicas de IA:

- preprocesamiento de imagenes;
- inferencia sagital y axial cuando este implementada;
- mediciones geometricas derivadas de mascaras;
- overlays y artefactos tecnicos;
- configuracion y trazabilidad de modelos;
- reportes del agente IA;
- notebooks y evidencia tecnica reproducible.

El AI Module no gestiona usuarios, permisos de producto, historias clinicas ni decisiones medicas.

## Responsabilidad del Backend

El backend Spring Boot debe:

- recibir solicitudes del frontend;
- gestionar autenticacion, autorizacion y persistencia;
- orquestar archivos y estados de procesamiento;
- llamar al AI Module por HTTP o por un canal interno controlado;
- guardar resultados estructurados y editables;
- exponer al frontend resultados listos para revision.

## Responsabilidad del Frontend

El frontend React debe:

- permitir carga y seguimiento de estudios segun el flujo de producto;
- mostrar resultados, overlays y mediciones;
- permitir revision, edicion y validacion profesional;
- comunicar claramente que los resultados son asistivos.

## Flujo de datos

1. El usuario interactua con el frontend.
2. El frontend envia la solicitud al backend Spring Boot.
3. El backend valida permisos, registra la solicitud y prepara referencias a archivos.
4. El backend invoca el AI Module.
5. El AI Module procesa, consulta modelos/resultados y devuelve JSON tecnico.
6. El backend persiste el resultado editable.
7. El frontend presenta la salida para revision profesional.

## Regla human-in-the-loop

El sistema es asistivo y requiere revision profesional. Las respuestas del AI Module deben incluir o preservar `human_review_required=true` cuando correspondan. Ningun componente debe presentar la salida como diagnostico clinico, recomendacion terapeutica o decision medica automatizada.
