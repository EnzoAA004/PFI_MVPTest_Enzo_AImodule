# P10-AI Hardening

Este documento registra el cierre acotado de seguridad, observabilidad y
degradacion explicita del AI Module.

## TraceId

- El header `X-Trace-Id` se normaliza en cada request.
- Si el cliente no envia traceId, el servicio genera uno con prefijo `trace-`.
- Las respuestas exitosas y de error devuelven el mismo traceId en header y body.
- Los logs de errores relevantes conservan `traceId`, `caseId` y tipo de excepcion
  cuando aplica, sin publicar mensajes internos.

## Sanitizacion publica

Las respuestas HTTP no deben exponer:

- `sourcePath`, `inputPath`, `outputFiles` ni rutas absolutas;
- directorios internos de runtime, modelos, outputs o temporales;
- hosts internos;
- tokens, secretos, autorizaciones o passwords;
- stack traces o mensajes crudos de dependencias.

Los endpoints de health/readiness/modelos publican estado, checksums,
readiness, trainingStatus y quality gates, pero no rutas locales.

## Modos de inferencia y fallback

- `requestedInferenceMode` representa lo solicitado por el cliente.
- `effectiveInferenceMode` o `aiOutput.inferenceMode` representa lo ejecutado.
- `real_baseline` solo se informa como efectivo si el modelo real se ejecuto
  correctamente y no hubo fallback.
- Si falla la inferencia real y `allowContractFallback=true`, el resultado se
  degrada explicitamente a `contract`, con flags de fallback y motivo publico
  sanitizado.
- Si falla la inferencia real y `allowContractFallback=false`, la respuesta es
  error HTTP controlado. No se informa un exito real falso.

## Readiness de modelos

El readiness conserva la diferencia entre:

- baseline aprobado (`baselineReady=true`);
- candidato ejecutable para inferencia real (`availableForRealInference=true`,
  `readiness=real_candidate_ready`).

Los manifests invalidos, artifacts ausentes o modelos no disponibles no deben
promoverse a exito real.

## Compatibilidad

Este hardening mantiene el contrato multiplanar vigente. Los assets publicos se
identifican por `assetName`, `relativePath`, `mediaType`, `size` y metadatos
seguros. Los blobs y rutas internas permanecen fuera de las respuestas publicas.
