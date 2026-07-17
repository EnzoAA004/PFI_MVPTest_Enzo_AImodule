# Estado del proyecto y roadmap al 100% (producto final deployado)

Fecha: 2026-07-16
Objetivo declarado: NO un MVP al 50%, sino un producto final funcional, deployado y defendible como tesis profesional de última entrega.

---

## 1. Qué está HECHO (verificado con tests/evidencia)

### AI Module — completo a nivel runtime real
| Ticket | Qué se logró |
|---|---|
| AI-001/002 | Checkpoints inspeccionados; manifests endurecidos con SHA (MATCH), validación estricta |
| AI-005/006 | Carga real strict=True de sagital (SagittalUNet2D, 4 clases) y axial (AxialUNet2D, 6 clases) |
| AI-007/008 | real_baseline por plano con input real (fixtures), 4 outputs + runId/traceId |
| AI-012 | Corrida multiplanar real (ambos planos real_baseline, multiplanarRunId común) |
| AI-013/014 | inputId server-side + upload seguro (allowlist, tamaño, anti-traversal) |
| AI-015/016 | Asset registry + serving seguro (PNG al browser; .npy raw y .pt bloqueados) |
| DEV-001 | Dockerfile AI (torch/pydicom/SimpleITK), imagen 411 MB, /health OK, .pt por volumen |
| QUAL-003/003b | Evaluador oficial Dice/IoU por clase + macro, con label_group_mapping |
| QUAL-005 | Notebook final de entrenamiento (en ejecución en Colab) |

### Backend — contrato + proxies + persistencia + seguridad
| Ticket | Qué se logró |
|---|---|
| BE-000/001 | Suite verde (JDK 21), DTOs de multiplanar run congelados |
| BE-002/003/004 | Proxy de upload, proxy de assets, ejecución multiplanar con inputId |
| BE-005b | Persistencia REAL probada con PostgreSQL/Testcontainers (migraciones versionadas) |
| BE-006 | Persistencia de cada corrida (modelo, artifactHash, métricas, assets, traceId) |
| BE-007 | Workflow de revisión profesional (accepted/observed/rejected/edited) |
| BE-009 | Auditoría de eventos clave (upload/run/review/errores), metadata sanitizada |
| BE-010 | Roles sobre JWT (ADMIN sync/diagnostics; DOCTOR/REVIEWER review), 403 auditado |
| DEV-002a | Dockerfile backend (multi-stage Maven→JRE 21) |

### Frontend — flujo E2E + viewer + revisión + readiness
| Ticket | Qué se logró |
|---|---|
| FE-001→005 | Tipos del contrato, workspace por caso, carga de RM, ejecución, render de overlays |
| FE-006/008 | Viewer sagital/axial lado a lado con opacidad; landmarks + tabla de mediciones |
| FE-009 | Formulario de revisión profesional con refresco de estado |
| FE-011 | Panel de readiness (health/readiness/models/runtime, evidencia real vs contract) |
| DEV-002b | Dockerfile frontend (Vite build + nginx, URL backend por runtime) |

### Resultado
Flujo completo funcionando: seleccionar caso → subir RM → correr análisis real (real_baseline) → ver overlays/landmarks/mediciones → revisar profesionalmente → todo persistido y auditado. Los tres servicios containerizan.

---

## 2. Qué FALTA del backlog (pendiente)

### P0 en curso / próximos
- **QUAL-002/004/005 (calidad ≥70%)**: reentrenamiento en Colab + held-out limpio por paciente + evaluación final. EN CURSO. Es el gate honesto de la tesis.
- **DEV-002 (docker compose)**: orquestar todo con un comando. EN CURSO (esperando modelos reales).
- **DEV-003**: variables de entorno documentadas (parcial vía .env.example).
- **DEV-006**: estrategia final de storage de artifacts (.pt fuera de git, materialización).
- **AI-009**: ordenar serie DICOM axial por metadata (ingesta DICOM real).
- **AI-003**: verificación de hash antes de reemplazar artifacts descargados.
- **TEST-001**: contrato JSON golden multiplanar.
- **TEST-002/003**: E2E local contract y real_baseline.
- **TEST-004**: pruebas de seguridad de uploads/assets.
- **VAL-001**: set de casos demo deidentificados.
- **VAL-004**: guion de demo final.
- **DOC-001/002/003/005**: requerimientos RF/RNF, ERD, capturas de demo, anexo de instalación.

### P1/P2 (calidad y pulido)
- AI-004 (lock sync), AI-010 (contornos), AI-011 (coordinate space), AI-017 (mask-preview por clase), AI-018 (quality schema).
- BE-008 (versionado de mediciones), BE-011 (errores 502/503), BE-012 (OpenAPI).
- FE-007 (capas por clase), FE-010 (edición de mediciones), FE-012 (historial de corridas), FE-013 (pulido UX), FE-014 (3D honesto deshabilitado).
- DEV-004 (healthcheck en deployment), DEV-005 (CI mínimo).
- TEST-005 (degradación), TEST-006 (perf smoke), VAL-002/003 (checklist/feedback), DOC-004 (capítulos tesis).

### Deudas de higiene
- Push del repo AI Module (trabajo acumulado sin pushear).
- Commitear AGENTS.md en cada repo.
- Definición anatómica de clases axiales raw_* (pendiente de rastreo).

---

## 3. Brechas para "100% deployado profesional" (lo que el backlog NO cubre y hay que AGREGAR)

El backlog actual llega a "producto funcional local demostrable". Para un producto **deployado y profesional** faltan epics nuevos:

### Epic DEPLOY — despliegue en la nube (inferencia)
- DEPLOY-001: elegir proveedor y arquitectura (frontend estático + backend + AI + PostgreSQL gestionado). Entrenamiento queda en Colab; la nube solo sirve inferencia.
- DEPLOY-002: registry de imágenes + materialización de artifacts en cloud (con verificación SHA).
- DEPLOY-003: dominio + HTTPS/TLS, y prueba en vivo end-to-end con latencia real.
- DEPLOY-004: gestión de secretos (no .env en repo; secret manager/variables del proveedor).

### Epic SEC — hardening de seguridad
- SEC-001: flujo de auth completo (registro, aprobación de PENDING_APPROVAL, gestión de contraseña/expiración de token).
- SEC-002: TLS extremo a extremo, CORS de producción, rate limiting en endpoints de upload/run.
- SEC-003: escaneo de dependencias (vulnerabilidades) y política de actualización.
- SEC-004: validación de deidentificación garantizada en ingesta (no PII en logs/DB/assets).

### Epic OBS — observabilidad
- OBS-001: logging estructurado con traceId correlacionado (ya hay traceId; falta el logging correlacionado).
- OBS-002: métricas básicas y monitoreo/health para el orquestador (uptime, latencia, errores).
- OBS-003: seguimiento de errores en producción.

### Epic DATA — ciclo de vida de datos
- DATA-001: backups de PostgreSQL y política de retención.
- DATA-002: política de retención/borrado de inputs/outputs y trazabilidad de borrado.

### Epic CICD — integración y despliegue continuo
- CICD-001: pipeline automatizado build+test (los 3 repos) en cada push/PR (extiende DEV-005).
- CICD-002: despliegue automatizado a la nube desde main (CD).

### Epic LEGAL/ETHICS — encuadre profesional (producto médico-adyacente)
- LEG-001: disclaimer visible y política de uso (no diagnóstico, no certificado, revisión profesional obligatoria).
- LEG-002: política de manejo de datos, consentimiento y deidentificación documentada.
- VAL-CLIN-001: protocolo de validación por médicos con registro de acuerdo (ya propuesto en la revisión crítica).

### Cierre de calidad (ya iniciado)
- QUAL-002/004/005: alcanzar Dice macro ≥0.70 en held-out por paciente, reportado con y sin raw_0, con model cards actualizadas.
- PERF-001/002: medición de latencia y presupuesto de UX.

---

## 4. Roadmap sugerido al 100%

1. **Cerrar demo local reproducible** (esta semana): DEV-002 (compose) + DEV-003 + TEST-002/003. Con esto el producto se levanta con un comando y se prueba end-to-end.
2. **Cerrar el gate de calidad**: QUAL-005 (training) → QUAL-002/004 (held-out limpio + evaluación) → model cards finales. Objetivo Dice macro ≥0.70.
3. **Seguridad y trazabilidad**: SEC-001..004, TEST-004, LEG-001/002. Deja el producto defendible en lo ético/seguro.
4. **Deploy en la nube**: DEPLOY-001..004 + OBS-001..003 + DATA-001/002 + CICD-001/002. Producto realmente "deployado".
5. **Pulido y evidencia**: FE-007/010/012/013, BE-008/011/012, VAL-001/002/004, DOC-001..005. Demo, capturas, documentación y anexos.
6. **Evolución honesta**: 3D-001/002 como trabajo futuro documentado (no simulado).

## 5. Definición de "100% terminado" (criterios de aceptación del producto final)
- Se levanta con un comando (compose local) y está deployado y accesible por HTTPS en la nube.
- Modelos reales con Dice macro ≥0.70 en held-out por paciente, documentado honestamente (incluida la limitación de raw_0).
- Flujo completo: auth con roles → carga → análisis real → viewer con mediciones → revisión profesional → persistencia + auditoría, todo trazable.
- Seguridad: sin secretos en repo, TLS, deidentificación garantizada, sin PII en logs/DB.
- CI/CD verde en los 3 repos; backups de DB configurados.
- Documentación completa: OpenAPI, ERD, arquitectura, anexo de instalación, guion de demo, encuadre legal/ético, y evidencia para la tesis.
