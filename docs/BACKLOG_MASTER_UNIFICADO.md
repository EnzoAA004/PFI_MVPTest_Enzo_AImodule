
# Backlog maestro 100% - MVP final de tesis (UNIFICADO)

> Documento unificado (2026-07-16): fusiona el backlog maestro de 100 requerimientos con la revision critica (hito M1.5 de Calidad y epics QUAL/PERF/DEPLOY/VAL-CLIN). Los tickets originales se conservan intactos; lo agregado esta marcado como *(revision critica)*. Ver tambien `BACKLOG_REFACTOR_CRITICO.md`.

Proyecto: Plataforma de apoyo al análisis estructural de RM lumbar con segmentación IA, revisión profesional y trazabilidad.

Alcance correcto del "100%": producto funcional y defendible para tesis/PFI. No implica software médico certificado, diagnóstico autónomo, integración PACS/RIS completa ni validación clínica regulatoria.

Estado de partida asumido:
- Tres repositorios: Frontend React/Vite/TypeScript, Backend Spring Boot y AI Module FastAPI/PyTorch.
- Modelos finales identificados: sagittal_spider_multiclass_final_best.pt y axial_t2_alkafri_final_best.pt.
- Gestión de artifacts, manifests, runtime PyTorch, orquestación multiplanar, backend proxy y frontend base ya implementados o en integración.
- Pendientes críticos: validación con checkpoints reales, ingesta real de RM, serving seguro de assets, persistencia completa de runs/reviews, viewer multiplanar final, pruebas end-to-end y cierre de demo.

Principio operativo:
1. El planificador divide, prioriza y define criterios de aceptación.
2. El agente de código implementa un ticket chico por vez.
3. Cada ticket debe terminar con tests o evidencia concreta.
4. No se permite marcar como real_baseline una salida que no haya ejecutado PyTorch con artifact verificado.
5. No se permite presentar 3D paciente-específico sin stack completo, spacing, orientación y máscaras volumétricas.
6. No se permite exponer paths internos, datos identificables o archivos arbitrarios por endpoints públicos.
7. *(revisión crítica)* No se declara terminado el AI Module solo por correr real_baseline: se exige además superar el umbral de calidad (>=70%) sobre un conjunto de test held-out. real_baseline es condición necesaria, no suficiente.

Convención de prioridades:
- P0: bloquea el producto final o la demo end-to-end.
- P1: necesario para entregar un producto sólido.
- P2: mejora importante, pero no bloqueante.
- P3: evolución posterior o refinamiento.

Estados sugeridos:
Backlog -> Ready for Planning -> Ready for Codex -> In Progress -> Code Review -> Testing -> Done -> Documented.

Definition of Done global:
- Código implementado en el repo correcto.
- Tests relevantes ejecutados o comando de prueba indicado.
- Sin mocks engañosos ni fallback silencioso.
- Logs/errores sin filtrar tokens ni paths sensibles.
- Contrato actualizado en frontend/backend/AI si aplica.
- Evidencia capturable para demo si corresponde.


## Roadmap ejecutivo

| Hito | Nombre | Objetivo |
|---|---|---|
| M0 | Congelamiento y diagnóstico de repos | Saber qué compila y cuál es el contrato real antes de seguir. |
| M1 | Modelos reales y hardening de artifacts | Que los checkpoints finales carguen y queden listos para real_baseline. |
| M1.5 | Calidad y evaluación del modelo *(revisión crítica)* | Que el modelo alcance el umbral (>=70%) en test held-out, no solo que ejecute. |
| M2 | Ingesta, assets y corrida multiplanar | Que el producto pueda recibir inputs, ejecutar y mostrar outputs. |
| M3 | Persistencia, review y auditoría | Que el resultado no sea efímero y tenga revisión profesional trazable. |
| M4 | DevOps y E2E local | Que todo pueda levantarse y probarse como producto completo. |
| M5 | UX, demo y validación | Que el flujo sea defendible, usable y con evidencia. |
| M6 | Cierre final y evolución 3D | Cerrar documentación final y dejar 3D como evolución honesta. |

## Camino crítico P0

- **M0-001** [Todos] Congelar contratos y estado real de repos — depende de: sin dependencia
- **M0-002** [Todos] Ejecutar baseline de tests actual — depende de: M0-001
- **AI-001** [AI Module] Validar checkpoints finales reales en entorno controlado — depende de: M0-002
- **AI-002** [AI Module] Endurecer validación de manifests — depende de: AI-001
- **AI-003** [AI Module] Verificar hash antes de reemplazar artifacts descargados — depende de: AI-002
- **AI-005** [AI Module] Probar carga real de modelo sagital final — depende de: AI-001,AI-002
- **AI-006** [AI Module] Probar carga real de modelo axial final — depende de: AI-001,AI-002
- **AI-007** [AI Module] Strict real_baseline para sagital con input de prueba — depende de: AI-005
- **AI-008** [AI Module] Strict real_baseline para axial con input de prueba — depende de: AI-006
- **AI-009** [AI Module] Ordenar serie DICOM axial por metadata — depende de: AI-008
- **AI-012** [AI Module] Prueba real de POST /multiplanar/run con ambos modelos — depende de: AI-007,AI-008
- **AI-013** [AI Module] Diseñar inputId server-side para archivos cargados — depende de: AI-012
- **AI-014** [AI Module] Implementar upload seguro en AI Module — depende de: AI-013
- **AI-015** [AI Module] Implementar asset registry por runId — depende de: AI-014
- **AI-016** [AI Module] Servir assets seguros desde AI Module — depende de: AI-015
- **BE-001** [Backend] Congelar DTOs de multiplanar run — depende de: M0-002
- **BE-002** [Backend] Proxy de upload hacia AI Module — depende de: AI-014,BE-001
- **BE-003** [Backend] Proxy de assets seguro — depende de: AI-016,BE-001
- **BE-004** [Backend] Ejecutar multiplanar run usando inputId — depende de: BE-002,AI-012
- **BE-005** [Backend] Modelo de datos Study/Series/Input/Run — depende de: BE-004
- **BE-006** [Backend] Persistir respuesta de corrida — depende de: BE-005
- **BE-007** [Backend] Modelo de revisión profesional — depende de: BE-006
- **BE-009** [Backend] Auditoría de eventos clave — depende de: BE-007
- **BE-010** [Backend] Roles para acciones sensibles — depende de: BE-009
- **FE-001** [Frontend] Actualizar tipos frontend según contrato final — depende de: BE-001,BE-004
- **FE-002** [Frontend] Vincular workspace al caso seleccionado — depende de: FE-001
- **FE-003** [Frontend] Pantalla/flujo de carga de RM — depende de: BE-002,FE-002
- **FE-004** [Frontend] Ejecutar análisis desde inputs cargados — depende de: FE-003,BE-004
- **FE-005** [Frontend] Renderizar overlays servidos por backend — depende de: BE-003,FE-004
- **FE-006** [Frontend] Viewer sagital/axial lado a lado — depende de: FE-005
- **FE-008** [Frontend] Mostrar landmarks y mediciones — depende de: FE-006
- **FE-009** [Frontend] Formulario de revisión profesional — depende de: BE-007,FE-008
- **FE-011** [Frontend] Panel de readiness para demo — depende de: BE-003,BE-004
- **DEV-001** [AI Module] Dockerfile AI con dependencias PyTorch/SimpleITK — depende de: AI-007,AI-008
- **DEV-002** [Todos] Docker Compose local completo — depende de: DEV-001,BE-006,FE-004
- **DEV-003** [Todos] Variables de entorno documentadas — depende de: DEV-002
- **DEV-006** [AI Module] Estrategia de storage de artifacts final — depende de: AI-003
- **TEST-001** [Todos] Contrato JSON golden para multiplanar run — depende de: AI-012,BE-004,FE-001
- **TEST-002** [Todos] E2E local contract mode — depende de: DEV-002,FE-004
- **TEST-003** [Todos] E2E local real_baseline mode — depende de: AI-007,AI-008,DEV-002
- **TEST-004** [AI+Backend] Pruebas de seguridad de uploads/assets — depende de: AI-014,AI-016,BE-003
- **VAL-001** [Docs+AI] Set de casos demo deidentificados — depende de: AI-014,TEST-003
- **VAL-004** [Docs] Guion de demo final — depende de: TEST-002,TEST-003,FE-011
- **DOC-001** [Docs] Actualizar requerimientos funcionales/no funcionales — depende de: BE-007,FE-009
- **DOC-002** [Docs+Backend] ERD final y modelo de datos — depende de: BE-005,BE-007,BE-009
- **DOC-003** [Docs+Frontend] Capturas finales de demo — depende de: VAL-004
- **DOC-005** [Docs] Anexo técnico de instalación y ejecución — depende de: DEV-002,DEV-003
- **QUAL-001** [AI Module] Definir métrica y umbral de aceptación (>=70%) — depende de: sin dependencia *(revisión crítica)*
- **QUAL-002** [AI Module/Docs] Congelar test held-out deidentificado (split por paciente) — depende de: QUAL-001 *(revisión crítica)*
- **QUAL-003** [AI Module] Evaluador reproducible por clase — depende de: QUAL-002,AI-005,AI-006 *(revisión crítica)*
- **QUAL-004** [AI Module] Medir baseline actual (modelo chico) contra test held-out — depende de: QUAL-003 *(revisión crítica)*
- **QUAL-005** [AI Module/Colab] Escalar entrenamiento hasta el umbral — depende de: QUAL-004 *(revisión crítica)*
- **QUAL-006** [AI Module/Docs] Model card / lineage por artifact — depende de: QUAL-001 *(revisión crítica)*
- **PERF-001** [AI Module] Medir latencia de inferencia local — depende de: AI-007,AI-008 *(revisión crítica)*
- **DEPLOY-001** [Todos] Estrategia de hosting de inferencia (serve, no train) — depende de: DEV-001,QUAL-005 *(revisión crítica)*
- **DEPLOY-002** [AI Module] Materialización de artifacts en cloud — depende de: DEPLOY-001,DEV-006 *(revisión crítica)*
- **DEPLOY-003** [Todos] Prueba en vivo end-to-end en cloud — depende de: DEPLOY-002,FE-004 *(revisión crítica)*
- **VAL-CLIN-001** [Docs] Protocolo de validación por médicos — depende de: QUAL-005,FE-009 *(revisión crítica)*

## Tickets por fase


### M0

#### M0-001 — Congelar contratos y estado real de repos
- Epic: GOB
- Repo: Todos
- Prioridad: P0
- Dependencias: Ninguna
- Descripción: Relevar endpoints, DTOs, tipos frontend y módulos AI existentes para evitar duplicar o romper contratos.
- Criterios de aceptación: Inventario de endpoints, tipos y DTOs; lista de brechas reales; backlog ajustado si aparece deuda no registrada.
- Tests/evidencia: npm test/build, mvn test/package, pytest mínimo o reporte de fallos reales.

#### M0-002 — Ejecutar baseline de tests actual
- Epic: TEST
- Repo: Todos
- Prioridad: P0
- Dependencias: M0-001
- Descripción: Ejecutar suites actuales de AI, backend y frontend antes de agregar más funcionalidad.
- Criterios de aceptación: Se documentan comandos, resultado y fallos. Ningún ticket P0 nuevo se inicia sin saber si la base compila.
- Tests/evidencia: AI: pytest. Backend: mvn test. Frontend: npm run build/test.

#### M0-003 — Crear archivo BACKLOG_MASTER.md en repositorio de documentación
- Epic: GOB
- Repo: Docs
- Prioridad: P1
- Dependencias: M0-001
- Descripción: Versionar el backlog maestro y enlazarlo con la planificación técnica.
- Criterios de aceptación: Archivo versionado, actualizado con IDs de tickets y criterio de ejecución por agente.
- Tests/evidencia: Revisión manual.

#### M0-004 — Definir contrato de prompts Planner -> Codex
- Epic: GOB
- Repo: Docs
- Prioridad: P1
- Dependencias: M0-003
- Descripción: Estandarizar cómo el planificador entrega tickets al agente de código.
- Criterios de aceptación: Template de prompt con contexto, repositorio, archivos, restricciones, DoD y tests.
- Tests/evidencia: Prueba con un ticket pequeño de documentación o test.


### M1

#### AI-001 — Validar checkpoints finales reales en entorno controlado
- Epic: AI-ARTIFACT
- Repo: AI Module
- Prioridad: P0
- Dependencias: M0-002
- Descripción: Ejecutar script de inspección sobre sagittal_spider_multiclass_final_best.pt y axial_t2_alkafri_final_best.pt.
- Criterios de aceptación: Se confirma existencia, tamaño, SHA-256, checkpoint keys, state_dict keys, num_classes, base_channels y target_size.
- Tests/evidencia: Script en Colab/local con salida pegada en repo/docs.

#### AI-002 — Endurecer validación de manifests
- Epic: AI-ARTIFACT
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-001
- Descripción: Validar consistencia estricta entre manifest, registry y artifact real.
- Criterios de aceptación: modelKey, artifactFile, inputPlane, classes, metrics 0..1, dataset, version y optional sha coinciden.
- Tests/evidencia: pytest unitario para manifest válido, inválido y mismatch.

#### AI-003 — Verificar hash antes de reemplazar artifacts descargados
- Epic: AI-ARTIFACT
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-002
- Descripción: Evitar reemplazar un modelo local válido por una descarga corrupta.
- Criterios de aceptación: Descarga a tmp, verificación SHA-256, replace atómico solo si pasa.
- Tests/evidencia: pytest con artifact fake válido e inválido.

#### AI-004 — Lock de sincronización de modelos
- Epic: AI-ARTIFACT
- Repo: AI Module
- Prioridad: P1
- Dependencias: AI-003
- Descripción: Evitar sync concurrente que corrompa tmp o artifacts.
- Criterios de aceptación: Lock por proceso/archivo o mutex; respuesta clara si ya hay sync en curso.
- Tests/evidencia: pytest o test manual con dos requests simultáneos.

#### AI-005 — Probar carga real de modelo sagital final
- Epic: AI-RUNTIME
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-001,AI-002
- Descripción: Ejecutar torch.load + arquitectura + load_state_dict(strict=True) con el artifact final sagital.
- Criterios de aceptación: El modelo carga y queda en eval sin errores. Si falla, se corrige adaptador de arquitectura.
- Tests/evidencia: pytest o script con artifact real local.

#### AI-006 — Probar carga real de modelo axial final
- Epic: AI-RUNTIME
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-001,AI-002
- Descripción: Ejecutar torch.load + arquitectura + load_state_dict(strict=True) con el artifact final axial.
- Criterios de aceptación: El modelo carga y queda en eval sin errores. Si falla, se corrige adaptador de arquitectura.
- Tests/evidencia: pytest o script con artifact real local.

#### AI-007 — Strict real_baseline para sagital con input de prueba
- Epic: AI-RUNTIME
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-005
- Descripción: Ejecutar /pipeline/run con requested real_baseline y allowContractFallback=false usando input sagital válido.
- Criterios de aceptación: effectiveInferenceMode=real_baseline; genera input.png, mask.npy, confidence.npy, overlay.png.
- Tests/evidencia: pytest/integration test con artifact real o smoke local documentado.

#### AI-008 — Strict real_baseline para axial con input de prueba
- Epic: AI-RUNTIME
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-006
- Descripción: Ejecutar /pipeline/run con requested real_baseline y allowContractFallback=false usando input axial válido.
- Criterios de aceptación: effectiveInferenceMode=real_baseline; genera input.png, mask.npy, confidence.npy, overlay.png.
- Tests/evidencia: pytest/integration test con artifact real o smoke local documentado.

#### BE-001 — Congelar DTOs de multiplanar run
- Epic: BE-CONTRACT
- Repo: Backend
- Prioridad: P0
- Dependencias: M0-002
- Descripción: Alinear DTOs Java con respuesta actual del AI Module.
- Criterios de aceptación: No campos críticos perdidos: runId, traceId, effectiveMode, planes, assets, review.
- Tests/evidencia: mvn test con mock de respuesta realista.

### M1.5 — Calidad y evaluación del modelo *(revisión crítica)*

#### QUAL-001 — Definir métrica y umbral de aceptación (>=70%)
- Epic: QUAL
- Repo: AI Module / Docs
- Prioridad: P0
- Dependencias: Ninguna
- Descripción: Operacionalizar el "70%". Definir métrica primaria (p. ej. Dice macro sin fondo) y umbral por clase/global, y de dónde sale el requisito.
- Criterios de aceptación: métrica primaria y secundarias definidas; umbral fijado; documentado en docs/.
- Tests/evidencia: documento revisado.

#### QUAL-002 — Congelar test held-out deidentificado (split por paciente)
- Epic: QUAL
- Repo: AI Module / Docs
- Prioridad: P0
- Dependencias: QUAL-001
- Descripción: Separar test set que NO se usó para entrenar ni early-stopping; split por paciente (no por slice) para evitar fuga; versionar el listado.
- Criterios de aceptación: split reproducible sin fuga; lista de casos fija; deidentificación verificada.
- Tests/evidencia: script de split + hash del listado.

#### QUAL-003 — Evaluador reproducible por clase
- Epic: QUAL
- Repo: AI Module
- Prioridad: P0
- Dependencias: QUAL-002,AI-005,AI-006
- Descripción: Evaluar un checkpoint sobre el test held-out; reportar métrica global + por clase + matriz de confusión / Dice por estructura.
- Criterios de aceptación: corrida reproducible; número por clase; sin fuga.
- Tests/evidencia: pytest del evaluador + corrida real documentada.

#### QUAL-004 — Medir baseline actual (modelo chico) contra test held-out
- Epic: QUAL
- Repo: AI Module
- Prioridad: P0
- Dependencias: QUAL-003
- Descripción: Correr E12 sagital y E10 axial sobre el test held-out y reportar el número real de hoy; explicitar la brecha contra el umbral.
- Criterios de aceptación: número real por plano documentado sin maquillar.
- Tests/evidencia: reporte en docs/.

#### QUAL-005 — Escalar entrenamiento hasta el umbral
- Epic: QUAL
- Repo: AI Module / Colab
- Prioridad: P0
- Dependencias: QUAL-004
- Descripción: Ruta de mejora: dataset completo, más capacidad (base_channels), augmentation, regularización, más épocas. Entrenamiento en Colab; el repo consume el .pt resultante.
- Criterios de aceptación: checkpoint que alcanza >= umbral en test held-out; manifest actualizado con SHA/métricas nuevas.
- Tests/evidencia: reporte de QUAL-003 sobre el nuevo checkpoint superando el umbral.

#### QUAL-006 — Model card / lineage por artifact
- Epic: QUAL
- Repo: AI Module / Docs
- Prioridad: P1
- Dependencias: QUAL-001
- Descripción: Ficha por .pt: notebook de origen, dataset, split, hiperparámetros (num_classes, base_channels, target_size), métricas en test, fecha, versión, SHA-256. Resuelve la ambigüedad E9/E10 y el target_size axial perdido.
- Criterios de aceptación: model card por modelo en models/final/; campos dataset/version del manifest poblados con datos reales.
- Tests/evidencia: revisión + validación de manifest (AI-002).


### M2

#### AI-009 — Ordenar serie DICOM axial por metadata
- Epic: AI-RUNTIME
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-008
- Descripción: Reemplazar selección lexicográfica por InstanceNumber/ImagePositionPatient cuando se reciba directorio DICOM.
- Criterios de aceptación: La selección de corte axial es estable y trazable; fallback documentado si falta metadata.
- Tests/evidencia: pytest con DICOM sintético o mock de metadata.

#### AI-010 — Mejorar contornos de máscaras
- Epic: AI-RUNTIME
- Repo: AI Module
- Prioridad: P1
- Dependencias: AI-007,AI-008
- Descripción: Reemplazar contorno por orden angular por contornos por componente cuando sea posible.
- Criterios de aceptación: Contornos no unen componentes desconectados; mantiene estructura por clase/componente.
- Tests/evidencia: pytest sobre máscaras con dos componentes.

#### AI-011 — Resample o metadata de coordenadas originales
- Epic: AI-RUNTIME
- Repo: AI Module
- Prioridad: P1
- Dependencias: AI-007,AI-008
- Descripción: Documentar y/o implementar mapeo de outputs 256x256 a geometría original.
- Criterios de aceptación: Respuesta indica coordinateSpace=model_256 o original_image; no hay ambigüedad.
- Tests/evidencia: pytest de metadata en respuesta.

#### AI-012 — Prueba real de POST /multiplanar/run con ambos modelos
- Epic: AI-MULTI
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-007,AI-008
- Descripción: Ejecutar una corrida que combine sagital y axial reales en un mismo workspace.
- Criterios de aceptación: multiplanarRunId común; child runIds; ambos planes real_baseline o error explícito.
- Tests/evidencia: pytest/integration test o smoke con capturas JSON.

#### AI-013 — Diseñar inputId server-side para archivos cargados
- Epic: AI-INGEST
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-012
- Descripción: Evitar que backend/frontend pasen paths internos al pipeline.
- Criterios de aceptación: POST /inputs devuelve inputId, caseId, plane, format, size; pipeline acepta inputId.
- Tests/evidencia: pytest endpoint inputs + pipeline con inputId.

#### AI-014 — Implementar upload seguro en AI Module
- Epic: AI-INGEST
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-013
- Descripción: Recibir multipart con RM deidentificada o demo, validar extensión/tamaño y guardar con nombre server-side.
- Criterios de aceptación: No path traversal; extensiones permitidas; límite de tamaño; respuesta sin paths internos.
- Tests/evidencia: pytest upload válido/inválido/path traversal.

#### AI-015 — Implementar asset registry por runId
- Epic: AI-ASSET
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-014
- Descripción: Registrar assets generados sin exponer filesystem arbitrario.
- Criterios de aceptación: runId/plane/assetName se resuelve solo a assets allowlist.
- Tests/evidencia: pytest asset registry.

#### AI-016 — Servir assets seguros desde AI Module
- Epic: AI-ASSET
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-015
- Descripción: GET /assets/{runId}/{plane}/{assetName} para input/overlay/mask-preview.
- Criterios de aceptación: Solo assets permitidos; no se sirve .pt, .npy raw al browser salvo endpoint interno controlado.
- Tests/evidencia: pytest 200 allowlist, 404/403 para path traversal y asset no permitido.

#### AI-017 — Generar mask-preview PNG por clase
- Epic: AI-ASSET
- Repo: AI Module
- Prioridad: P1
- Dependencias: AI-016
- Descripción: Además de mask.npy, generar previews visuales para frontend.
- Criterios de aceptación: Por clase: mask preview, overlay global y metadata de color/opacidad.
- Tests/evidencia: pytest de existencia y metadata.

#### BE-002 — Proxy de upload hacia AI Module
- Epic: BE-AI
- Repo: Backend
- Prioridad: P0
- Dependencias: AI-014,BE-001
- Descripción: Exponer POST /api/ai/inputs y reenviar multipart al AI Module.
- Criterios de aceptación: Backend no guarda paths inseguros; valida tamaño/tipo; devuelve inputId.
- Tests/evidencia: mvn test controller + client mock.

#### BE-003 — Proxy de assets seguro
- Epic: BE-AI
- Repo: Backend
- Prioridad: P0
- Dependencias: AI-016,BE-001
- Descripción: Exponer GET /api/ai/assets/{runId}/{plane}/{assetName} y streamear desde AI Module.
- Criterios de aceptación: Frontend nunca llama directo al AI Module; no path traversal; content-type correcto.
- Tests/evidencia: mvn test 200/404/path traversal.

#### BE-004 — Ejecutar multiplanar run usando inputId
- Epic: BE-AI
- Repo: Backend
- Prioridad: P0
- Dependencias: BE-002,AI-012
- Descripción: Actualizar POST /api/ai/multiplanar/run para aceptar inputIds por plano.
- Criterios de aceptación: Permite selected case + inputs reales; conserva fallback explícito.
- Tests/evidencia: mvn test con request realista.

#### FE-001 — Actualizar tipos frontend según contrato final
- Epic: FE-CONTRACT
- Repo: Frontend
- Prioridad: P0
- Dependencias: BE-001,BE-004
- Descripción: Alinear TypeScript types para inputs, assets, runs, review y runtime status.
- Criterios de aceptación: No any innecesario en contratos críticos; errores tipados.
- Tests/evidencia: npm run build.

#### FE-002 — Vincular workspace al caso seleccionado
- Epic: FE-CASE
- Repo: Frontend
- Prioridad: P0
- Dependencias: FE-001
- Descripción: Eliminar dependencia de caseId demo fijo en flujo principal.
- Criterios de aceptación: Al elegir un estudio, el workspace usa ese caseId y sus inputs/runs.
- Tests/evidencia: npm build + prueba manual.

#### FE-003 — Pantalla/flujo de carga de RM
- Epic: FE-UPLOAD
- Repo: Frontend
- Prioridad: P0
- Dependencias: BE-002,FE-002
- Descripción: Permitir cargar archivos permitidos para sagital y/o axial.
- Criterios de aceptación: UI muestra validación, progreso, inputId y errores claros.
- Tests/evidencia: npm build + test component si existe.

#### FE-004 — Ejecutar análisis desde inputs cargados
- Epic: FE-RUN
- Repo: Frontend
- Prioridad: P0
- Dependencias: FE-003,BE-004
- Descripción: Botón de análisis usa inputIds y muestra estado de ejecución.
- Criterios de aceptación: Al finalizar muestra runId, traceId, effective modes y planes.
- Tests/evidencia: npm build + prueba manual.

#### FE-005 — Renderizar overlays servidos por backend
- Epic: FE-ASSET
- Repo: Frontend
- Prioridad: P0
- Dependencias: BE-003,FE-004
- Descripción: Mostrar input/overlay/mask-preview usando URLs del backend.
- Criterios de aceptación: No usa paths locales del AI Module; maneja 404/error.
- Tests/evidencia: npm build + prueba manual.


### M3

#### AI-018 — Normalizar quality schema final
- Epic: AI-QUALITY
- Repo: AI Module
- Prioridad: P1
- Dependencias: AI-012
- Descripción: Definir quality summary común para planes reales y contract.
- Criterios de aceptación: Frontend recibe confidence, flags, warnings, realInferenceFailure si aplica.
- Tests/evidencia: pytest schema.

#### BE-005 — Modelo de datos Study/Series/Input/Run
- Epic: BE-DOMAIN
- Repo: Backend
- Prioridad: P0
- Dependencias: BE-004
- Descripción: Crear entidades o tablas para representar estudios de demo/deidentificados y corridas.
- Criterios de aceptación: Study, StudySeries/InputResource, StudyRun con IDs, estado, timestamps y trazabilidad.
- Tests/evidencia: mvn test repository/service; migration validada.

#### BE-006 — Persistir respuesta de corrida
- Epic: BE-DOMAIN
- Repo: Backend
- Prioridad: P0
- Dependencias: BE-005
- Descripción: Guardar resumen de cada run y enlaces a assets sin guardar blobs pesados en DB.
- Criterios de aceptación: runId, modelKey, modelVersion, artifactHash, modes, quality, assets, traceId persistidos.
- Tests/evidencia: mvn test service + repository.

#### BE-007 — Modelo de revisión profesional
- Epic: BE-REVIEW
- Repo: Backend
- Prioridad: P0
- Dependencias: BE-006
- Descripción: Persistir decisiones aceptado/observado/descartado y correcciones.
- Criterios de aceptación: ProfessionalReview con reviewer, timestamp, status, notes y cambios de medición.
- Tests/evidencia: mvn test CRUD y validaciones.

#### BE-008 — Versionado de mediciones corregidas
- Epic: BE-REVIEW
- Repo: Backend
- Prioridad: P1
- Dependencias: BE-007
- Descripción: Guardar valor automático y valor editado sin perder original.
- Criterios de aceptación: MeasurementReview o estructura equivalente con before/after y motivo opcional.
- Tests/evidencia: mvn test.

#### BE-009 — Auditoría de eventos clave
- Epic: BE-AUDIT
- Repo: Backend
- Prioridad: P0
- Dependencias: BE-007
- Descripción: Registrar login, upload, sync, run, review y errores significativos.
- Criterios de aceptación: AuditEvent con actor, action, entityId, traceId, timestamp y metadata segura.
- Tests/evidencia: mvn test audit service.

#### BE-010 — Roles para acciones sensibles
- Epic: BE-SEC
- Repo: Backend
- Prioridad: P0
- Dependencias: BE-009
- Descripción: Restringir sync/cache clear/admin diagnostics según rol.
- Criterios de aceptación: Profesional puede revisar; admin puede sync/cache; endpoints protegidos.
- Tests/evidencia: mvn security tests.

#### BE-011 — Normalizar errores AI como 502/503 cuando corresponde
- Epic: BE-ERROR
- Repo: Backend
- Prioridad: P1
- Dependencias: BE-004
- Descripción: Evitar HTTP 200 para fallos de dependencia salvo endpoints diagnostics explícitos.
- Criterios de aceptación: Controllers de operación devuelven status semántico; UI puede mostrar error real.
- Tests/evidencia: mvn controller tests.

#### FE-006 — Viewer sagital/axial lado a lado
- Epic: FE-VIEWER
- Repo: Frontend
- Prioridad: P0
- Dependencias: FE-005
- Descripción: Dos paneles con imagen base, overlay, opacidad y metadata del plano.
- Criterios de aceptación: Puede mostrar sagital y axial simultáneamente; oculta panel faltante con estado claro.
- Tests/evidencia: npm build + capturas demo.

#### FE-007 — Control de capas por clase
- Epic: FE-VIEWER
- Repo: Frontend
- Prioridad: P1
- Dependencias: FE-006,AI-017
- Descripción: Mostrar/ocultar clases, cambiar opacidad y ver leyenda.
- Criterios de aceptación: Cada máscara tiene label/color/confidence si está disponible.
- Tests/evidencia: npm build + prueba manual.

#### FE-008 — Mostrar landmarks y mediciones
- Epic: FE-VIEWER
- Repo: Frontend
- Prioridad: P0
- Dependencias: FE-006
- Descripción: Mostrar centroides/landmarks y tabla de mediciones por plano/clase.
- Criterios de aceptación: Las unidades y coordinateSpace se ven claramente.
- Tests/evidencia: npm build + capturas demo.

#### FE-009 — Formulario de revisión profesional
- Epic: FE-REVIEW
- Repo: Frontend
- Prioridad: P0
- Dependencias: BE-007,FE-008
- Descripción: Permitir aceptar, observar o descartar un run con notas.
- Criterios de aceptación: Guarda decisión y refresca estado de caso/run.
- Tests/evidencia: npm build + test/manual.

#### FE-011 — Panel de readiness para demo
- Epic: FE-DIAG
- Repo: Frontend
- Prioridad: P0
- Dependencias: BE-003,BE-004
- Descripción: Mostrar health, readiness, model status, sync status y runtime status.
- Criterios de aceptación: Permite evidenciar que la demo no depende de mock silencioso.
- Tests/evidencia: npm build + captura demo.

#### TEST-001 — Contrato JSON golden para multiplanar run
- Epic: TEST
- Repo: Todos
- Prioridad: P0
- Dependencias: AI-012,BE-004,FE-001
- Descripción: Guardar ejemplos JSON contract y real_baseline para validar compatibilidad.
- Criterios de aceptación: AI, backend y frontend consumen el mismo contrato sin divergencia.
- Tests/evidencia: snapshot/schema tests.


### M4

#### BE-012 — OpenAPI o documentación de endpoints
- Epic: BE-CONTRACT
- Repo: Backend
- Prioridad: P1
- Dependencias: BE-004,BE-007
- Descripción: Generar documentación de API para frontend y defensa.
- Criterios de aceptación: Endpoints principales documentados con request/response.
- Tests/evidencia: Build y revisión manual.

#### FE-010 — Edición de mediciones
- Epic: FE-REVIEW
- Repo: Frontend
- Prioridad: P1
- Dependencias: BE-008,FE-009
- Descripción: Editar valor de medición conservando original.
- Criterios de aceptación: Se ve valor automático, valor editado, unidad y autor de cambio.
- Tests/evidencia: npm build + prueba manual.

#### FE-012 — Historial de corridas por caso
- Epic: FE-HISTORY
- Repo: Frontend
- Prioridad: P1
- Dependencias: BE-006
- Descripción: Listar runs previos y estado de revisión.
- Criterios de aceptación: Permite abrir run histórico y ver assets si existen.
- Tests/evidencia: npm build.

#### DEV-001 — Dockerfile AI con dependencias PyTorch/SimpleITK
- Epic: DEVOPS
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-007,AI-008
- Descripción: Asegurar que AI Module pueda construir y arrancar con torch, pydicom, SimpleITK.
- Criterios de aceptación: Imagen construye; health responde; tamaño/tiempo documentado.
- Tests/evidencia: docker build/run local o CI.

#### DEV-002 — Docker Compose local completo
- Epic: DEVOPS
- Repo: Todos
- Prioridad: P0
- Dependencias: DEV-001,BE-006,FE-004
- Descripción: Levantar frontend, backend, AI Module y PostgreSQL localmente.
- Criterios de aceptación: Un comando documentado permite correr flujo demo local.
- Tests/evidencia: docker compose up + smoke.

#### DEV-003 — Variables de entorno documentadas
- Epic: DEVOPS
- Repo: Todos
- Prioridad: P0
- Dependencias: DEV-002
- Descripción: Documentar env vars de frontend/backend/AI, tokens y defaults seguros.
- Criterios de aceptación: .env.example por repo; sin secretos reales.
- Tests/evidencia: Revisión manual + build.

#### DEV-004 — Healthcheck y readiness en deployment
- Epic: DEVOPS
- Repo: Todos
- Prioridad: P1
- Dependencias: DEV-002
- Descripción: Configurar health/readiness para servicios desplegados.
- Criterios de aceptación: Cada servicio reporta estado y dependencias críticas.
- Tests/evidencia: Smoke deploy.

#### DEV-006 — Estrategia de storage de artifacts final
- Epic: DEVOPS
- Repo: AI Module
- Prioridad: P0
- Dependencias: AI-003
- Descripción: Definir storage final para .pt y manifests con acceso privado.
- Criterios de aceptación: URI/token configurables; sync probado; no secretos en repo.
- Tests/evidencia: sync smoke + documentación.

#### TEST-002 — E2E local contract mode
- Epic: TEST
- Repo: Todos
- Prioridad: P0
- Dependencias: DEV-002,FE-004
- Descripción: Flujo completo sin modelos reales, útil para disponibilidad de demo.
- Criterios de aceptación: Login/worklist/run contract/review funciona.
- Tests/evidencia: Playwright o checklist manual con capturas.

#### TEST-003 — E2E local real_baseline mode
- Epic: TEST
- Repo: Todos
- Prioridad: P0
- Dependencias: AI-007,AI-008,DEV-002
- Descripción: Flujo completo con artifacts reales.
- Criterios de aceptación: Upload/selección, run real, assets, review y persistencia funcionan.
- Tests/evidencia: Playwright/manual + logs/capturas.

#### TEST-004 — Pruebas de seguridad de uploads/assets
- Epic: TEST
- Repo: AI+Backend
- Prioridad: P0
- Dependencias: AI-014,AI-016,BE-003
- Descripción: Verificar límites, extensiones, path traversal, assets no permitidos.
- Criterios de aceptación: Casos maliciosos bloqueados con status correcto.
- Tests/evidencia: pytest + mvn tests.

#### VAL-001 — Set de casos demo deidentificados
- Epic: VALIDATION
- Repo: Docs+AI
- Prioridad: P0
- Dependencias: AI-014,TEST-003
- Descripción: Preparar casos demo reproducibles para sagital y axial sin PII.
- Criterios de aceptación: Carpeta/manifest con inputs, origen, licencia y uso en demo.
- Tests/evidencia: Revisión manual + smoke run.


### M5

#### FE-013 — Pulido UX para demo final
- Epic: FE-UX
- Repo: Frontend
- Prioridad: P1
- Dependencias: FE-006,FE-009,FE-011
- Descripción: Mejorar textos, estados vacíos, loaders, errores y layout responsive.
- Criterios de aceptación: Flujo de demo claro sin explicar desde consola.
- Tests/evidencia: Capturas + npm build.

#### FE-014 — Panel 3D futuro deshabilitado pero honesto
- Epic: FE-3D
- Repo: Frontend
- Prioridad: P2
- Dependencias: FE-006
- Descripción: Mostrar tercer panel como evolución futura solo si no hay reconstrucción real.
- Criterios de aceptación: Debe indicar requisitos faltantes: stack, spacing, orientación, máscaras volumétricas.
- Tests/evidencia: npm build + captura.

#### DEV-005 — CI mínimo por repositorio
- Epic: DEVOPS
- Repo: Todos
- Prioridad: P1
- Dependencias: M0-002
- Descripción: Ejecutar build/test en cada push/PR.
- Criterios de aceptación: GitHub Actions o equivalente: AI pytest, backend mvn test, frontend build.
- Tests/evidencia: CI verde.

#### TEST-005 — Prueba de degradación explícita
- Epic: TEST
- Repo: Todos
- Prioridad: P1
- Dependencias: TEST-003
- Descripción: Simular artifact ausente/fallo PyTorch y verificar UI/Backend/AI.
- Criterios de aceptación: No se oculta fallback; UI muestra modo efectivo y failure info.
- Tests/evidencia: tests + captura.

#### TEST-006 — Performance smoke CPU
- Epic: TEST
- Repo: AI Module
- Prioridad: P1
- Dependencias: AI-007,AI-008
- Descripción: Medir carga inicial y latencia de inferencia CPU en ambiente objetivo.
- Criterios de aceptación: Tiempos documentados; cache evita recarga innecesaria.
- Tests/evidencia: script benchmark.

#### VAL-002 — Checklist de revisión profesional
- Epic: VALIDATION
- Repo: Docs+Frontend
- Prioridad: P1
- Dependencias: FE-009
- Descripción: Crear formulario de evaluación cualitativa sobre usabilidad/confianza.
- Criterios de aceptación: Preguntas y escala 1-5 integradas en documentación o pantalla auxiliar.
- Tests/evidencia: Revisión con al menos un profesional si se consigue.

#### VAL-004 — Guion de demo final
- Epic: VALIDATION
- Repo: Docs
- Prioridad: P0
- Dependencias: TEST-002,TEST-003,FE-011
- Descripción: Escribir recorrido de demo con capturas esperadas y plan B.
- Criterios de aceptación: Guion minuto a minuto; evidencia de cada capa.
- Tests/evidencia: Ensayo de demo.

#### DOC-001 — Actualizar requerimientos funcionales/no funcionales
- Epic: DOC
- Repo: Docs
- Prioridad: P0
- Dependencias: BE-007,FE-009
- Descripción: Formalizar RF/RNF finales del producto implementado.
- Criterios de aceptación: RF/RNF trazables a tickets, endpoints y pantallas.
- Tests/evidencia: Revisión documental.

#### DOC-002 — ERD final y modelo de datos
- Epic: DOC
- Repo: Docs+Backend
- Prioridad: P0
- Dependencias: BE-005,BE-007,BE-009
- Descripción: Documentar diagrama de base de datos as-is/to-be.
- Criterios de aceptación: Entidades, relaciones y justificación.
- Tests/evidencia: Diagrama + revisión.

#### DOC-003 — Capturas finales de demo
- Epic: DOC
- Repo: Docs+Frontend
- Prioridad: P0
- Dependencias: VAL-004
- Descripción: Generar pack de capturas para rúbrica y entrega final.
- Criterios de aceptación: Login, worklist, readiness, sync, run, viewer, review, auditoría.
- Tests/evidencia: Capturas versionadas.


### M6

#### VAL-003 — Matriz de feedback y ajustes
- Epic: VALIDATION
- Repo: Docs
- Prioridad: P1
- Dependencias: VAL-002
- Descripción: Registrar feedback profesional y convertirlo en tickets.
- Criterios de aceptación: Tabla problema/impacto/decisión/ticket.
- Tests/evidencia: Documento o issue tracker.

#### DOC-004 — Actualizar capítulos 18/19 con evidencia final
- Epic: DOC
- Repo: Docs
- Prioridad: P1
- Dependencias: TEST-003,DOC-003
- Descripción: Reemplazar pendientes por resultados reales donde corresponda.
- Criterios de aceptación: No quedan afirmaciones de scaffold si el producto está integrado.
- Tests/evidencia: Revisión cruzada.

#### DOC-005 — Anexo técnico de instalación y ejecución
- Epic: DOC
- Repo: Docs
- Prioridad: P0
- Dependencias: DEV-002,DEV-003
- Descripción: Documentar cómo levantar el producto y ejecutar demo.
- Criterios de aceptación: Comandos, env vars, troubleshooting y plan B.
- Tests/evidencia: Prueba por una persona externa/equipo.

#### 3D-001 — Spike de requisitos para 3D real
- Epic: 3D
- Repo: AI+Frontend+Docs
- Prioridad: P2
- Dependencias: TEST-003
- Descripción: Definir requisitos mínimos para reconstrucción 3D paciente-específica.
- Criterios de aceptación: Documento: stack, spacing, orientation, segmentación por slice, marching cubes.
- Tests/evidencia: Informe de factibilidad.

#### 3D-002 — Mock honesto de panel 3D futuro
- Epic: 3D
- Repo: Frontend
- Prioridad: P3
- Dependencias: FE-014,3D-001
- Descripción: Mostrar diseño futuro sin simular resultado clínico inexistente.
- Criterios de aceptación: Panel indica estado experimental/deshabilitado.
- Tests/evidencia: npm build + captura.


## Epics agregados: rendimiento, deploy y validación clínica *(revisión crítica)*

#### PERF-001 — Medir latencia de inferencia local
- Epic: PERF
- Repo: AI Module
- Prioridad: P1
- Dependencias: AI-007,AI-008
- Descripción: Medir tiempo por plano y multiplanar (carga + preproceso + forward + postproceso) en local.
- Criterios de aceptación: p50/p95 documentados por plano y multiplanar.
- Tests/evidencia: script de medición + tabla en docs/.

#### PERF-002 — Presupuesto de latencia para UX del frontend
- Epic: PERF
- Repo: Frontend / Docs
- Prioridad: P2
- Dependencias: PERF-001
- Descripción: Definir latencia aceptable y cómo la absorbe el frontend (async, spinner, estado de progreso).
- Criterios de aceptación: presupuesto acordado; contrato de estados de corrida para el front.

#### DEPLOY-001 — Estrategia de hosting de inferencia (serve, no train)
- Epic: DEPLOY
- Repo: Todos
- Prioridad: P1
- Dependencias: DEV-001,QUAL-005
- Descripción: Definir dónde/cómo se hostea AI Module + backend para uso en vivo. Guardarraíl: para el MVP NO se monta entrenamiento en la nube; entrenar queda en Colab, la nube solo sirve inferencia.
- Criterios de aceptación: arquitectura de deploy documentada; separación train (Colab) / serve (cloud); recursos estimados desde PERF-001.

#### DEPLOY-002 — Materialización de artifacts en cloud
- Epic: DEPLOY
- Repo: AI Module
- Prioridad: P1
- Dependencias: DEPLOY-001,DEV-006
- Descripción: Los .pt (fuera de git) se materializan por config en cloud, con verificación SHA (AI-003).
- Criterios de aceptación: arranque en cloud materializa y valida artifacts; sin .pt en el repo.

#### DEPLOY-003 — Prueba en vivo end-to-end en cloud
- Epic: DEPLOY
- Repo: Todos
- Prioridad: P1
- Dependencias: DEPLOY-002,FE-004
- Descripción: Subir una RM real deidentificada, correr el pipeline en cloud, medir tiempo real y visualizar en el frontend.
- Criterios de aceptación: corrida real en cloud visible en el front; latencia real registrada; evidencia para demo.

#### VAL-CLIN-001 — Protocolo de validación por médicos
- Epic: VAL-CLIN
- Repo: Docs
- Prioridad: P1
- Dependencias: QUAL-005,FE-009
- Descripción: Definir qué se muestra a los médicos, qué evalúan y cómo se registra el acuerdo/desacuerdo; encuadre honesto de limitaciones (no diagnóstico, revisión obligatoria).
- Criterios de aceptación: protocolo escrito; instrumento de registro; muestra mínima de casos; limitaciones documentadas.

## Template para Fable/Planner

Usar este formato para planificar cada ticket antes de mandarlo a Codex:

```text
Actuá como planner técnico del PFI. Ticket: <ID>.
Contexto: producto con Frontend React, Backend Spring Boot y AI Module FastAPI/PyTorch.
Objetivo: <resultado esperado>.
Repo afectado: <repo>.
Dependencias cumplidas: <sí/no y evidencia>.
Restricciones:
- no cambiar contratos no relacionados;
- no marcar real_baseline sin ejecución PyTorch real;
- no exponer paths internos ni secrets;
- mantener humanReviewRequired=true.
Entregá:
1. archivos probables a modificar;
2. pasos de implementación;
3. tests a ejecutar;
4. riesgos;
5. prompt exacto para Codex.
```

## Template para Codex/agente de código

```text
Implementá el ticket <ID>.
Repositorio: <repo>.
Objetivo: <objetivo concreto>.
Archivos sugeridos: <lista>.
No hagas refactors no pedidos.
No cambies contratos públicos salvo que el ticket lo pida.
Criterios de aceptación:
- <AC1>
- <AC2>
Tests obligatorios:
- <comandos>
Al finalizar, informá:
1. archivos modificados;
2. tests ejecutados y resultado;
3. limitaciones pendientes;
4. cómo probarlo manualmente.
```

## Regla de avance

No iniciar M3 si M2 no puede demostrar al menos una corrida end-to-end con inputId y assets visibles.
No iniciar demo final si no existe plan B en contract mode y plan A en real_baseline mode.
No declarar 100% producto de tesis si no existen revisión profesional persistida, trazabilidad y evidencia reproducible.
No declarar el AI Module terminado si no supera el umbral de calidad (>=70%) en test held-out (QUAL-005). *(revisión crítica)*
