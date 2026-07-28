# P10.5-A — Contrato volumétrico navegable (canónico, versionado)

Este documento es la fuente única de verdad del contrato acordado entre AI Module, Backend y Frontend para exponer, persistir y navegar el stack completo de cortes de cada `.mha` sagital/axial, sin cambiar la forma de carga actual. Copias idénticas de este archivo viven en `docs/P10_5_A_VOLUMETRIC_CONTRACT.md` de los tres repositorios; esta es la versión canónica (AI Module), porque es el repo que produce la evidencia geométrica original.

**No implementa nada todavía.** Es el cierre de P10.5-A. El siguiente bloque (P10.5-B, AI Module) recién puede empezar cuando este contrato quede aprobado.

---

## 1. Auditoría de los contratos actuales (con evidencia real de código)

### 1.1 AI Module (`ai_service/pfi_ai_service/real_inference_runtime.py`)

Esta es la única fuente real de metadata volumétrica hoy. Nada de lo que sigue es hipotético — es lo que el runtime ya hace:

- `load_input()` (líneas 141-180): para `.mha`/`.mhd` usa `SimpleITK.ReadImage`, obtiene `array = sitk.GetArrayFromImage(image)` (el **volumen completo** en memoria), y captura `spacing = image.GetSpacing()`, `metadata["origin"] = image.GetOrigin()`, `metadata["direction"] = image.GetDirection()`. Es decir: **el volumen completo y su geometría ya se leen siempre**, incluso hoy que solo se usa un corte.
- `canonicalize_loaded_input()` (líneas 224-248): normaliza orientación sagital (`move_axis_0_to_last` cuando aplica), y calcula `spacingXyz`, `arrayAxisSpacingNative`, `arrayAxisSpacingCanonical`, `inputShapeNative`, `inputShapeCanonical`. **`origin` y `direction` NO se propagan a este diccionario** — se pierden después de `load_input()`. Gap real, no hipotético.
- `slice_axis_for()` (líneas 270-283) + `select_slice()` (líneas 291-321): eligen **un solo eje y un solo índice** de corte (ventana ± 3 alrededor del centro para sagital, elegido por score de foreground; centro fijo para axial). Devuelven `(image_2d, selected_slice, slice_count, axis)`. **A partir de acá el resto del volumen (`loaded.array`) nunca se vuelve a tocar** — este es el punto exacto donde hoy se descarta el stack.
- `run_real_inference()` (líneas 485-648) arma la respuesta final. Los campos volumétricos que **ya existen en el JSON de salida** (líneas 617-638): `selectedSlice`, `selectedAxis`, `sliceCount`, `inputShapeNative`, `inputShapeCanonical`, `inputOrientationTransform`, `spacingXyz`, `arrayAxisSpacingNative`, `arrayAxisSpacingCanonical`, `inPlaneSpacing`, `inPlaneSpacingUnit`, `sourceShape`, `processedShape`.
- **Hallazgo de seguridad a tener en cuenta en P10.5-B/C** (línea 641-642): el mismo bloque `metadata` incluye hoy `"sourcePath": str(loaded.path)` (ruta absoluta del `.mha` en disco) y `"outputFiles": outputs` (rutas absolutas de `imagePath`/`maskPath`/`overlayPath`/etc.). Esto **nunca debe reaparecer en el contrato de slices nuevo**, y conviene verificar en P10.5-B que el Backend efectivamente lo filtra hoy antes de reenviarlo al Frontend (no se confirmó en esta auditoría; queda como tarea explícita, no como hecho verificado).
- `asset_registry.py`: `ALLOWED_ASSET_NAMES` (línea 9) es un `frozenset` **fijo** de 6 nombres literales (`input.png`, `mask.npy`, `confidence.npy`, `overlay.png`, `mask-preview.png`, `lumbar-3d-mesh.json`). No admite un patrón tipo `slice-000.png`. Esto tiene que cambiar en P10.5-B (ver §8).

### 1.2 Backend (Java) — auditoría exacta con archivo:línea

- **`dto/AiPlaneInputV2Dto.java:7-20`** (record) ya declara: `inputId`, `format`, `sizeBytes`, `nativeShape`, `canonicalShape`, `orientationTransform`, **`spacingXyzMm`**, **`canonicalAxisSpacingMm`**, `selectedSliceIndex`, `sliceCount`, `selectedAxis`, `inPlaneSpacingMm`. Es decir: **el Backend ya tiene nombres de campo reservados para spacing completo de 3 ejes** (`spacingXyzMm`/`canonicalAxisSpacingMm`), alineados 1:1 con lo que el AI Module ya calcula (`spacingXyz`/`arrayAxisSpacingCanonical`). El contrato nuevo debe reusar exactamente estos nombres, no inventar otros.
- **`dto/AiPlaneAssetV2Dto.java:6-12`** (record): `assetName`, `role`, `contentType`, `generated`, `relativePath`. **No tiene `url`, `sha256` ni `sizeBytes`** — esos se agregan después, en la capa de persistencia/publicación (ver abajo). `relativePath` es lo primero que se elimina antes de publicar nada (`PublicThreeDAssetPublisher.sanitize()`, líneas 42-60, lo quita recursivamente de cualquier mapa/lista anidado).
- **`dto/MultiplanarRunResponseDto.java`**: `PlaneDto.metadata` (línea 81) y `PlaneDto.assets` (línea 80) son **`Map<String,Object>` sin tipar** — no hay un DTO fuerte hoy para el bloque de metadata/asset por plano; es JSON suelto. Esto es una debilidad arquitectónica preexistente (no introducida por P10.5-A), pero significa que agregar un campo `slices` nuevo no rompe nada a nivel de compilación — y también significa que **conviene que el catálogo de slices sí tenga un DTO tipado nuevo**, en vez de perpetuar el patrón de mapa suelto, dado que un catálogo con límites de índice es exactamente el tipo de estructura que se beneficia de validación en tiempo de compilación.
- **`service/MultiplanarRunPersistenceService.java`**: persiste el run en `domain_study_runs` vía `planeSnapshot()` (líneas 238-258), que guarda el bloque `input` (línea 248) tal cual viene del AI Module dentro de la columna JSONB `metrics_snapshot`. **No existe hoy ninguna columna ni estructura relacional para sliceCount/selectedSliceIndex/spacing** — todo vive como JSON suelto dentro de esa columna. Los assets se persisten **uno por `(studyRunId, plane, assetName)`** vía `addArtifacts()` (líneas 136-153), con `isValidAssetMetadata()` (línea 197) exigiendo que `relativePath` contenga literalmente `"/" + plane + "/" + assetName"`, e `isAllowedAsset()` (líneas 200-203) hardcodeado a los mismos 4 nombres fijos que el AI Module.
- **`domain_run_artifacts`** (`docs/postgres_schema.sql:126-140`): `UNIQUE (study_run_id, plane, asset_name)` (línea 139). **Esta es la restricción que importa**: como es única por `(plane, asset_name)` y no fija qué valores puede tomar `asset_name` a nivel de base de datos (el enum vive solo en código Java/Python), **agregar N previews por plano NO requiere migración de esquema** — alcanza con que `asset_name` tome valores como `slice-000.png`, `slice-001.png`, etc., y con relajar los allow-lists hardcodeados (Java y Python) de un enum fijo a un patrón validado.
- **`domain_run_asset_payloads`** (`docs/postgres_schema.sql:142-152`): PK `artifact_id` (1:1 con `domain_run_artifacts`), con `sha256`, `size_bytes` (máx. 5 MB por fila), `storage_kind`, `content_type` ya validado como `image/png` o el JSON del mesh. **Un preview PNG de un slice encaja exactamente en la rama `image/png` ya existente** — no hay que tocar `PostgresRunAssetContentStorage.validatePayloadContract()`.
- **`AiBackendController.java:97-104`**: `GET /api/ai/assets/{runId}/{plane}/{assetName}` ya sirve cualquier asset registrado, autenticado (sin rol extra, solo `AuthFilter` default), con `ETag` calculado desde el sha256 y `Cache-Control: private, max-age=86400` (`AiBackendService.durableAsset()`, líneas 343-351). **Un slice preview servido como `assetName=slice-008.png` usa este mismo endpoint sin cambios de ruta.**
- **`PublicThreeDAssetPublisher.meshUrl()`** (líneas 38-40): confirma el patrón de URL pública durable: `/api/ai/assets/{runId}/{plane}/{assetName}`, siempre relativa al mismo origen, nunca absoluta a un host del AI Module.

**Conclusión de la auditoría Backend**: no hace falta una migración de base de datos para persistir el catálogo de previews. Alcanza con (a) tratar cada preview de slice como un `asset_name` más dentro de la restricción única existente, y (b) guardar el catálogo (índice, `hasResults`, ids de mediciones/landmarks) como JSON dentro de la columna `metrics_snapshot` ya existente, igual que hoy se guarda `input`. Esto es una decisión de diseño para P10.5-C, documentada acá para que no se re-derive desde cero.

### 1.3 Frontend (TypeScript)

- **`src/contracts/canonicalMultiplanarRun.ts`**: `CanonicalPlaneInput` (líneas 43-52) ya tiene `inputId`, `nativeShape`, `canonicalShape`, `orientationTransform`, `selectedSliceIndex`, `sliceCount`, `selectedAxis`, `inPlaneSpacingMm` — coincide con lo que el AI Module y el Backend ya envían. **No tiene `spacingXyzMm`/`canonicalAxisSpacingMm` todavía** — hay que agregarlos para cerrar el círculo de los tres repos con el mismo nombre.
- `CanonicalPlaneAsset` (líneas 21-27): `assetName` es una unión fija `"input.png" | "overlay.png" | "mask-preview.png"` — no admite variantes de slice hoy. `role`, `contentType`, `generated`, `url` sí existen; `sha256`/`sizeBytes` no.
- `src/adapters/multiplanarRunAdapter.ts:386-388`: ya parsea `selectedSliceIndex`, `sliceCount`, `selectedAxis` desde `metadata.selectedSlice`/`metadata.sliceCount`/`metadata.selectedAxis` — el parser real ya sabe leer estos campos.
- **Cuidado con colisión de nombres**: ya existen dos tipos de "serie" con otro propósito, que el contrato nuevo NO debe pisar:
  - `CanonicalPlaneSeriesItem` (`canonicalMultiplanarRun.ts:54-58`) — lista `plane/imageUrl/overlayUrl`, usada hoy por `StudyReviewView.tsx`/`canonicalRunSelectors.ts` para un propósito distinto (comparación de series legacy en modo demo).
  - `StudySeries` (`appTypes.ts:177-193`) y `MriViewerSeries`/`MriViewerModel` (`viewModels/mriViewerViewModel.ts:16-69`) — el dominio visual actual del visor de un solo corte por plano (`selectedSlice`/`sliceCount` singulares, sin catálogo).
  
  El contrato de P10.5-A **extiende `CanonicalPlaneInput` de forma aditiva** (agrega `slices`, no reemplaza nada), precisamente para no chocar con ninguno de estos tres tipos existentes. El futuro visor (P10.5-D, no ahora) construirá un view-model nuevo a partir de `CanonicalPlaneInput.slices` sin tocar `MriViewerSeries`.

---

## 2. Hecho confirmado con fixtures reales

Fuente: `PFI_MVPTest_Enzo_AImodule/backlogProducto/E13_resultados_cierre.md` y `E13_patch_axis_detection.md` (evaluación real, no notebook exploratorio), más `ai_service/tests/fixtures/real_baseline/fixture_summary.json`. Los `.mha` en sí están excluidos del repo (`*.mha` en `.gitignore`, viven en Google Drive/Colab) — estos son los números ya documentados y verificados por una corrida real, no inventados para esta tarea.

| Campo | `101_t1.mha` | `101_t2.mha` |
|---|---|---|
| Rol en el dataset | SPIDER, sujeto 101, secuencia T1 | SPIDER, sujeto 101, secuencia T2 |
| `nativeShape` (shape leído por SimpleITK) | `[298, 320, 17]` | `[352, 384, 17]` |
| `sliceCount` | 17 | 17 |
| `selectedAxis` | 2 (detectado dinámicamente, coincide con `argmin(shape)`) | 2 (fijado por el checkpoint, `sagittal_axis: 2`) |
| `selectedSliceIndex` (0-indexed) | 8 | **No documentado en el repo — no inventar.** Requiere correr el pipeline real para confirmarlo en P10.5-B. |
| `displayIndex` (1-indexed, lo que ve el usuario) | 8 + 1 = **9** → explica exactamente el "9 de 17" que reportás hoy en producción | pendiente |
| Métricas Dice de ejemplo (no clínicas) | vertebra 0.9035, canal 0.9026, disc 0.8296 | no documentado |

**Relación `selectedSliceIndex` → "9/17" confirmada**: el runtime es 0-indexed (`select_slice()` devuelve un índice de array), pero el string que ve el usuario/doctor es 1-indexed. La regla es `displayIndex = selectedSliceIndex + 1`. Esto queda fijado como parte del contrato (§3) para no dejarlo ambiguo.

**Spacing, `origin`, `direction` reales**: SimpleITK los calcula siempre (`image.GetSpacing()/GetOrigin()/GetDirection()`, `real_inference_runtime.py:158-160`) pero no hay un valor numérico documentado en el repo para `101_t1`/`101_t2` específicamente — solo la confirmación de que el campo se calcula. Los fixtures de la §4 marcan `spacingXyzMm` con un valor de ejemplo plausible pero **explícitamente etiquetado como no verificado**, y `originMm`/`directionMatrix` como `null` con `geometryComplete: false`, que es el comportamiento correcto cuando no hay corrida real todavía para confirmarlo.

**Axial — advertencia importante**: el pipeline de evaluación E13 usó un input axial `.npy` 2D (`AXIAL_ALKAFRI/.../pair_0001_image.npy`, shape `[320, 320]`, sin eje de stack) para probar el modelo axial de forma aislada — **no es evidencia de que un `.mha` axial real tenga la misma forma que uno sagital**. El fixture axial de la §4 usa una estructura conforme al contrato pero con valores de ejemplo explícitamente marcados como no verificados contra un archivo real. **Confirmar la forma real de un `.mha` axial multi-corte es la primera tarea de P10.5-B**, antes de generalizar la lógica de `select_slice()` para axial.

---

## 3. Contrato canónico propuesto

### 3.1 Principio de diseño

Extensión **aditiva** de `CanonicalPlaneInput` (ya existe en los tres repos con nombres ya alineados en 8 de los 12 campos). No se crea un tipo paralelo que duplique `sliceCount`/`selectedSliceIndex`/`selectedAxis`/`nativeShape`/`canonicalShape` — eso violaría "reutilizar los campos que ya existen". Un run persistido **sin** el campo `slices` sigue siendo 100% válido (estudio legacy).

### 3.2 `CanonicalPlaneInput` extendido (nuevos campos marcados `NEW`)

```ts
type CanonicalPlaneInput = {
  // --- ya existentes, sin cambios ---
  inputId?: string;
  nativeShape?: number[];
  canonicalShape?: number[];
  orientationTransform?: string;
  selectedSliceIndex?: number;      // 0-indexed
  sliceCount?: number;
  selectedAxis?: number;
  inPlaneSpacingMm?: number[];      // spacing 2D del corte seleccionado (ya existente)

  // --- NEW P10.5-A ---
  seriesId?: string;                // identidad estable del volumen. Hoy === inputId (ver §3.3). Nunca un path.
  sourceFormat?: "mha" | "mhd" | "dcm" | "npy" | "png" | "unknown";
  spacingXyzMm?: number[] | null;   // reusa el nombre YA reservado en AiPlaneInputV2Dto.java:14
  canonicalAxisSpacingMm?: number[] | null; // reusa el nombre YA reservado en AiPlaneInputV2Dto.java:15
  originMm?: number[] | null;       // SimpleITK GetOrigin() crudo. NUNCA DICOM ImagePositionPatient.
  directionMatrix?: number[] | null;// SimpleITK GetDirection() crudo (cosenos de dirección, 4, 6 o 9 valores según ndim).
  geometryComplete?: boolean;       // true solo si spacingXyzMm + originMm + directionMatrix están completos y son consistentes con nativeShape.ndim
  slices?: CanonicalSliceEntry[];   // el catálogo. Ausente = run legacy sin catálogo (ver §7).
};
```

### 3.3 Decisión sobre `seriesId` vs `sourceInputId`

El Excel los pide como campos separados. Se definen así: `seriesId` es la identidad pública/estable del volumen para el Frontend y para vincular resultados; `sourceInputId` (alias de `inputId`, ya existente) es la identidad interna asignada por `POST /api/ai/inputs`. **Hoy son el mismo valor** (`seriesId === inputId`), porque el flujo actual sube un input opaco por corrida y no hay reuso de un mismo volumen entre corridas. Se dejan como dos campos (no uno solo) para no tener que romper el contrato el día que eso cambie (por ejemplo, si a futuro un mismo volumen se reutiliza en una corrida nueva sin resubir el archivo). **No usar ningún path como identidad, nunca** (ya es una regla existente del proyecto, confirmada también en `isDurableMeshAssetUrl`/`isAuthorizedBackendUrl` del Frontend).

### 3.4 `CanonicalSliceEntry` (tipo nuevo)

```ts
type CanonicalSliceEntry = {
  index: number;                       // 0-indexed, 0..sliceCount-1
  displayIndex: number;                // 1-indexed = index + 1. Nunca se recalcula distinto en ningún repo.
  previewAsset?: CanonicalSliceAsset;  // ausente = preview no generado/no disponible todavía
  hasResults: boolean;                 // true solo en el/los índices donde el modelo realmente infirió
  overlayAsset?: CanonicalSliceAsset;  // presente solo si hasResults === true
  measurementIds?: string[];           // subconjunto de CanonicalMeasurement.id cuya evidencia viene de este slice
  landmarkIds?: string[];              // subconjunto de CanonicalLandmark.id
};
```

### 3.5 `CanonicalSliceAsset` (reusa la forma de `CanonicalPlaneAsset`, con `assetName` de patrón en vez de enum fijo)

```ts
type CanonicalSliceAssetName = string; // valida contra el patrón ^slice-\d{3}(-overlay)?\.png$ — NO es CanonicalAssetName (esa unión sigue fija para input.png/overlay.png/mask-preview.png del corte único legacy)

type CanonicalSliceAsset = {
  assetName: CanonicalSliceAssetName; // ej. "slice-008.png", "slice-008-overlay.png"
  role: "slice-preview" | "slice-overlay";
  contentType: "image/png";
  generated: true;
  url: string;        // SIEMPRE /api/ai/assets/{runId}/{plane}/{assetName} — mismo patrón que PublicThreeDAssetPublisher.meshUrl()
  sha256?: string;     // ya se calcula hoy en RunAssetSnapshotService.sha256() a nivel de persistencia; agregar al JSON público es una decisión de P10.5-C, no de este documento
  sizeBytes?: number;  // ya existe como size_bytes en domain_run_asset_payloads
};
```

### 3.6 Qué el contrato público NUNCA expone (regla dura, no negociable)

Confirmado como riesgo real en §1.1, no hipotético:

- `sourcePath`, `outputFiles`, `inputPath`/`input_path`, o cualquier ruta absoluta del AI Module.
- `/tmp`, `/app`, `/content`, cualquier path de disco Windows o Linux.
- Host interno del AI Module (Cloudflare Quick Tunnel, IP, `localhost`, `.internal`).
- `relativePath` — se elimina siempre antes de publicar (ya es la convención existente, `PublicThreeDAssetPublisher.sanitize()`).
- `StudyInstanceUID`, `SeriesInstanceUID`, `SOPInstanceUID`, `FrameOfReferenceUID`, `ImagePositionPatient` — campos DICOM que **no existen** en los `.mha` de SPIDER; no se fabrican. Si en el futuro el input es DICOM real (INGEST-001, fuera de alcance), se agregan como campos opcionales nuevos, nunca reusando `originMm`/`directionMatrix` para simular lo que no son.

### 3.7 Metadata que SÍ se conserva (solo lo realmente disponible en `.mha`)

`nativeShape`/`canonicalShape` (shape), `spacingXyzMm`/`canonicalAxisSpacingMm` (spacing), `originMm` (si SimpleITK lo devuelve un valor no trivial — un `.mha` sin origin explícito puede devolver `(0,0,0)`, que se considera "no informado" y se marca `null`, no `[0,0,0]` engañoso), `directionMatrix` (si existe), `selectedAxis`, `sliceCount`. Nada más.

---

## 4. Fixtures (8 escenarios, `docs/fixtures/p10-5-a/`)

Todos basados en los números reales de §2 donde existen, y explícitamente marcados `"verified": false` en los campos que no tienen evidencia documentada (regla: nunca fabricar un número y presentarlo como real).

1. `01-sagittal-valid-101-t1.json` — serie sagital válida, 101_t1.mha, 17 slices, `hasResults` solo en índice 8.
2. `02-axial-valid-example.json` — serie axial válida, **estructuralmente conforme pero con `"verified": false`** en shape/spacing (ver advertencia §2).
3. `03-legacy-single-slice.json` — run histórico pre-P10.5, sin campo `slices` (ni `seriesId` ni geometría extendida). Debe seguir abriendo con el visor actual.
4. `04-full-previews.json` — 17/17 slices con `previewAsset` generado, 1 con `overlayAsset`.
5. `05-overlay-only-inferred-slice.json` — todas las entradas con `previewAsset`, solo el índice 8 con `hasResults:true` + `overlayAsset`; el resto `hasResults:false` sin `overlayAsset`.
6. `06-missing-preview.json` — el índice 3 aparece en el catálogo pero sin `previewAsset` (falla parcial de generación) — el Frontend debe mostrar "vista previa no disponible", nunca romper.
7. `07-index-out-of-range.json` — una entrada con `index: 17` cuando `sliceCount: 17` (rango válido es 0-16) — caso que la validación (§5) debe rechazar.
8. `08-incompatible-metadata.json` — `selectedAxis: 5` para un array de 3 dimensiones (rango válido 0-2) y `sliceCount: 0` — caso que el AI Module debe rechazar antes de emitir el contrato, no solo el Frontend.

(Los 8 archivos JSON se generan como parte de esta entrega, ver §11.)

---

## 5. Esquema de validación

Reglas que cualquier consumidor (Backend al persistir, Frontend al parsear) debe aplicar, en este orden:

1. Si `slices` está ausente → tratar como legacy (§7), no validar nada más de este bloque.
2. `sliceCount` debe ser un entero > 0. Si es 0 o negativo → rechazar el catálogo completo (no el run entero; el run sigue siendo válido sin catálogo, degradado a legacy).
3. `selectedAxis` debe estar en `[0, nativeShape.length)`. Fuera de rango → rechazar el catálogo.
4. `selectedSliceIndex` debe estar en `[0, sliceCount)`.
5. Cada `CanonicalSliceEntry.index` debe ser único, entero, y estar en `[0, sliceCount)`. Un índice fuera de rango invalida esa entrada específica (se descarta, no se rompe el catálogo completo) — mismo criterio defensivo que ya usa `isDurableMeshAssetUrl`/`parseThreeD` en el Frontend hoy.
6. `displayIndex` debe ser exactamente `index + 1`. Si no coincide, se recalcula del lado del consumidor y se ignora el valor recibido (nunca se confía ciegamente en un valor derivado que debería ser determinístico).
7. `hasResults: true` sin `overlayAsset` es válido (el overlay puede no haberse generado todavía) — no es lo mismo que `hasResults: false`.
8. `overlayAsset` presente con `hasResults: false` es **inválido** — nunca debe existir un overlay en un slice donde no hubo inferencia. Si ocurre, se descarta el `overlayAsset` y se loguea como anomalía de contrato (mismo patrón que `ContractError` ya usado en el Frontend).
9. Toda URL de asset (`previewAsset.url`, `overlayAsset.url`) pasa por la misma política de origen ya construida en P10-C.1 (`isAuthorizedBackendUrl`/`isDurableMeshAssetUrl`): solo `/api/...` relativo o el origin exacto de `API_BASE_URL`. Cualquier otra cosa se descarta antes de intentar el fetch, nunca después.

---

## 6. Estados de error

| Estado | Cuándo ocurre | Comportamiento requerido |
|---|---|---|
| `legacy_series` | `slices` ausente | Visor de corte único actual (sin cambios); nunca fabricar un catálogo falso. |
| `series_incomplete` | `slices.length < sliceCount` | El catálogo es parcial (generación en curso o falló para algunos índices); mostrar los índices disponibles, marcar los faltantes como "no disponible", nunca bloquear la navegación completa. |
| `preview_missing` | Una entrada existe pero `previewAsset` es `undefined`/`null` | Slot vacío con mensaje explícito ("vista previa no disponible"), nunca placeholder inventado ni imagen de otro slice. |
| `slice_out_of_range` | Se pide un índice fuera de `[0, sliceCount)` | Rechazar la solicitud (400 en el endpoint del Backend que se defina en P10.5-C), nunca hacer clamp silencioso que oculte el error. |
| `geometry_incomplete` | `spacingXyzMm`/`originMm`/`directionMatrix` incompletos o `geometryComplete: false` | Bloquea únicamente features que dependan de geometría real (coordinación espacial futura, VOL-COORD) — nunca bloquea la navegación básica de slices, que no depende de esto. |
| `metadata_incompatible` | `selectedAxis` fuera de rango, `sliceCount <= 0`, shape con ndim fuera de `{2,3}` | El AI Module rechaza generar el catálogo antes de emitirlo (falla temprano, no propaga metadata corrupta); el Backend nunca debe intentar "arreglar" un contrato incompatible. |

---

## 7. Estrategia de compatibilidad legacy

- Un run persistido **antes** de P10.5-B/C no tiene `slices`, `seriesId`, `spacingXyzMm`, etc. — todos esos campos son opcionales (`?`), por lo que el parser existente (`multiplanarRunAdapter.ts`) sigue funcionando sin cambios sobre datos viejos.
- **Nunca se fabrica un catálogo retroactivo** para una corrida histórica: si no fue persistido en su momento, no existe. El visor legacy de corte único sigue siendo la única forma de ver esos estudios.
- Estudios solo-sagitales (sin plano axial) siguen siendo válidos — el campo `axial` de `CanonicalMultiplanarRun.planes` ya es opcional, sin cambios.
- `threeD` sigue siendo opcional e independiente — el proxy 3D experimental no se toca en este contrato.
- `selectedSliceIndex`/`sliceCount` ya existentes **no cambian de significado**: siguen representando el slice que efectivamente usó la inferencia, ahora simplemente también aparecen replicados como una entrada dentro de `slices[]` con `hasResults: true`.

---

## 8. Responsabilidad exacta por repositorio

### AI Module (P10.5-B — siguiente bloque, no ahora)
- `ai_service/pfi_ai_service/real_inference_runtime.py`: `run_real_inference()` (línea 485) debe generar un preview PNG por cada índice de `loaded.array` a lo largo de `selected_axis` (reusando `resize_image()`/`robust_percentile_normalize()`, ya existentes), no solo del `selected_slice`. `load_input()`/`canonicalize_loaded_input()` deben propagar `origin`/`direction` hasta el diccionario final (hoy se calculan y se pierden, §1.1). Eliminar `sourcePath`/`outputFiles` de cualquier bloque que pueda llegar sin filtrar al Backend, o confirmar que el Backend ya los filtra (a verificar, no asumir).
- `ai_service/pfi_ai_service/asset_registry.py`: `ALLOWED_ASSET_NAMES`/`PUBLIC_BROWSER_ASSET_NAMES` (líneas 9-11) pasan de frozenset fijo a validación por patrón (`^slice-\d{3}(-overlay)?\.png$` además de los 6 nombres legacy).
- Primera tarea real de P10.5-B: correr el pipeline sobre un `.mha` axial real y documentar su shape/spacing real (hoy no existe esa evidencia, §2).

### Backend (P10.5-C)
- `dto/AiPlaneInputV2Dto.java`: agregar `seriesId`, `sourceFormat`, `originMm`, `directionMatrix`, `geometryComplete`, `slices` (nuevo tipo `List<AiSliceEntryV2Dto>`).
- Nuevo `dto/AiSliceEntryV2Dto.java` y `dto/AiSliceAssetV2Dto.java` (tipados, no `Map<String,Object>` — ver razón en §1.2).
- `service/MultiplanarRunPersistenceService.java`: `isAllowedAsset()` (líneas 200-203) pasa de enum fijo a patrón; `isValidAssetMetadata()` (línea 197) ajusta el chequeo de `relativePath` para aceptar el patrón de slice. El catálogo se persiste dentro de `metrics_snapshot` JSONB (sin migración de esquema, confirmado en §1.2).
- `service/RunAssetSnapshotService.java`: `PUBLIC_ASSETS` (línea 24) pasa de set fijo a patrón; `isPublicAsset()` (línea 136) ya acepta cualquier nombre sin `/`/`\`/`..`, solo falta ampliar el allow-list.
- `service/PostgresRunAssetContentStorage.java`: **sin cambios** — un preview PNG ya encaja en la rama `image/png` existente de `validatePayloadContract()`.
- `AiBackendController.java`: **sin cambios de ruta** — `GET /api/ai/assets/{runId}/{plane}/{assetName}` ya sirve cualquier asset registrado; un slice preview es solo un `assetName` más.
- Confirmar explícitamente (tarea, no supuesto) que `MultiplanarRunResponsePresenter.java`/`CanonicalMultiplanarRunLegacyPresenter.java` no reenvían `sourcePath`/`outputFiles` si el AI Module llegara a emitirlos sin filtrar.

### Frontend (P10.5-D)
- `src/contracts/canonicalMultiplanarRun.ts`: extender `CanonicalPlaneInput` y `CanonicalPlaneAsset` según §3.2/§3.5; agregar `CanonicalSliceEntry`/`CanonicalSliceAsset`.
- `src/adapters/multiplanarRunAdapter.ts`: agregar `parseSlices()` siguiendo el mismo patrón defensivo que `parseThreeD()`/`isDurableMeshAssetUrl()` ya usan (descartar entradas inválidas, nunca fabricar).
- Nuevo view-model (no toca `mriViewerViewModel.ts` actual) para el futuro visor de stack — se diseña en P10.5-D, no en este documento.
- No se toca `AnalysisTimelineView.tsx` (la carga sigue siendo un `.mha` por plano) ni `StudyReviewView.tsx` en este bloque.

---

## 9. Criterios de aceptación E2E (para P10.5-F, no ahora)

- Subir `101_t1.mha` como sagital → la respuesta incluye `slices.length === sliceCount` y exactamente una entrada con `hasResults: true` en `index === selectedSliceIndex`.
- El `displayIndex` de esa entrada es siempre `selectedSliceIndex + 1`.
- Reabrir el estudio con el AI Module apagado devuelve el mismo catálogo completo (mismos índices, mismos `previewAsset.url`, misma revisión) — sin ninguna llamada al AI Module.
- Un estudio persistido antes de P10.5 se sigue abriendo (modo legacy, sin catálogo, sin error).
- Ninguna URL de `previewAsset`/`overlayAsset` apunta a un host distinto del backend configurado (mismo test que ya existe para `isDurableMeshAssetUrl`, extendido a los nuevos nombres de asset).

---

## 10. Fuera de alcance de este documento (recordatorio)

Visor React completo, endpoints HTTP definitivos del Backend (`GET /api/studies/{caseId}/series/{seriesId}/slices/{index}` queda como propuesta conceptual, no atada), persistencia real de los 34 previews de un caso de ejemplo, inferencia sobre todos los cortes, crosshair sagital-axial, DICOM real, comparación longitudinal, scoring clínico. Todo eso es P10.5-B en adelante o epics posteriores del roadmap.

---

## 11. Entrega

- Este documento (copiado en los 3 repos).
- 8 fixtures JSON en `docs/fixtures/p10-5-a/` (AI Module), copiados a los otros dos repos.
- Actualización de `ESTADO_Y_ROADMAP_100.md` (AI Module), `MVP_PROGRESS.md` (Frontend), y un nuevo doc equivalente en Backend — ver commit propuesto.
