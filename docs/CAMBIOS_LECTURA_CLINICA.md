# AI Module — cambios para la estación de lectura

Fecha: 2026-07-30/31. Rama: `feat/study-zip-ingestion`.

Todo lo de acá salió de una misma tanda de trabajo: hacer que el módulo entregue lo
que una estación de lectura necesita para mostrar un estudio, y no solo el corte que
la IA analizó.

---

## 1. Los PNG se generan en resolución nativa

**Archivo:** `ai_service/pfi_ai_service/real_inference_runtime.py`

**Qué pasaba.** `save_outputs` guardaba como `input.png` la misma imagen que se le
daba de comer al modelo, es decir el corte ya redimensionado a `targetSize`
(256×256). El frontend dimensiona el marco del visor con `naturalWidth/Height`, así
que el médico veía el estudio a 256 px y con la relación de aspecto deformada.

**Qué se hizo.** El resize a 256×256 sigue existiendo porque es la entrada que el
checkpoint espera, pero ya no es lo que se guarda. Se agregó `native_slice()`, que
devuelve el corte en su resolución original, y `save_outputs` recibe ese arreglo
como `render_image`: de ahí salen `input.png`, `overlay.png` y `mask-preview.png`.

La máscara se remuestrea a la resolución nativa con `upsample_labels()`, por vecino
más cercano, para que superponga exactamente sobre esos píxeles sin inventar clases
intermedias. `mask.npy` y `confidence.npy` **no** cambian: siguen en la grilla del
modelo, que es donde se midieron.

**Resultado medido** con `101_t2.mha`: los PNG pasaron de 256×256 a 384×352.

---

## 2. Catálogo de previsualizaciones por corte

**Archivos:** `real_inference_runtime.py`, `asset_registry.py`

**Qué pasaba.** Solo el corte inferido tenía imagen persistida. El visor mostraba
"17 cortes" en la barra de stack pero un único cuadro con contenido.

**Qué se hizo.** `save_slice_previews()` escribe un PNG por corte de la serie
(`slice-000.png` … `slice-016.png`), en resolución nativa, y los registra como
assets.

Decisiones que vale la pena dejar escritas:

- **Sin superposición.** Cada preview es la imagen sola. La segmentación existe
  únicamente para el corte inferido; pintarla sobre los demás mostraría una máscara
  que no les corresponde.
- **Techo de 512 cortes** (`MAX_SLICE_PREVIEWS`). Una serie más larga se deja sin
  catálogo, no recortada: persistir un subconjunto haría que unos cortes tengan
  imagen y otros no, sin explicación.
- **Fuera de la lista de assets del contrato.** Quedan registrados para poder
  servirlos, pero no se publican en `assets`: serían cientos de entradas repitiendo
  el patrón `slice-NNN.png`. En su lugar se informa `quality.slicePreviewCount`, que
  es lo que el visor necesita para distinguir "este corte no tiene imagen" de "sí la
  tiene" sin tener que pedir el PNG y tratar el 404 como respuesta.

**Nombre de asset por patrón.** Es el único que no puede vivir en una lista fija,
porque la cantidad de cortes depende del estudio. El patrón
(`^slice-\d{3,5}\.png$`) es tan estricto como la lista —solo dígitos, extensión
fija, sin separadores de ruta— así que no amplía la superficie de path traversal que
la lista ya cerraba.

---

## 3. Niveles lumbares por disco (L1-L2 … L5-S1)

**Archivo:** `real_inference_runtime.py`

**Qué pasaba.** El modelo `sagittal_spider` segmenta `vertebra_group`, `canal` y
`disc_group` — bloques fusionados, no instancias. No había nada a lo que llamar
"L4", y `build_measurements` ponía el nombre del plano en el campo `level`.

**Qué se hizo.**

- `connected_instances()` separa la máscara `disc_group` en sus componentes conexas
  (vía SimpleITK) y las ordena de superior a inferior. En el arreglo sagital
  canónico la fila 0 es superior, así que basta ordenar por centroide de fila.
- `lumbar_disc_levels()` **cuenta desde abajo**, que es como se numera una lumbar en
  la práctica: el espacio discal más inferior es L5-S1 y desde ahí se sube L4-L5,
  L3-L4, L2-L3, L1-L2. Los espacios por encima del quinto quedan sin nivel: son
  T12-L1 o más altos, fuera de la nomenclatura lumbar.
- El supuesto es que el encuadre llega a la unión lumbosacra, que es lo que define a
  un protocolo de RM lumbar y el revisor puede verificar de un vistazo. **Si el
  estudio muestra menos de cinco espacios discales**, el encuadre no es una lumbar
  completa: no se sabe desde dónde contar y todas las mediciones quedan sin nivel,
  antes que desplazar la numeración entera.

**Mediciones por instancia.** `build_measurements` ahora emite un juego de
área/ancho/alto **por disco** en vez de uno para el grupo. "Altura del disco L4-L5"
es un hallazgo reportable; "altura del grupo de discos" no significa nada clínico.
`canal` y `vertebra_group` siguen como medición de grupo, que es lo que son.

Un disco sin nivel confirmado se emite sin `linkedLandmarks`: el landmark del grupo
no identifica a ese disco en particular, y apuntar a él sería señalar un punto que
no le corresponde.

---

## 4. Campos que el contrato v2 estaba descartando

**Archivos:** `multiplanar_v2_models.py`, `multiplanar_v2_executor.py`

Tres campos se calculaban bien en el runtime y se perdían al construir la respuesta
v2, porque el ejecutor los fijaba a mano o el modelo no los declaraba:

| Campo | Qué pasaba | Qué se hizo |
|---|---|---|
| `measurement.level` | `plane_measurement_v2` ponía `level=None` fijo | Se lee del runtime con `text_or_none()` |
| `measurement.sliceIndex` | No existía en `PlaneMeasurementV2` | Campo nuevo; el runtime lo llena con el corte inferido |
| `quality.slicePreviewCount` | No existía en `PlaneQualityV2` | Campo nuevo, con default 0 para corridas anteriores |

Sin `sliceIndex` una medición no se puede ubicar en la serie: el revisor ve el número
pero no qué imagen lo produjo.

---

## Verificación

Corrida real con `101_t2.mha` (17 cortes) a través del stack completo:

- PNG de entrada y overlay en 384×352.
- 17 previsualizaciones escritas; `slicePreviewCount: 17`.
- 24 mediciones: 3 de canal, 3 de grupo vertebral y 3 por cada uno de los 6 discos
  detectados. Los 5 inferiores con nivel L1-L2 … L5-S1; el sexto (superior, T12-L1)
  sin nivel, como corresponde.
- `sliceIndex: 7` en todas.

El módulo no tiene `pytest` instalado en su imagen, así que la verificación fue
end-to-end contra el stack levantado y por inspección directa de los artefactos en
`/app/outputs`.

---

## Lo que sigue pendiente

- **`origin` y `direction` de paciente** (VOL-CONTRACT-04). `read_dicom_series` los
  captura en metadata pero `PlaneInputV2` no los publica, así que el frontend no
  puede dibujar la línea de referencia entre sagital y axial. Propagarlos es
  mecánico; verificarlos requiere un estudio con ambos planos reales.
- **Máscaras por clase.** Hoy el overlay es un PNG compuesto, por eso los controles
  de visibilidad por clase (vértebra / canal / disco) están deshabilitados en el
  visor.
- **Semántica del modelo axial.** Sus clases son `raw_0`, `raw_50`, `raw_100`… El
  checkpoint no informa qué segmenta. No se arregla desde el código: hay que mapear
  contra el dataset original o reentrenar.
