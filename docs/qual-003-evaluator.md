# QUAL-003 - Evaluador reproducible por clase (Dice/IoU)

## Objetivo

`scripts/evaluate_model.py` evalua un checkpoint real contra un held-out deidentificado externo al repositorio y reporta Dice e IoU por clase, mas macro foreground excluyendo fondo (`class_id=0`).

No genera diagnosticos ni recomendaciones clinicas. Solo mide solapamiento geometrico entre mascara predicha y mascara ground-truth.

## Layout esperado del held-out

El directorio pasado por `--test-dir` debe contener:

```text
heldout/
  images/
    case_001.npy
    case_002.npy
  masks/
    case_001.npy
    case_002.npy
```

Tambien se acepta que la mascara use sufijo `_mask`, por ejemplo `masks/case_001_mask.npy`.

Formatos soportados:

- Imagenes: `.npy`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.mha`, `.mhd`, `.dcm`.
- Mascaras GT: `.npy`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`.

Las mascaras deben contener labels enteros. Si el checkpoint incluye `label_group_mapping`, el evaluador remapea la GT cruda al espacio de clases del modelo antes de puntuar. Si el checkpoint no incluye mapping y la GT usa labels crudos fuera de `0..num_classes-1`, se debe pasar `--label-map`; si no, el evaluador frena con error claro. El fondo debe ser label `0`.

## Comando

Sagital:

```powershell
$env:PYTHONPATH="ai_service"
.\.venv\Scripts\python.exe scripts\evaluate_model.py `
  --plane sagittal `
  --checkpoint models\final\sagittal_spider_multiclass_final_best.pt `
  --test-dir D:\heldout_sagittal `
  --num-classes 4 `
  --target-size 256 `
  --output docs\qual-003-eval-report.json
```

Axial:

```powershell
$env:PYTHONPATH="ai_service"
.\.venv\Scripts\python.exe scripts\evaluate_model.py `
  --plane axial `
  --checkpoint models\final\axial_t2_alkafri_final_best.pt `
  --test-dir D:\heldout_axial `
  --num-classes 6 `
  --target-size 256 `
  --label-map D:\heldout_axial\label_map.json `
  --output docs\qual-003-eval-report-axial.json
```

En Colab/Linux, usar `python scripts/evaluate_model.py` con rutas montadas de Drive, por ejemplo `/content/drive/MyDrive/...`. No copiar datasets ni checkpoints al repo.

## Argumentos

- `--plane {sagittal|axial}`: selecciona arquitectura/model key.
- `--checkpoint`: path local al `.pt` materializado fuera de Git.
- `--test-dir`: directorio held-out con `images/` y `masks/`.
- `--num-classes`: cantidad total de clases incluyendo fondo.
- `--target-size`: `N` o `H,W`; default `256`.
- `--label-map`: JSON opcional para checkpoints sin mapping embebido, como axial si la GT no esta ya en espacio de clases del modelo.
- `--output`: JSON de salida; default `docs/qual-003-eval-report.json`.


## Mapping de labels GT

El checkpoint sagital final contiene `label_group_mapping`. El formato embebido confirmado por los notebooks es:

```json
{
  "raw_label": class_id
}
```

Ejemplo conceptual: `{ "10": 1, "20": 2 }` transforma pixeles GT con label crudo `10` a clase del modelo `1`, y label crudo `20` a clase del modelo `2`.

Para JSON externo con `--label-map`, tambien se acepta el formato inverso legible:

```json
{
  "1": [10, 11],
  "2": [20]
}
```

En ambos casos, los valores finales deben quedar en `0..num_classes-1`. El label `0` se conserva como fondo. Si aparecen labels GT sin mapping, el evaluador frena para evitar puntuar basura.

## Preprocesamiento y prediccion

El script reutiliza el runtime real:

- `load_input(...)` para cargar `.npy`, imagenes, MHA/MHD o DICOM.
- `select_slice(...)` y `resize_image(...)` para seleccionar/preprocesar la imagen igual que inferencia real.
- `build_checkpoint_model(...)` para cargar la arquitectura strict desde checkpoint.

Las mascaras GT se redimensionan al `target-size` con nearest-neighbor para conservar labels discretos.

## Metricas

Para cada clase se acumulan interseccion, union, pixeles predichos y pixeles GT sobre todos los casos.

- Dice: `2 * intersection / (pred_pixels + gt_pixels)`.
- IoU: `intersection / union`.
- Si una clase esta ausente en GT y prediccion, Dice/IoU quedan `null` y la clase se anota como excluida.
- El reporte incluye `classStats` por clase con `gt_present_cases`, `pred_present_cases`, pixeles GT/pred, interseccion y union.
- Si una clase foreground tiene `gt_present_cases=0`, el reporte incluye un `WARNING` prominente y `macroForegroundReliable=false`.
- `diceMacroForeground` e `iouMacroForeground` promedian solo clases foreground (`1..num_classes-1`) con metrica definida; si hay warnings de GT vacia, tratar el macro como no confiable hasta revisar mapping/dataset.

## Evidencia de tests unitarios

Comando ejecutado:

```powershell
$env:PYTHONPATH="ai_service"
.\.venv\Scripts\python.exe -m pytest ai_service\tests\test_qual003_metrics.py -q
```

Resultado:

```text
....                                                                     [100%]
7 passed in 7.81s
```

Los tests cubren solapamiento perfecto, disjunto, parcial conocido, clase ausente en GT+pred, remapeo de GT cruda con `label_group_mapping`, error por labels fuera de rango sin mapping y warning por clase foreground sin GT.

## Nota de reproducibilidad

El reporte JSON incluye `nCases`, metricas por clase, macro foreground, clases ausentes y metadatos por caso sin paths internos de salida ni datos identificables. El checkpoint se reporta por nombre de archivo, no por contenido ni path absoluto.
