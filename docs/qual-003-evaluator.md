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

Las mascaras deben contener labels enteros con el mismo mapping de clases usado durante entrenamiento. El fondo debe ser label `0`.

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
  --output docs\qual-003-eval-report-axial.json
```

En Colab/Linux, usar `python scripts/evaluate_model.py` con rutas montadas de Drive, por ejemplo `/content/drive/MyDrive/...`. No copiar datasets ni checkpoints al repo.

## Argumentos

- `--plane {sagittal|axial}`: selecciona arquitectura/model key.
- `--checkpoint`: path local al `.pt` materializado fuera de Git.
- `--test-dir`: directorio held-out con `images/` y `masks/`.
- `--num-classes`: cantidad total de clases incluyendo fondo.
- `--target-size`: `N` o `H,W`; default `256`.
- `--output`: JSON de salida; default `docs/qual-003-eval-report.json`.

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
- `diceMacroForeground` e `iouMacroForeground` promedian solo clases foreground (`1..num_classes-1`) con metrica definida.

## Evidencia de tests unitarios

Comando ejecutado:

```powershell
$env:PYTHONPATH="ai_service"
.\.venv\Scripts\python.exe -m pytest ai_service\tests\test_qual003_metrics.py -q
```

Resultado:

```text
....                                                                     [100%]
4 passed in 2.51s
```

Los tests cubren solapamiento perfecto, disjunto, parcial conocido y clase ausente en GT+pred.

## Nota de reproducibilidad

El reporte JSON incluye `nCases`, metricas por clase, macro foreground, clases ausentes y metadatos por caso sin paths internos de salida ni datos identificables. El checkpoint se reporta por nombre de archivo, no por contenido ni path absoluto.