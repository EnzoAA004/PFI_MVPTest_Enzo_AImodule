# AI-001 - Checkpoint inspection evidence

Repo de trabajo: PFI_MVPTest_Enzo_AImodule
Ruta de trabajo: C:\Users\enzoa\OneDrive\Documentos\1.ReposGitHub-Backends\PFI_MVPTest_Enzo_AImodule
Interprete: .venv\Scripts\python.exe

Precheck de artifacts:
```text
sagittal_spider_multiclass_final_best.pt: 1975947 bytes
axial_t2_alkafri_final_best.pt: 1973243 bytes
```

Git hygiene:
```text
.pt correctamente gitignoreados: si
.gitignore:37:*.pt	models/final/sagittal_spider_multiclass_final_best.pt
.gitignore:37:*.pt	models/final/axial_t2_alkafri_final_best.pt
pt trackeados en src/lumbar_mri/models: no
```

Comando ejecutado:
```text
$env:PYTHONPATH="ai_service"
.venv\Scripts\python.exe scripts\inspect_final_checkpoints.py
```

Salida real:
```text
# AI-001 checkpoint inspection

## sagittal_spider
- Nombre / path relativo: sagittal_spider_multiclass_final_best.pt / sagittal_spider_multiclass_final_best.pt
- Fuente de path: PFI_MODEL_DIR/settings
- Existe: si | Tamano (bytes): 1975947
- SHA-256: 7dd393cc750311c98003516d8110136310c31e8b6f0f00b6815f949fd61ef15b | vs manifest: SIN_MANIFEST
- Detalle manifest: sha256 esperado AUSENTE en manifest
- Keys top-level del checkpoint: ["base_channels", "class_weights", "epoch", "label_group_mapping", "model_state_dict", "num_classes", "sagittal_axis", "slice_strategy", "target_size", "val_dice_macro_no_bg"]
- State_dict source: model_state_dict
- Keys del state_dict (muestra): ["bottleneck.block.0.bias", "bottleneck.block.0.weight", "bottleneck.block.1.bias", "bottleneck.block.1.num_batches_tracked", "bottleneck.block.1.running_mean", "bottleneck.block.1.running_var", "bottleneck.block.1.weight", "bottleneck.block.3.bias", "bottleneck.block.3.weight", "bottleneck.block.4.bias", "bottleneck.block.4.num_batches_tracked", "bottleneck.block.4.running_mean", "bottleneck.block.4.running_var", "bottleneck.block.4.weight", "dec1.block.0.bias", "dec1.block.0.weight", "dec1.block.1.bias", "dec1.block.1.num_batches_tracked", "dec1.block.1.running_mean", "dec1.block.1.running_var", "dec1.block.1.weight", "dec1.block.3.bias", "dec1.block.3.weight", "dec1.block.4.bias", "... (82 mas)"] | total=106
- num_classes: 4 (STORED desde num_classes)
- base_channels: 16 (STORED desde base_channels)
- target_size: (256, 256) (STORED desde target_size)

## axial_t2_alkafri
- Nombre / path relativo: axial_t2_alkafri_final_best.pt / axial_t2_alkafri_final_best.pt
- Fuente de path: PFI_MODEL_DIR/settings
- Existe: si | Tamano (bytes): 1973243
- SHA-256: a8b91f563e101c9c4a3cf2c0ec84af29cbcfabd76eb76b2d06dbd13f7d7c78d6 | vs manifest: SIN_MANIFEST
- Detalle manifest: sha256 esperado AUSENTE en manifest
- Keys top-level del checkpoint: ["d1.net.0.bias", "d1.net.0.weight", "d1.net.1.bias", "d1.net.1.num_batches_tracked", "d1.net.1.running_mean", "d1.net.1.running_var", "d1.net.1.weight", "d1.net.3.bias", "d1.net.3.weight", "d1.net.4.bias", "d1.net.4.num_batches_tracked", "d1.net.4.running_mean", "d1.net.4.running_var", "d1.net.4.weight", "d2.net.0.bias", "d2.net.0.weight", "d2.net.1.bias", "d2.net.1.num_batches_tracked", "d2.net.1.running_mean", "d2.net.1.running_var", "d2.net.1.weight", "d2.net.3.bias", "d2.net.3.weight", "d2.net.4.bias", "d2.net.4.num_batches_tracked", "d2.net.4.running_mean", "d2.net.4.running_var", "d2.net.4.weight", "d3.net.0.bias", "d3.net.0.weight", "d3.net.1.bias", "d3.net.1.num_batches_tracked", "d3.net.1.running_mean", "d3.net.1.running_var", "d3.net.1.weight", "d3.net.3.bias", "d3.net.3.weight", "d3.net.4.bias", "d3.net.4.num_batches_tracked", "d3.net.4.running_mean", "d3.net.4.running_var", "d3.net.4.weight", "e1.net.0.bias", "e1.net.0.weight", "e1.net.1.bias", "e1.net.1.num_batches_tracked", "e1.net.1.running_mean", "e1.net.1.running_var", "e1.net.1.weight", "e1.net.3.bias", "e1.net.3.weight", "e1.net.4.bias", "e1.net.4.num_batches_tracked", "e1.net.4.running_mean", "e1.net.4.running_var", "e1.net.4.weight", "e2.net.0.bias", "e2.net.0.weight", "e2.net.1.bias", "e2.net.1.num_batches_tracked", "e2.net.1.running_mean", "e2.net.1.running_var", "e2.net.1.weight", "e2.net.3.bias", "e2.net.3.weight", "e2.net.4.bias", "e2.net.4.num_batches_tracked", "e2.net.4.running_mean", "e2.net.4.running_var", "e2.net.4.weight", "e3.net.0.bias", "e3.net.0.weight", "e3.net.1.bias", "e3.net.1.num_batches_tracked", "e3.net.1.running_mean", "e3.net.1.running_var", "e3.net.1.weight", "e3.net.3.bias", "e3.net.3.weight", "e3.net.4.bias", "e3.net.4.num_batches_tracked", "e3.net.4.running_mean", "e3.net.4.running_var", "e3.net.4.weight", "mid.net.0.bias", "mid.net.0.weight", "mid.net.1.bias", "mid.net.1.num_batches_tracked", "mid.net.1.running_mean", "mid.net.1.running_var", "mid.net.1.weight", "mid.net.3.bias", "mid.net.3.weight", "mid.net.4.bias", "mid.net.4.num_batches_tracked", "mid.net.4.running_mean", "mid.net.4.running_var", "mid.net.4.weight", "out.bias", "out.weight", "u1.bias", "u1.weight", "u2.bias", "u2.weight", "u3.bias", "u3.weight"]
- State_dict source: <checkpoint>
- Keys del state_dict (muestra): ["d1.net.0.bias", "d1.net.0.weight", "d1.net.1.bias", "d1.net.1.num_batches_tracked", "d1.net.1.running_mean", "d1.net.1.running_var", "d1.net.1.weight", "d1.net.3.bias", "d1.net.3.weight", "d1.net.4.bias", "d1.net.4.num_batches_tracked", "d1.net.4.running_mean", "d1.net.4.running_var", "d1.net.4.weight", "d2.net.0.bias", "d2.net.0.weight", "d2.net.1.bias", "d2.net.1.num_batches_tracked", "d2.net.1.running_mean", "d2.net.1.running_var", "d2.net.1.weight", "d2.net.3.bias", "d2.net.3.weight", "d2.net.4.bias", "... (82 mas)"] | total=106
- num_classes: 6 (INFERIDO desde out.weight)
- base_channels: 16 (INFERIDO desde e1.net.0.weight)
- target_size: AUSENTE (INFERIDO no disponible sin cargar arquitectura ni inferencia)

## Bloqueos
- ninguno
```

Primer diagnostico:
- Los dos artifacts finales estan materializados localmente en models/final y pesan mas de 0 bytes.
- Los manifests laterales existen, pero no incluyen SHA-256 esperado; por eso la comparacion queda SIN_MANIFEST.
- Sagital guarda metadata de arquitectura en el checkpoint; axial esta guardado como state_dict crudo, por eso num_classes/base_channels se infieren desde tensores y target_size queda AUSENTE.
- No se ejecuto inferencia, no se cargo state_dict en arquitectura y no se declara real_baseline en este ticket.


