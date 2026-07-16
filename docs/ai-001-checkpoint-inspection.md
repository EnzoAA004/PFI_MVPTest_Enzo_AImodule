# AI-001 - Checkpoint inspection evidence

Repo de trabajo: PFI_MVPTest_Enzo_AImodule
Ruta de trabajo: C:\Users\enzoa\OneDrive\Documentos\1.ReposGitHub-Backends\PFI_MVPTest_Enzo_AImodule

Precheck de artifacts:
```text
PFI_MODEL_DIR=AUSENTE; settings resuelve models/final
models/final/sagittal_spider_multiclass_final_best.pt: AUSENTE
models/final/axial_t2_alkafri_final_best.pt: AUSENTE
models/final/sagittal_spider_multiclass_final_best.pt.manifest.json: PRESENTE
models/final/axial_t2_alkafri_final_best.pt.manifest.json: PRESENTE
```

Comando de inspeccion:
```text
NO EJECUTADO: el ticket indica frenar si algun artifact .pt sigue ausente o en 0 bytes.
```

Resultado por artifact:

## sagittal_spider
- Nombre / path relativo: sagittal_spider_multiclass_final_best.pt / models/final/sagittal_spider_multiclass_final_best.pt
- Existe: no | Tamano (bytes): 0
- SHA-256: AUSENTE | vs manifest: SIN_MANIFEST (artifact .pt ausente; manifest lateral presente pero no se puede comparar sin archivo)
- Keys top-level del checkpoint: NO_INSPECCIONADO por artifact ausente
- Keys del state_dict (muestra): NO_INSPECCIONADO por artifact ausente
- num_classes / base_channels / target_size: AUSENTE por artifact ausente

## axial_t2_alkafri
- Nombre / path relativo: axial_t2_alkafri_final_best.pt / models/final/axial_t2_alkafri_final_best.pt
- Existe: no | Tamano (bytes): 0
- SHA-256: AUSENTE | vs manifest: SIN_MANIFEST (artifact .pt ausente; manifest lateral presente pero no se puede comparar sin archivo)
- Keys top-level del checkpoint: NO_INSPECCIONADO por artifact ausente
- Keys del state_dict (muestra): NO_INSPECCIONADO por artifact ausente
- num_classes / base_channels / target_size: AUSENTE por artifact ausente

Primer diagnostico:
- Los paths se resolvieron desde settings/PFI_MODEL_DIR hacia models/final porque PFI_MODEL_DIR no esta seteado.
- Los manifests laterales existen, pero ambos checkpoints .pt finales no estan materializados localmente.
- No se ejecuto torch.load, no se cargo state_dict en arquitectura y no se ejecuto inferencia.
- No se declara real_baseline en este ticket.

