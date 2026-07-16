# AI-001 - Checkpoint inspection evidence

Repo de trabajo: PFI_MVPTest_Enzo_AImodule
Ruta de trabajo: C:\Users\enzoa\OneDrive\Documentos\1.ReposGitHub-Backends\PFI_MVPTest_Enzo_AImodule

Comando ejecutado:
$env:PYTHONPATH="ai_service"
.venv\Scripts\python.exe scripts\inspect_final_checkpoints.py

Salida:
```text
# AI-001 checkpoint inspection

## sagittal_spider
- Nombre / path relativo: sagittal_spider_multiclass_final_best.pt / sagittal_spider_multiclass_final_best.pt
- Fuente de path: PFI_MODEL_DIR/settings
- Existe: no | Tamano (bytes): 0
- SHA-256: AUSENTE | vs manifest: SIN_MANIFEST
- Detalle manifest: artifact no materializado localmente
- Bloqueo: artifact no materializado localmente

## axial_t2_alkafri
- Nombre / path relativo: axial_t2_alkafri_final_best.pt / axial_t2_alkafri_final_best.pt
- Fuente de path: PFI_MODEL_DIR/settings
- Existe: no | Tamano (bytes): 0
- SHA-256: AUSENTE | vs manifest: SIN_MANIFEST
- Detalle manifest: artifact no materializado localmente
- Bloqueo: artifact no materializado localmente

## Bloqueos
- sagittal_spider_multiclass_final_best.pt: artifact no materializado localmente
- axial_t2_alkafri_final_best.pt: artifact no materializado localmente
```

Primer diagnostico:
- Los paths se resolvieron desde settings/PFI_MODEL_DIR hacia los artifacts finales esperados.
- Ambos artifacts finales requeridos no estan materializados localmente en el repo real indicado.
- No se ejecuto inferencia ni se cargo state_dict en arquitectura; la inspeccion se detuvo en existencia/hash por bloqueo de artifact ausente.
- No se declara real_baseline en este ticket.

