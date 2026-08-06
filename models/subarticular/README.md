# Checkpoint del clasificador subarticular (P10.6)

Este directorio existe para que `docker-compose` pueda montarlo. El artefacto **no está en
Git**: pesa, es un release del modelo y no del código, y su hash está clavado en el código.

## Qué falta acá

```
frozen_subarticular_checkpoint.pt
```

Se obtiene del Drive del proyecto, en `PFI_MVP/models/P10_6_rsna_findings/`.

## Verificar antes de usarlo

El clasificador comprueba el SHA-256 **antes** de deserializar, así que un archivo que no
sea exactamente el esperado se rechaza en la carga y no en la predicción.

```bash
sha256sum frozen_subarticular_checkpoint.pt
# d41262d57b13c146a48ab15f5e183cc6a55fc92724b7d0c286cea1f2ce26e84a
```

El valor lo publica el propio servicio en `/health` (`checkpointHashExpected`) y vive en
`ai_service/pfi_ai_service/subarticular_frozen_classifier.py`.

## Cómo saber si quedó bien montado

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

En `degenerativeFindingModels.subarticular.status`:

| Valor | Significa |
|---|---|
| `not_configured` | falta `PFI_SUBARTICULAR_CHECKPOINT_PATH` |
| `artifact_missing` | la variable está, el archivo no |
| `invalid_hash` | el archivo está pero no es el esperado |
| `available` / `loaded` | listo |

Mientras no esté, `POST /degenerative-findings/subarticular/predict` responde 503 con
`SUBARTICULAR_CHECKPOINT_UNAVAILABLE`, y el backend lo traduce a
`AI_SUBARTICULAR_UNAVAILABLE`. El resto del sistema funciona normalmente.
