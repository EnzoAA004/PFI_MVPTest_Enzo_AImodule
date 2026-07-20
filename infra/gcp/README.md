# Entrenamiento final en Google Compute Engine

## Estado actual

La configuracion reproducible de la VM de entrenamiento esta guardada en esta carpeta. La VM todavia no fue creada desde estos archivos, no hay GPU consumiendo, y la cuenta sigue sin usar GPU por esta configuracion versionada.

Esta carpeta pertenece a la infraestructura del AI Module. El frontend y el backend no participan en el entrenamiento final de los modelos.

## Archivos existentes

### create-pfi-training-t4-v1.sh

Fuente de verdad operativa para crear la VM de entrenamiento. Contiene la configuracion de NVIDIA T4, `n1-standard-4`, disco, cuenta de servicio, politica de mantenimiento y limite de 12 horas por inicio.

### pfi-training-t4-v1.rest.http

Referencia REST exportada desde Google Cloud Console. Sirve para trazabilidad y comparacion, no como flujo principal de ejecucion.

### pfi-training-t4-v1.exported-reference.tf.disabled

Referencia de Terraform incompleta. No ejecutar ni renombrar a `.tf` sin adaptarla: no contiene los controles criticos de duracion maxima y accion final `STOP` que si estan en el script `gcloud` y en la referencia REST.

### training-vm.env.example

Contrato de variables de entorno para la VM de entrenamiento. Debe copiarse a `training-vm.env` para uso local. `training-vm.env` no se commitea.

### prepare-training-vm.sh

Prepara filesystem, checkout y entorno Python de la VM. No descarga datasets, no sube artifacts, no entrena y no instala torch/CUDA/drivers.

### preflight-training-vm.sh

Valida en modo read-only el contrato estatico y, dentro de la VM, el entorno real. No ejecuta los scripts de descarga ni sincronizacion.

### download-training-data.sh

Planifica o ejecuta descargas controladas desde Cloud Storage al disco local de la VM. El default es dry-run y `--execute` es obligatorio para transferir.

### sync-training-artifacts.sh

Planifica o ejecuta restauracion de resume y subida segura de checkpoints/artifacts por corrida. El default es dry-run y `--execute` es obligatorio para transferir.

## Arquitectura de almacenamiento

Cloud Storage:

```text
gs://pfi-rm-lumbar-artifacts-2026-ef/
|-- datasets/
|-- models/
|-- resume/
|-- manifests/
`-- outputs/
```

Disco local:

```text
/opt/pfi/
|-- PFI_MVPTest_Enzo_AImodule/
|-- data/
`-- outputs/
    `-- final_training/
        |-- resume/
        |-- models/
        |-- manifests/
        `-- logs/
```

GCS es persistencia. El disco local es espacio de trabajo. No se debe entrenar directamente desde GCS: los datasets deben descargarse al disco local antes de entrenar. Los checkpoints deben guardarse primero localmente y sincronizarse despues. No eliminar la VM hasta verificar que los artifacts necesarios estan en GCS.

## Identidad de corrida

`PFI_RUN_ID` identifica una corrida de entrenamiento y debe mantenerse estable entre reinicios para poder restaurar checkpoints. El valor default versionado es:

```bash
PFI_RUN_ID=pfi-final-training-v1
```

Debe cumplir `^[a-z0-9][a-z0-9-]{0,62}$`. Una corrida nueva se crea cambiando explicitamente `PFI_RUN_ID`; no se genera automaticamente con fecha/hora.

Los prefijos derivados se calculan al hacer `source` del env:

```bash
PFI_GCS_RUN_MODELS_URI="${PFI_GCS_MODELS_URI}/${PFI_RUN_ID}"
PFI_GCS_RUN_RESUME_URI="${PFI_GCS_RESUME_URI}/${PFI_RUN_ID}"
PFI_GCS_RUN_MANIFESTS_URI="${PFI_GCS_MANIFESTS_URI}/${PFI_RUN_ID}"
PFI_GCS_RUN_OUTPUTS_URI="${PFI_GCS_OUTPUTS_URI}/${PFI_RUN_ID}"
```

Ningun script sincroniza contra la raiz del bucket. Los destinos deben contener `models/`, `resume/`, `manifests/` u `outputs/` y el `PFI_RUN_ID`.

## Estructura de datasets en GCS

```text
datasets/
|-- SPIDER/
|   |-- images/
|   `-- masks/
|-- AXIAL_ALKAFRI/
|-- metadata/
|   |-- E5_multiclass_label_mapping.json
|   `-- E9_t2_final_labels_curated_split.csv
`-- bootstrap_models/
    `-- sagittal_spider_multiclass_final_best.pt
```

`bootstrap_models/` es opcional si el mapping JSON existe. `SPIDER`, `AXIAL_ALKAFRI` y `metadata` son requeridos para preparar datos reales.

## Cuenta de servicio

Cuenta prevista:

```text
pfi-training-vm@pfi-asplanatti-fabrello-v1.iam.gserviceaccount.com
```

La VM usa identidad adjunta. No utiliza clave JSON. La cuenta tiene acceso sobre los objetos del bucket segun la configuracion IAM definida fuera del repositorio. No guardar credenciales, tokens ni claves en este repositorio.

## Configuracion de la VM

- nombre: `pfi-training-t4-v1`
- zona: `us-central1-a`
- maquina: `n1-standard-4`
- GPU: `1 NVIDIA T4`
- imagen: `pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260713`
- disco: `100 GB pd-balanced`
- autoDelete: `true`
- maxRunDuration: `43200` segundos
- accion final: `STOP`
- automaticRestart: `false`
- IP externa efimera
- HTTP/HTTPS no habilitados

## Seguridad de costos

Guardar esta configuracion en GitHub cuesta USD 0. Crear archivos de configuracion no crea la VM. La VM comienza a consumir credito al ejecutarse.

El techo conservador de computo por inicio es 12 horas, aproximadamente USD 6,50 mas disco, IP y operaciones. Al detenerse dejan de cobrarse CPU/GPU; el disco continua cobrando mientras exista. El presupuesto de USD 50 es una alerta, no un bloqueo. Cada inicio vuelve a otorgar hasta 12 horas.

No crear la VM hasta tener notebook portable, datasets y preflight listos.

## Uso futuro del archivo env

```bash
cp infra/gcp/training-vm.env.example infra/gcp/training-vm.env
set -a
source infra/gcp/training-vm.env
set +a
```

`training-vm.env` es local, no debe commitearse y no debe contener secretos. Modificarlo solamente para rutas o parametros de una corrida.

## Preparacion local de la VM

Uso futuro dentro de la VM:

```bash
cp infra/gcp/training-vm.env.example infra/gcp/training-vm.env

bash infra/gcp/prepare-training-vm.sh \
  --env-file infra/gcp/training-vm.env
```

Dry-run local o previo:

```bash
bash infra/gcp/prepare-training-vm.sh \
  --dry-run \
  --skip-apt \
  --skip-python \
  --env-file infra/gcp/training-vm.env.example
```

El dry-run informa acciones y no modifica el sistema.

## Preflight

`preflight-training-vm.sh` es read-only. No entrena, no descarga datos, no sube datos, no crea archivos dentro de `/opt/pfi`, no ejecuta notebooks y no modifica recursos de Google Cloud.

Preflight estatico desde el checkout local:

```bash
bash infra/gcp/preflight-training-vm.sh \
  --mode static \
  --env-file infra/gcp/training-vm.env.example
```

El modo `static` valida el contrato de variables, rutas, archivos del repo, notebook v4 como JSON, arquitectura Python, gitignore, ausencia de artifacts trackeados, scripts nuevos con sintaxis Bash, modo `100755`, ausencia de comandos GCS destructivos y control explicito de `--execute`.

Preflight real dentro de la VM:

```bash
bash infra/gcp/preflight-training-vm.sh \
  --mode vm \
  --env-file infra/gcp/training-vm.env
```

El modo `vm` debe ejecutarse dentro de Compute Engine. Agrega validaciones de Linux, comandos, venv, GPU/CUDA, `nvidia-smi`, metadata de Compute Engine, identidad adjunta, bucket/prefijos GCS en modo solo lectura, disco y datasets locales. Tambien informa `PFI_RUN_ID` en el resumen. Si el prefijo remoto de resume para la corrida esta vacio, debe informarse sin fallar en la primera corrida.

## Descarga de datos

Dry-run:

```bash
bash infra/gcp/download-training-data.sh \
  --component all \
  --dry-run \
  --env-file infra/gcp/training-vm.env
```

Ejecucion real futura, solo dentro de la VM preparada:

```bash
bash infra/gcp/download-training-data.sh \
  --component all \
  --execute \
  --env-file infra/gcp/training-vm.env
```

El script nunca elimina archivos locales, no usa `--delete-unmatched-destination-objects`, no usa `gsutil` y no crea directorios en `--execute`: `prepare-training-vm.sh` debe haber preparado los destinos. `bootstrap` y `resume` pueden estar vacios; datasets requeridos no pueden estar vacios.

Componentes disponibles: `spider`, `axial`, `metadata`, `bootstrap`, `resume`, `datasets`, `all`. `--require-resume` solo debe usarse cuando se espera restaurar checkpoints previos.

## Restauracion de resume

```bash
bash infra/gcp/sync-training-artifacts.sh \
  --mode pull-resume \
  --execute \
  --env-file infra/gcp/training-vm.env
```

Si el prefijo `resume/${PFI_RUN_ID}` esta vacio, la primera corrida sigue siendo valida y no se crean archivos locales.

## Sincronizacion de checkpoints

Dry-run para checkpoints de resume:

```bash
bash infra/gcp/sync-training-artifacts.sh \
  --mode push-resume \
  --dry-run \
  --env-file infra/gcp/training-vm.env
```

Ejecucion real futura:

```bash
bash infra/gcp/sync-training-artifacts.sh \
  --mode push-resume \
  --execute \
  --env-file infra/gcp/training-vm.env
```

Solo se consideran `*.last_checkpoint.pt`, `*.best_checkpoint.pt` y JSON de estado. Archivos recientes, temporales, ocultos, symlinks y nombres no reconocidos se omiten.

## Sincronizacion final

```bash
bash infra/gcp/sync-training-artifacts.sh \
  --mode push-final \
  --execute \
  --env-file infra/gcp/training-vm.env
```

Sube modelos finales permitidos, manifests y logs seleccionados hacia prefijos separados por `PFI_RUN_ID`. No sincroniza el directorio de salida completo de forma generica.

## Seguridad

- `PFI_SYNC_DRY_RUN=1` es el default seguro.
- `--execute` es obligatorio para descargar o subir.
- Cambiar `PFI_SYNC_DRY_RUN=0` no alcanza por si solo para escribir en GCS.
- No se borra ningun destino local ni remoto.
- No se usa `gsutil`, `gcloud storage rm`, `gcloud storage mv` ni `--delete-unmatched-destination-objects`.
- Los artifacts quedan separados por `PFI_RUN_ID`.
- Checkpoints recientes se omiten para evitar subir archivos en escritura.
- La seleccion de artifacts usa staging temporal fuera del repo y fuera de `/opt/pfi`.
- El manifest registra SHA-256, tamanio, destino y metadata minima de entorno sin credenciales ni rutas internas completas.
- No eliminar la VM antes de verificar `models/`, `resume/` y `manifests/` en GCS.

## Trabajo pendiente

Implementados:

1. `prepare-training-vm.sh`
2. `preflight-training-vm.sh`
3. `download-training-data.sh`
4. `sync-training-artifacts.sh`

Pendientes:

1. `train_final_models_v5_cloud_portable.ipynb`
2. `run-final-training.sh`
3. carga real de datasets
4. cuota GPU
5. creacion de VM

## Fuente de verdad

- creacion operativa: `create-pfi-training-t4-v1.sh`;
- valores de ejecucion: `training-vm.env`;
- notebook final Colab vigente: `train_final_models_v4_final.ipynb`;
- futuro notebook cloud: `train_final_models_v5_cloud_portable.ipynb`;
- no usar el Terraform disabled.