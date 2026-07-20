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

## Prerrequisitos antes de crear la VM

- notebook v5 portable validado;
- datasets presentes en GCS;
- CSV axial revisado;
- preflight implementado;
- sync implementado;
- cuota global GPUs (all regions) = 1;
- cuota T4 `us-central1` = 1;
- credito y presupuesto revisados;
- comando de eliminacion conocido.

## Uso futuro del archivo env

```bash
cp infra/gcp/training-vm.env.example infra/gcp/training-vm.env
set -a
source infra/gcp/training-vm.env
set +a
```

`training-vm.env` es local, no debe commitearse y no debe contener secretos. Modificarlo solamente para rutas o parametros de una corrida.

## Trabajo pendiente

1. `prepare-training-vm.sh`
2. `preflight-training-vm.sh`
3. `sync-training-artifacts.sh`
4. `train_final_models_v5_cloud_portable.ipynb`
5. `run-final-training.sh`
6. carga y validacion de datasets
7. habilitacion de cuota
8. creacion de VM

## Fuente de verdad

- creacion operativa: `create-pfi-training-t4-v1.sh`;
- valores de ejecucion: `training-vm.env`;
- notebook final Colab vigente: `train_final_models_v4_final.ipynb`;
- futuro notebook cloud: `train_final_models_v5_cloud_portable.ipynb`;
- no usar el Terraform disabled.
