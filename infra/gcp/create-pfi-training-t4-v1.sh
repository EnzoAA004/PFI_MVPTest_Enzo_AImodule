#!/usr/bin/env bash
set -euo pipefail

# Configuración validada para la VM de entrenamiento final con NVIDIA T4.
# No ejecutar hasta que:
# - la cuenta de facturación permita GPUs;
# - la cuota global GPUs (all regions) sea 1;
# - la cuota NVIDIA T4 de us-central1 sea 1;
# - el notebook esté adaptado a rutas de VM y Cloud Storage.
#
# Seguridad de costos:
# - duración máxima por inicio: 12 horas;
# - acción al finalizar: STOP;
# - reinicio automático: desactivado;
# - el disco se elimina al borrar la instancia, no al detenerla.

gcloud compute instances create pfi-training-t4-v1 \
    --project=pfi-asplanatti-fabrello-v1 \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --instance-termination-action=STOP \
    --max-run-duration=43200s \
    --service-account=pfi-training-vm@pfi-asplanatti-fabrello-v1.iam.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --accelerator=count=1,type=nvidia-tesla-t4 \
    --create-disk=auto-delete=yes,boot=yes,device-name=pfi-training-t4-v1,image=projects/ml-images/global/images/pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260713,mode=rw,size=100,type=pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=none
