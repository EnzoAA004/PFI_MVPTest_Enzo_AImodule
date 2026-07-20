#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_CHECKOUT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_CHECKOUT_ROOT" ]]; then
  REPO_CHECKOUT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
fi

ENV_FILE=""
MODE="vm"
PASS=0; WARN=0; FAIL=0
S_PROJECT=NO_OBTENIDO; S_VM=NO_OBTENIDO; S_ZONE=NO_OBTENIDO; S_SA=NO_OBTENIDO; S_HEAD=NO_OBTENIDO
S_PY=NO_OBTENIDO; S_TORCH=NO_OBTENIDO; S_CUDA=NO_OBTENIDO; S_GPU=NO_OBTENIDO; S_DISK=NO_OBTENIDO
S_BUCKET=NO_OBTENIDO; S_SPIDER=NO_OBTENIDO; S_AXIAL=NO_OBTENIDO
STRUCTURAL_VARS=(PFI_VM_ROOT PFI_REPO_ROOT PFI_DATA_ROOT PFI_OUTPUT_ROOT PFI_TRAIN_OUTPUT_DIR PFI_VENV_DIR PFI_GCS_BUCKET_URI PFI_GCP_PROJECT_ID PFI_GCP_ZONE PFI_VM_NAME PFI_VM_SERVICE_ACCOUNT)
ALL_VARS=(
  PFI_GCP_PROJECT_ID PFI_GCP_REGION PFI_GCP_ZONE PFI_VM_NAME PFI_VM_SERVICE_ACCOUNT
  PFI_GCS_BUCKET_URI PFI_GCS_DATASETS_URI PFI_GCS_MODELS_URI PFI_GCS_RESUME_URI PFI_GCS_MANIFESTS_URI PFI_GCS_OUTPUTS_URI
  PFI_VM_ROOT PFI_REPO_URL PFI_REPO_ROOT PFI_DATA_ROOT PFI_OUTPUT_ROOT PFI_TRAIN_OUTPUT_DIR
  PFI_LOCAL_RESUME_DIR PFI_LOCAL_MODELS_DIR PFI_LOCAL_MANIFESTS_DIR PFI_LOCAL_LOGS_DIR PFI_PYTHON_BIN PFI_VENV_DIR
  PFI_MIN_FREE_DISK_GB REQUIRE_GPU RUN_SAGITTAL RUN_AXIAL PFI_SMOKE_RUN PFI_RESUME_TRAINING
  PFI_MAX_EPOCHS PFI_EARLY_STOP_PATIENCE PFI_PREFLIGHT_ONLY PFI_SYNC_DRY_RUN PFI_DOWNLOAD_DATASETS PFI_SYNC_RESUME PFI_SYNC_FINAL_ARTIFACTS
  SPIDER_IMAGES_DIR SPIDER_MASKS_DIR SPIDER_LABEL_GROUP_MAPPING_JSON SPIDER_CHECKPOINT_FOR_LABEL_MAP
  AXIAL_E9_CURATED_SPLIT_CSV AXIAL_IMAGES_DIR AXIAL_MASKS_DIR AXIAL_IMAGE_COL AXIAL_MASK_COL AXIAL_PATIENT_COL AXIAL_SPLIT_COL
)

pass(){ PASS=$((PASS+1)); printf '[PASS] %s\n' "$*"; }
warn(){ WARN=$((WARN+1)); printf '[WARN] %s\n' "$*"; }
fail(){ FAIL=$((FAIL+1)); printf '[FAIL] %s\n' "$*"; }
info(){ printf '[INFO] %s\n' "$*"; }
usage(){ cat <<'USAGE'
Uso: preflight-training-vm.sh [--env-file PATH] [--mode static|vm]
Read-only: no crea archivos, no instala, no descarga/sube datos, no ejecuta notebooks ni entrenamiento.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) [[ $# -ge 2 ]] || { printf '[FAIL] --env-file requiere PATH\n' >&2; exit 2; }; ENV_FILE="$2"; shift 2;;
    --mode) [[ $# -ge 2 ]] || { printf '[FAIL] --mode requiere static|vm\n' >&2; exit 2; }; MODE="$2"; shift 2;;
    --help) usage; exit 0;;
    *) printf '[FAIL] argumento desconocido: %s\n' "$1" >&2; usage >&2; exit 2;;
  esac
done
[[ "$MODE" == static || "$MODE" == vm ]] || { printf '[FAIL] --mode invalido: %s\n' "$MODE" >&2; exit 2; }

select_env(){ if [[ -n "$ENV_FILE" ]]; then printf '%s\n' "$ENV_FILE"; elif [[ -f "$SCRIPT_DIR/training-vm.env" ]]; then printf '%s\n' "$SCRIPT_DIR/training-vm.env"; else printf '%s\n' "$SCRIPT_DIR/training-vm.env.example"; fi; }
load_env(){ ENV_FILE="$(select_env)"; [[ -f "$ENV_FILE" ]] || { printf '[FAIL] env-file inexistente: %s\n' "$ENV_FILE" >&2; exit 2; }; set -a; source "$ENV_FILE"; set +a; pass "env cargable: $ENV_FILE"; }
validate_structural_vars(){ local missing=() v; for v in "${STRUCTURAL_VARS[@]}"; do [[ -n "${!v:-}" ]] || missing+=("$v"); done; if [[ ${#missing[@]} -gt 0 ]]; then printf '[FAIL] variables estructurales faltantes: %s\n' "${missing[*]}" >&2; exit 2; fi; }
req(){ local n="$1"; [[ -n "${!n:-}" ]] && pass "var $n" || fail "variable ausente: $n"; }
cmd(){ command -v "$1" >/dev/null 2>&1; }
need_cmd(){ cmd "$1" && pass "comando $1" || fail "comando faltante: $1"; }
norm(){ python3 -c 'import os,sys; print(os.path.normpath(sys.argv[1]))' "$1" 2>/dev/null || printf '%s\n' "$1"; }
path_ok(){ local n="$1" v="${!n:-}" r pv; [[ -n "$v" && "$v" != / && "$v" == /* ]] || { fail "$n ruta invalida: ${v:-AUSENTE}"; return; }; r="$(norm "$PFI_VM_ROOT")"; pv="$(norm "$v")"; if [[ "$n" != PFI_VM_ROOT && "$pv" != "$r" && "$pv" != "$r"/* ]]; then fail "$n fuera de PFI_VM_ROOT"; else pass "$n ruta segura"; fi; }
eqv(){ local n="$1" e="$2"; [[ "${!n:-}" == "$e" ]] && pass "$n=$e" || fail "$n esperado $e obtenido ${!n:-AUSENTE}"; }
boolv(){ local n="$1" v="${!n:-}"; [[ "$v" == 0 || "$v" == 1 ]] && pass "$n booleano" || fail "$n debe ser 0/1"; }
compile_python_readonly(){ local path="$1" py=python; cmd python || py=python3; "$py" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); compile(p.read_text(encoding="utf-8"), str(p), "exec")' "$path"; }
worktree_check(){
  local status
  status="$(git -C "$REPO_CHECKOUT_ROOT" status --porcelain)"
  if [[ -z "$status" ]]; then pass "worktree clean"; else warn "worktree dirty"; printf '%s\n' "$status" | sed 's/^/[WARN] git status: /'; fi
}
static_checks(){
  local n f py=python
  validate_structural_vars
  for n in "${ALL_VARS[@]}"; do req "$n"; done
  for n in PFI_VM_ROOT PFI_REPO_ROOT PFI_DATA_ROOT PFI_OUTPUT_ROOT PFI_TRAIN_OUTPUT_DIR PFI_LOCAL_RESUME_DIR PFI_LOCAL_MODELS_DIR PFI_LOCAL_MANIFESTS_DIR PFI_LOCAL_LOGS_DIR PFI_VENV_DIR; do path_ok "$n"; done
  eqv PFI_MAX_EPOCHS 80; eqv PFI_EARLY_STOP_PATIENCE 12; eqv PFI_RESUME_TRAINING 1; eqv REQUIRE_GPU 1; eqv PFI_SMOKE_RUN 0
  for n in RUN_SAGITTAL RUN_AXIAL PFI_PREFLIGHT_ONLY PFI_SYNC_DRY_RUN PFI_DOWNLOAD_DATASETS PFI_SYNC_RESUME PFI_SYNC_FINAL_ARTIFACTS; do boolv "$n"; done
  [[ "$PFI_GCS_BUCKET_URI" == gs://* ]] && pass "bucket gs://" || fail "bucket no usa gs://"
  [[ "$PFI_VM_SERVICE_ACCOUNT" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.iam\.gserviceaccount\.com$ ]] && pass "service account formato valido" || fail "service account formato invalido"
  for f in notebooks/train_final_models_v4_final.ipynb ai_service/pfi_ai_service/model_architectures.py requirements.txt infra/gcp/create-pfi-training-t4-v1.sh; do [[ -f "$REPO_CHECKOUT_ROOT/$f" ]] && pass "existe $f" || fail "falta $f"; done
  cmd python || py=python3
  if cmd "$py"; then
    "$py" -m json.tool "$REPO_CHECKOUT_ROOT/notebooks/train_final_models_v4_final.ipynb" >/dev/null && pass "notebook v4 JSON valido" || fail "notebook v4 JSON invalido"
    compile_python_readonly "$REPO_CHECKOUT_ROOT/ai_service/pfi_ai_service/model_architectures.py" && pass "model_architectures.py compila sin bytecode" || fail "model_architectures.py no compila"
  else
    fail "python no disponible"
  fi
  S_HEAD="$(git -C "$REPO_CHECKOUT_ROOT" rev-parse --short HEAD 2>/dev/null || printf NO_OBTENIDO)"
  git -C "$REPO_CHECKOUT_ROOT" diff --quiet -- notebooks/train_final_models_v4_final.ipynb && pass "notebook v4 no modificado" || fail "notebook v4 modificado"
  [[ -z "$(git -C "$REPO_CHECKOUT_ROOT" ls-files '*.pt')" ]] && pass "sin .pt trackeados" || fail "hay .pt trackeados"
  git -C "$REPO_CHECKOUT_ROOT" check-ignore -q infra/gcp/training-vm.env && pass "training-vm.env ignorado" || fail "training-vm.env no ignorado"
  git -C "$REPO_CHECKOUT_ROOT" check-ignore -q infra/gcp/training-vm.env.example && fail "env example ignorado" || pass "env example no ignorado"
  local secret_re
  secret_re='PRIVATE ''KEY|private_''key|client_''secret|access_''token|pass''word=|credentials\.json|BEGIN ''RSA|BEGIN ''OPENSSH'
  grep -REn "$secret_re" "$REPO_CHECKOUT_ROOT/infra/gcp/prepare-training-vm.sh" "$REPO_CHECKOUT_ROOT/infra/gcp/preflight-training-vm.sh" "$REPO_CHECKOUT_ROOT/infra/gcp/README.md" >/dev/null && fail "patrones de secreto encontrados" || pass "sin patrones de secretos infra"
  worktree_check
}

metadata_get(){ curl -fsS --max-time 2 -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/$1"; }
metadata_check(){
  [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]] && pass "sin GOOGLE_APPLICATION_CREDENTIALS" || fail "GOOGLE_APPLICATION_CREDENTIALS definido"
  local p i zf z e
  p="$(metadata_get project/project-id 2>/dev/null || true)"
  if [[ -z "$p" ]]; then fail "metadata server no responde"; return; fi
  i="$(metadata_get instance/name 2>/dev/null || true)"; zf="$(metadata_get instance/zone 2>/dev/null || true)"; e="$(metadata_get instance/service-accounts/default/email 2>/dev/null || true)"; z="${zf##*/}"
  S_PROJECT="$p"; S_VM="$i"; S_ZONE="$z"; S_SA="$e"
  [[ "$p" == "$PFI_GCP_PROJECT_ID" ]] && pass "metadata project" || fail "metadata project no coincide"
  [[ "$i" == "$PFI_VM_NAME" ]] && pass "metadata VM" || fail "metadata VM no coincide"
  [[ "$z" == "$PFI_GCP_ZONE" ]] && pass "metadata zone" || fail "metadata zone no coincide"
  [[ "$e" == "$PFI_VM_SERVICE_ACCOUNT" ]] && pass "metadata service account" || fail "metadata service account no coincide"
}
vm_system(){
  [[ "$(uname -s 2>/dev/null || printf unknown)" == Linux ]] && pass "Linux" || fail "mode vm requiere Linux/GCE"
  for c in git curl jq python3 gcloud nvidia-smi df; do need_cmd "$c"; done
  if [[ -d "$PFI_REPO_ROOT/.git" ]]; then
    pass "PFI_REPO_ROOT repo Git"; local o; o="$(git -C "$PFI_REPO_ROOT" remote get-url origin 2>/dev/null || true)"
    [[ "${o%.git}" == "${PFI_REPO_URL%.git}" ]] && pass "origin coincide" || fail "origin no coincide"
    info "rama: $(git -C "$PFI_REPO_ROOT" branch --show-current 2>/dev/null || true)"; S_HEAD="$(git -C "$PFI_REPO_ROOT" rev-parse --short HEAD 2>/dev/null || printf NO_OBTENIDO)"
    local st; st="$(git -C "$PFI_REPO_ROOT" status --porcelain)"; if [[ -z "$st" ]]; then pass "worktree clean"; else warn "worktree dirty"; printf '%s\n' "$st" | sed 's/^/[WARN] git status: /'; fi
  else fail "PFI_REPO_ROOT faltante o no git"; fi
  [[ -x "$PFI_VENV_DIR/bin/python" ]] && pass "venv python existe" || fail "venv python faltante"
  local d; for d in "$PFI_VM_ROOT" "$PFI_DATA_ROOT" "$PFI_OUTPUT_ROOT" "$PFI_TRAIN_OUTPUT_DIR" "$PFI_LOCAL_RESUME_DIR" "$PFI_LOCAL_MODELS_DIR" "$PFI_LOCAL_MANIFESTS_DIR" "$PFI_LOCAL_LOGS_DIR"; do [[ -d "$d" ]] && pass "dir existe $d" || fail "dir faltante $d"; done
  for d in "$PFI_OUTPUT_ROOT" "$PFI_TRAIN_OUTPUT_DIR" "$PFI_LOCAL_RESUME_DIR" "$PFI_LOCAL_MODELS_DIR" "$PFI_LOCAL_MANIFESTS_DIR" "$PFI_LOCAL_LOGS_DIR"; do [[ -w "$d" ]] && pass "dir escribible $d" || fail "dir no escribible $d"; done
}
disk_check(){
  local line total used avail mount agb tgb ugb
  line="$(df -Pk "$PFI_VM_ROOT" 2>/dev/null | awk 'NR==2{print $2,$3,$4,$6}')" || true
  [[ -n "$line" ]] || { fail "df fallo"; return; }
  read -r total used avail mount <<<"$line"
  agb=$((avail/1024/1024)); tgb=$((total/1024/1024)); ugb=$((used/1024/1024)); S_DISK="${agb}GB libres"
  info "disco mount=$mount total=${tgb}GB usado=${ugb}GB disponible=${agb}GB"
  (( agb >= PFI_MIN_FREE_DISK_GB )) && pass "disco sobre minimo operativo" || fail "disco bajo minimo operativo"
}
python_vm(){
  [[ -x "$PFI_VENV_DIR/bin/python" ]] || { fail "venv python faltante"; return; }
  local out rc; set +e
  out="$("$PFI_VENV_DIR/bin/python" - <<'PY'
import importlib, sys
mods=['numpy','scipy','pandas','matplotlib','skimage','sklearn','torch','torchvision','pytest','SimpleITK','pydicom','PIL','nbformat','nbconvert','jupyter','ipykernel']
missing=[]
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: missing.append(f'{m}:{e}')
if missing:
    print('MISSING='+' | '.join(missing)); raise SystemExit(1)
import torch
print('PY='+sys.version.split()[0]); print('TORCH='+torch.__version__); print('CUDA='+str(torch.version.cuda)); print('CUDA_AVAILABLE='+str(torch.cuda.is_available())); print('CUDA_COUNT='+str(torch.cuda.device_count()))
if torch.cuda.device_count()>=1:
    p=torch.cuda.get_device_properties(0); print('GPU='+p.name); print('GPU_MB='+str(p.total_memory//1024//1024))
PY
)"; rc=$?; set -e
  [[ "$rc" -eq 0 ]] || { fail "imports Python fallan: $out"; return; }
  pass "imports Python OK"; S_PY="$(printf '%s\n' "$out"|awk -F= '/^PY=/{print $2}')"; S_TORCH="$(printf '%s\n' "$out"|awk -F= '/^TORCH=/{print $2}')"; S_CUDA="$(printf '%s\n' "$out"|awk -F= '/^CUDA=/{print $2}')"
  local ca cc gn; ca="$(printf '%s\n' "$out"|awk -F= '/^CUDA_AVAILABLE=/{print $2}')"; cc="$(printf '%s\n' "$out"|awk -F= '/^CUDA_COUNT=/{print $2}')"; gn="$(printf '%s\n' "$out"|awk -F= '/^GPU=/{print $2}')"; S_GPU="${gn:-NO_GPU}"
  [[ "$REQUIRE_GPU" == 1 && "$ca" == True ]] && pass "CUDA disponible" || fail "CUDA no disponible con REQUIRE_GPU=1"
  [[ "${cc:-0}" -ge 1 ]] && pass "GPU detectada" || fail "sin GPU"
  [[ -n "$gn" && "$gn" != *T4* ]] && warn "GPU no contiene T4: $gn" || true
}
nvidia_check(){ local o; o="$(nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader 2>/dev/null)" && { pass "nvidia-smi OK"; info "nvidia-smi: $o"; } || fail "nvidia-smi fallo"; }

gcloud_ls_uri(){
  local uri="$1" label="$2" required_dataset="$3" out rc lower
  set +e; out="$(gcloud storage ls "$uri/" 2>&1)"; rc=$?; set -e
  lower="$(printf '%s' "$out" | tr '[:upper:]' '[:lower:]')"
  if [[ "$rc" -eq 0 ]]; then
    if [[ -z "$out" ]]; then
      if [[ "$label" == datasets && "$required_dataset" == 1 ]]; then fail "datasets vacio"; return 1; fi
      warn "prefijo vacio $label"; return 0
    fi
    pass "prefijo listable $label"; return 0
  fi
  if printf '%s' "$lower" | grep -Eq 'denied|forbidden|permission'; then fail "sin permiso $label"; return 1; fi
  if printf '%s' "$lower" | grep -Eq 'matched no objects|no urls matched|no objects|url matched no objects'; then
    if [[ "$label" == datasets && "$required_dataset" == 1 ]]; then fail "datasets vacio"; return 1; fi
    warn "prefijo vacio $label"; return 0
  fi
  fail "error desconocido al listar $label"; return 1
}
gcloud_check(){
  cmd gcloud || { fail "gcloud faltante"; S_BUCKET=FAIL; return; }
  info "gcloud project: $(gcloud config get-value project 2>/dev/null || printf NO_OBTENIDO)"
  gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | sed 's/^/[INFO] account: /' || warn "gcloud auth list no disponible"
  if gcloud storage ls "$PFI_GCS_BUCKET_URI/" >/dev/null 2>&1; then pass "bucket listable"; S_BUCKET=OK; else fail "bucket no listable"; S_BUCKET=FAIL; fi
  local before="$FAIL"
  gcloud_ls_uri "$PFI_GCS_DATASETS_URI" datasets "$PFI_DOWNLOAD_DATASETS" || true
  gcloud_ls_uri "$PFI_GCS_MODELS_URI" models 0 || true
  gcloud_ls_uri "$PFI_GCS_RESUME_URI" resume 0 || true
  gcloud_ls_uri "$PFI_GCS_MANIFESTS_URI" manifests 0 || true
  gcloud_ls_uri "$PFI_GCS_OUTPUTS_URI" outputs 0 || true
  [[ "$FAIL" -gt "$before" ]] && S_BUCKET=FAIL || [[ "$S_BUCKET" == OK ]] || S_BUCKET=WARN
}
count_files(){ find "$1" -type f 2>/dev/null | wc -l | tr -d ' '; }
spider_check(){
  local before="$FAIL" ok=1 im ms
  if [[ "$RUN_SAGITTAL" != 1 ]]; then info "SPIDER skip"; S_SPIDER=SKIP; return; fi
  [[ -d "$SPIDER_IMAGES_DIR" ]] && pass "SPIDER images dir" || { fail "SPIDER images dir faltante"; ok=0; }
  [[ -d "$SPIDER_MASKS_DIR" ]] && pass "SPIDER masks dir" || { fail "SPIDER masks dir faltante"; ok=0; }
  if [[ "$ok" -eq 1 ]]; then
    im="$(count_files "$SPIDER_IMAGES_DIR")"; ms="$(count_files "$SPIDER_MASKS_DIR")"
    info "SPIDER images=$im masks=$ms"
    [[ "$im" -gt 0 ]] && pass "SPIDER images no vacio" || fail "SPIDER images vacio"
    [[ "$ms" -gt 0 ]] && pass "SPIDER masks no vacio" || fail "SPIDER masks vacio"
    [[ "$im" == "$ms" ]] || warn "SPIDER conteos difieren"
  fi
  if [[ -f "$SPIDER_LABEL_GROUP_MAPPING_JSON" ]]; then
    "$PFI_VENV_DIR/bin/python" -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); assert isinstance(d,dict) and d' "$SPIDER_LABEL_GROUP_MAPPING_JSON" && pass "SPIDER mapping JSON" || fail "SPIDER mapping invalido"
  elif [[ -f "$SPIDER_CHECKPOINT_FOR_LABEL_MAP" ]]; then warn "SPIDER mapping JSON faltante; fallback presente"
  else fail "faltan SPIDER mapping y fallback"; fi
  [[ -f "$SPIDER_CHECKPOINT_FOR_LABEL_MAP" ]] && info "SPIDER fallback checkpoint presente" || warn "SPIDER fallback ausente; OK si JSON existe"
  [[ "$FAIL" -gt "$before" ]] && S_SPIDER=FAIL || { [[ "$WARN" -gt 0 ]] && S_SPIDER=WARN || S_SPIDER=OK; }
}
axial_check(){
  local before="$FAIL"
  if [[ "$RUN_AXIAL" != 1 ]]; then info "AXIAL skip"; S_AXIAL=SKIP; return; fi
  [[ -f "$AXIAL_E9_CURATED_SPLIT_CSV" ]] && pass "AXIAL CSV" || { fail "AXIAL CSV faltante"; S_AXIAL=FAIL; return; }
  [[ -d "$AXIAL_IMAGES_DIR" ]] && pass "AXIAL images dir" || fail "AXIAL images dir faltante"
  [[ -d "$AXIAL_MASKS_DIR" ]] && pass "AXIAL masks dir" || fail "AXIAL masks dir faltante"
  [[ -x "$PFI_VENV_DIR/bin/python" ]] || { fail "venv python faltante para AXIAL"; S_AXIAL=FAIL; return; }
  local out rc; set +e
  out="$("$PFI_VENV_DIR/bin/python" - <<'PY' "$AXIAL_E9_CURATED_SPLIT_CSV" "$AXIAL_IMAGES_DIR" "$AXIAL_MASKS_DIR" "$AXIAL_IMAGE_COL" "$AXIAL_MASK_COL" "$AXIAL_PATIENT_COL" "$AXIAL_SPLIT_COL"
import re, sys
from pathlib import Path
import pandas as pd
csv,img_root,mask_root,img_col,mask_col,patient_col,split_col=sys.argv[1:]
df=pd.read_csv(csv); req=[img_col,mask_col,patient_col,split_col]
missing_cols=[c for c in req if c not in df.columns]
if missing_cols:
    print('missing_cols='+','.join(missing_cols)); raise SystemExit(1)
nulls=df[req].isna().sum().to_dict()
print(f'rows_total={len(df)}')
print(f'patients_unique={df[patient_col].nunique(dropna=True)}')
print(f'splits={df[split_col].astype(str).value_counts(dropna=False).to_dict()}')
print(f'nulls={nulls}')
legacy_total=0; miss_i_total=0; miss_m_total=0
legacy_ex=[]; miss_i_ex=[]; miss_m_ex=[]
win=re.compile(r'^[A-Za-z]:[\\/]')
for _, row in df.iterrows():
    vals=[str(row[img_col]), str(row[mask_col])]
    for val in vals:
        low=val.lower()
        if '/content/drive' in low or 'google drive' in low or win.match(val):
            legacy_total += 1
            if len(legacy_ex) < 10: legacy_ex.append(val)
    img=Path(vals[0]); msk=Path(vals[1])
    if not img.is_absolute(): img=Path(img_root)/img
    if not msk.is_absolute(): msk=Path(mask_root)/msk
    if not img.exists():
        miss_i_total += 1
        if len(miss_i_ex) < 10: miss_i_ex.append(str(img))
    if not msk.exists():
        miss_m_total += 1
        if len(miss_m_ex) < 10: miss_m_ex.append(str(msk))
print(f'legacy_paths_total={legacy_total}')
print(f'missing_images_total={miss_i_total}')
print(f'missing_masks_total={miss_m_total}')
if legacy_ex: print('legacy_paths_examples='+' | '.join(legacy_ex))
if miss_i_ex: print('missing_images_examples='+' | '.join(miss_i_ex))
if miss_m_ex: print('missing_masks_examples='+' | '.join(miss_m_ex))
if len(df)==0 or df[[img_col,mask_col]].isna().any().any() or legacy_total or miss_i_total or miss_m_total:
    raise SystemExit(1)
PY
)"; rc=$?; set -e
  printf '%s\n' "$out" | sed 's/^/[INFO] AXIAL /'
  [[ "$rc" -eq 0 ]] && pass "AXIAL CSV y paths" || fail "AXIAL CSV/paths invalidos"
  [[ "$FAIL" -gt "$before" ]] && S_AXIAL=FAIL || S_AXIAL=OK
}
model_import(){ [[ -x "$PFI_VENV_DIR/bin/python" ]] || { fail "venv python faltante para import"; return; }; PYTHONPATH="$PFI_REPO_ROOT/ai_service" "$PFI_VENV_DIR/bin/python" -c 'from pfi_ai_service.model_architectures import SagittalUNet2D, AxialUNet2D, build_checkpoint_model' && pass "imports model_architectures" || fail "imports model_architectures fallan"; }
summary(){
  local result
  [[ "$FAIL" -eq 0 ]] && result=READY || result='NOT READY'
  printf '\n[INFO] Resumen preflight\n'
  info "modo: $MODE"; info "PASS: $PASS WARN: $WARN FAIL: $FAIL"; info "project: $S_PROJECT"; info "VM: $S_VM"; info "zone: $S_ZONE"; info "service account: $S_SA"; info "repo HEAD: $S_HEAD"; info "Python: $S_PY"; info "torch: $S_TORCH"; info "CUDA: $S_CUDA"; info "GPU: $S_GPU"; info "disco disponible: $S_DISK"; info "acceso bucket: $S_BUCKET"; info "SPIDER: $S_SPIDER"; info "AXIAL: $S_AXIAL"
  if [[ "$result" == READY ]]; then printf '[PASS] READY\n'; else printf '[FAIL] NOT READY\n'; fi
}
main(){ load_env; validate_structural_vars; static_checks; if [[ "$MODE" == vm ]]; then vm_system; disk_check; python_vm; nvidia_check; metadata_check; gcloud_check; spider_check; axial_check; model_import; fi; summary; [[ "$FAIL" -eq 0 ]] && exit 0 || exit 1; }
main "$@"