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
S_BUCKET=NO_OBTENIDO; S_SPIDER=NO_OBTENIDO; S_AXIAL=NO_OBTENIDO; S_RUN_ID=NO_OBTENIDO
STRUCTURAL_VARS=(PFI_VM_ROOT PFI_REPO_ROOT PFI_DATA_ROOT PFI_OUTPUT_ROOT PFI_TRAIN_OUTPUT_DIR PFI_VENV_DIR PFI_GCS_BUCKET_URI PFI_GCP_PROJECT_ID PFI_GCP_ZONE PFI_VM_NAME PFI_VM_SERVICE_ACCOUNT)
ALL_VARS=(
  PFI_GCP_PROJECT_ID PFI_GCP_REGION PFI_GCP_ZONE PFI_VM_NAME PFI_VM_SERVICE_ACCOUNT
  PFI_GCS_BUCKET_URI PFI_GCS_DATASETS_URI PFI_GCS_SPIDER_URI PFI_GCS_AXIAL_URI PFI_GCS_METADATA_URI PFI_GCS_BOOTSTRAP_MODELS_URI PFI_GCS_MODELS_URI PFI_GCS_RESUME_URI PFI_GCS_MANIFESTS_URI PFI_GCS_OUTPUTS_URI PFI_GCS_RUN_MODELS_URI PFI_GCS_RUN_RESUME_URI PFI_GCS_RUN_MANIFESTS_URI PFI_GCS_RUN_OUTPUTS_URI
  PFI_VM_ROOT PFI_REPO_URL PFI_REPO_ROOT PFI_DATA_ROOT PFI_OUTPUT_ROOT PFI_TRAIN_OUTPUT_DIR
  PFI_LOCAL_RESUME_DIR PFI_LOCAL_MODELS_DIR PFI_LOCAL_MANIFESTS_DIR PFI_LOCAL_LOGS_DIR PFI_PYTHON_BIN PFI_VENV_DIR
  PFI_RUN_ID PFI_MIN_FREE_DISK_GB REQUIRE_GPU RUN_SAGITTAL RUN_AXIAL PFI_SMOKE_RUN PFI_RESUME_TRAINING
  PFI_MAX_EPOCHS PFI_EARLY_STOP_PATIENCE PFI_PREFLIGHT_ONLY PFI_SYNC_DRY_RUN PFI_DOWNLOAD_DATASETS PFI_DOWNLOAD_RESUME PFI_SYNC_RESUME PFI_SYNC_FINAL_ARTIFACTS PFI_SYNC_MIN_FILE_AGE_SECONDS PFI_CLOUD_MODE PFI_USE_GOOGLE_DRIVE PFI_INSTALL_NOTEBOOK_DEPS PFI_TRAINING_ENV_FILE PFI_SYNC_EVERY_N_EPOCHS PFI_SYNC_FAILURE_IS_FATAL
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
validate_structural_vars(){ local missing=() v; for v in "${STRUCTURAL_VARS[@]}"; do [[ -n "${!v-}" ]] || missing+=("$v"); done; if [[ ${#missing[@]} -gt 0 ]]; then printf '[FAIL] variables estructurales faltantes: %s\n' "${missing[*]}" >&2; exit 2; fi; }
req(){ local n="$1"; local -n ref="$n"; [[ -n "${ref-}" ]] && pass "var $n" || fail "variable ausente: $n"; }
cmd(){ command -v "$1" >/dev/null 2>&1; }
need_cmd(){ cmd "$1" && pass "comando $1" || fail "comando faltante: $1"; }
norm(){ if command -v realpath >/dev/null 2>&1; then realpath -m "$1"; else printf '%s\n' "$1"; fi; }
path_ok(){ local n="$1" v r pv; local -n ref="$n"; v="${ref-}"; [[ -n "$v" && "$v" != / && "$v" == /* ]] || { fail "$n ruta invalida: ${v:-AUSENTE}"; return; }; r="$(norm "$PFI_VM_ROOT")"; pv="$(norm "$v")"; if [[ "$n" != PFI_VM_ROOT && "$pv" != "$r" && "$pv" != "$r"/* ]]; then fail "$n fuera de PFI_VM_ROOT"; else pass "$n ruta segura"; fi; }
eqv(){ local n="$1" e="$2"; local -n ref="$n"; [[ "${ref-}" == "$e" ]] && pass "$n=$e" || fail "$n esperado $e obtenido ${ref:-AUSENTE}"; }
boolv(){ local n="$1"; local -n ref="$n"; local v="${ref-}"; [[ "$v" == 0 || "$v" == 1 ]] && pass "$n booleano" || fail "$n debe ser 0/1"; }
compile_python_readonly(){ local path="$1" py=python; cmd python || py=python3; "$py" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); compile(p.read_text(encoding="utf-8"), str(p), "exec")' "$path"; }
worktree_check(){
  local status
  status="$(git -C "$REPO_CHECKOUT_ROOT" status --porcelain)"
  if [[ -z "$status" ]]; then pass "worktree clean"; else warn "worktree dirty"; printf '%s\n' "$status" | sed 's/^/[WARN] git status: /'; fi
}
validate_v5_notebook(){
  local py="${1:-python}" nb="$REPO_CHECKOUT_ROOT/notebooks/train_final_models_v5_cloud_portable.ipynb"
  "$py" - <<'PY' "$nb"
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); nb=json.loads(p.read_text(encoding='utf-8'))
assert nb.get('nbformat') == 4
text='\n'.join((''.join(c.get('source','')) if isinstance(c.get('source'),list) else c.get('source','')) for c in nb.get('cells',[]))
for i,c in enumerate(nb.get('cells',[])):
    if c.get('cell_type') == 'code':
        assert c.get('outputs', []) == [], f'outputs no vacios cell {i}'
        assert c.get('execution_count') is None, f'execution_count no null cell {i}'
        compile(''.join(c.get('source','')) if isinstance(c.get('source'),list) else c.get('source',''), f'{p}:cell{i}', 'exec')
for bad in ['!pip','%pip','shutil.rmtree','child.unlink','gcloud storage rm','gcloud storage mv','delete-unmatched-destination-objects','auth print-' + 'access-token','shell=True','GOOGLE_APPLICATION_' + 'CREDENTIALS=']:
    assert bad not in text, bad
for needed in ['atomic_torch_save','PFI_PREFLIGHT_ONLY','PFI_RUN_ID','PFI_LOCAL_MODELS_DIR','PFI_LOCAL_RESUME_DIR','PFI_LOCAL_MANIFESTS_DIR','push-resume','push-final']:
    assert needed in text, needed
assert 'torch.save(' not in text
PY
}
validate_helper_static(){
  local py="${1:-python}" helper="$REPO_CHECKOUT_ROOT/ai_service/pfi_ai_service/training/cloud_runtime.py" executor="$REPO_CHECKOUT_ROOT/ai_service/pfi_ai_service/training/notebook_executor.py" runner="$REPO_CHECKOUT_ROOT/infra/gcp/run-final-training.sh"
  "$py" - <<'PY' "$helper" "$executor" "$runner"
from pathlib import Path
import sys
p=Path(sys.argv[1]); executor=Path(sys.argv[2]); runner=Path(sys.argv[3]); text=p.read_text(encoding='utf-8')
compile(text, str(p), 'exec')
for bad in ['shell=True','gcloud storage','auth print-' + 'access-token','credentials' + '.json','rm -rf']:
    assert bad not in text, bad
exec_text=Path(executor).read_text(encoding='utf-8')
compile(exec_text, str(executor), 'exec')
for needed in ['nbclient','nbformat','NotebookClient']:
    assert needed in exec_text, needed
for bad in ['shell=True','nbconvert','exec(','gcloud','CUDA','torch.']:
    assert bad not in exec_text, bad
runner_text=Path(runner).read_text(encoding='utf-8')
for needed in ['--execute','--dry-run','preflight-training-vm.sh','download-training-data.sh','sync-training-artifacts.sh','notebook_executor','push-resume','push-all','PFI_PREFLIGHT_ONLY','PFI_SYNC_DRY_RUN']:
    assert needed in runner_text, needed
for bad in ['gcloud storage rm','gcloud storage mv','gsutil','delete-unmatched-destination-objects','auth print-' + 'access-token','git reset','git clean','git pull','sudo','pip install','rm -rf']:
    assert bad not in runner_text, bad
PY
}
static_checks(){
  local n f py=python
  validate_structural_vars
  for n in "${ALL_VARS[@]}"; do req "$n"; done
  for n in PFI_VM_ROOT PFI_REPO_ROOT PFI_DATA_ROOT PFI_OUTPUT_ROOT PFI_TRAIN_OUTPUT_DIR PFI_LOCAL_RESUME_DIR PFI_LOCAL_MODELS_DIR PFI_LOCAL_MANIFESTS_DIR PFI_LOCAL_LOGS_DIR PFI_VENV_DIR; do path_ok "$n"; done
  eqv PFI_MAX_EPOCHS 80; eqv PFI_EARLY_STOP_PATIENCE 12; eqv PFI_RESUME_TRAINING 1; eqv REQUIRE_GPU 1; eqv PFI_SMOKE_RUN 0
  for n in RUN_SAGITTAL RUN_AXIAL PFI_PREFLIGHT_ONLY PFI_SYNC_DRY_RUN PFI_DOWNLOAD_DATASETS PFI_DOWNLOAD_RESUME PFI_SYNC_RESUME PFI_SYNC_FINAL_ARTIFACTS PFI_CLOUD_MODE PFI_USE_GOOGLE_DRIVE PFI_INSTALL_NOTEBOOK_DEPS PFI_SYNC_FAILURE_IS_FATAL; do boolv "$n"; done
  [[ "$PFI_GCS_BUCKET_URI" == gs://* ]] && pass "bucket gs://" || fail "bucket no usa gs://"
  [[ "$PFI_RUN_ID" =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]] && { pass "PFI_RUN_ID valido"; S_RUN_ID="$PFI_RUN_ID"; } || fail "PFI_RUN_ID invalido"
  eqv PFI_PREFLIGHT_ONLY 1; eqv PFI_SYNC_DRY_RUN 1; eqv PFI_CLOUD_MODE 1; eqv PFI_USE_GOOGLE_DRIVE 0; eqv PFI_INSTALL_NOTEBOOK_DEPS 0
  [[ "$PFI_SYNC_MIN_FILE_AGE_SECONDS" =~ ^[0-9]+$ ]] && pass "PFI_SYNC_MIN_FILE_AGE_SECONDS entero" || fail "PFI_SYNC_MIN_FILE_AGE_SECONDS invalido"
  [[ "$PFI_SYNC_EVERY_N_EPOCHS" =~ ^[0-9]+$ && "$PFI_SYNC_EVERY_N_EPOCHS" -ge 1 ]] && pass "PFI_SYNC_EVERY_N_EPOCHS entero >=1" || fail "PFI_SYNC_EVERY_N_EPOCHS invalido"
  [[ "$PFI_TRAINING_ENV_FILE" == /* && "$PFI_TRAINING_ENV_FILE" == "$PFI_REPO_ROOT"/* ]] && pass "PFI_TRAINING_ENV_FILE dentro de PFI_REPO_ROOT" || fail "PFI_TRAINING_ENV_FILE inseguro"
  local uri bucket
  bucket="${PFI_GCS_BUCKET_URI%/}"
  for n in PFI_GCS_SPIDER_URI PFI_GCS_AXIAL_URI PFI_GCS_METADATA_URI PFI_GCS_BOOTSTRAP_MODELS_URI PFI_GCS_RUN_MODELS_URI PFI_GCS_RUN_RESUME_URI PFI_GCS_RUN_MANIFESTS_URI PFI_GCS_RUN_OUTPUTS_URI; do
    uri="${!n-}"
    [[ "$uri" == "$bucket"/* && "${uri%/}" != "$bucket" ]] && pass "$n bajo bucket" || fail "$n insegura"
  done
  for n in PFI_GCS_RUN_MODELS_URI PFI_GCS_RUN_RESUME_URI PFI_GCS_RUN_MANIFESTS_URI PFI_GCS_RUN_OUTPUTS_URI; do
    [[ "${!n}" == *"/$PFI_RUN_ID" || "${!n}" == *"/$PFI_RUN_ID/"* ]] && pass "$n contiene run_id" || fail "$n no contiene run_id"
  done
  [[ "$PFI_VM_SERVICE_ACCOUNT" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.iam\.gserviceaccount\.com$ ]] && pass "service account formato valido" || fail "service account formato invalido"
  for f in notebooks/train_final_models_v4_final.ipynb notebooks/train_final_models_v5_cloud_portable.ipynb ai_service/pfi_ai_service/model_architectures.py ai_service/pfi_ai_service/training/__init__.py ai_service/pfi_ai_service/training/cloud_runtime.py ai_service/tests/test_cloud_training_runtime.py ai_service/pfi_ai_service/training/notebook_executor.py ai_service/tests/test_notebook_executor.py ai_service/tests/test_final_training_runner.py infra/gcp/run-final-training.sh requirements.txt infra/gcp/create-pfi-training-t4-v1.sh infra/gcp/download-training-data.sh infra/gcp/sync-training-artifacts.sh; do [[ -f "$REPO_CHECKOUT_ROOT/$f" ]] && pass "existe $f" || fail "falta $f"; done
  cmd python || py=python3
  if cmd "$py"; then
    "$py" -m json.tool "$REPO_CHECKOUT_ROOT/notebooks/train_final_models_v4_final.ipynb" >/dev/null && pass "notebook v4 JSON valido" || fail "notebook v4 JSON invalido"
    compile_python_readonly "$REPO_CHECKOUT_ROOT/ai_service/pfi_ai_service/model_architectures.py" && pass "model_architectures.py compila sin bytecode" || fail "model_architectures.py no compila"
    validate_v5_notebook "$py" && pass "notebook v5 portable valido" || fail "notebook v5 invalido"
    validate_helper_static "$py" && pass "cloud_runtime.py compila y pasa safety" || fail "cloud_runtime.py invalido"
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
    for f in infra/gcp/download-training-data.sh infra/gcp/sync-training-artifacts.sh infra/gcp/run-final-training.sh; do
    bash -n "$REPO_CHECKOUT_ROOT/$f" && pass "$f sintaxis bash" || fail "$f sintaxis bash"
    mode="$(git -C "$REPO_CHECKOUT_ROOT" ls-files --stage "$f" | awk '{print $1}')"
    [[ "$mode" == 100755 ]] && pass "$f modo 100755" || fail "$f no esta trackeado como 100755"
  done
  if grep -RE -- '--delete-unmatched-destination-objects\|gcloud storage rm\|gcloud storage mv\|gsutil' "$REPO_CHECKOUT_ROOT/infra/gcp/download-training-data.sh" "$REPO_CHECKOUT_ROOT/infra/gcp/sync-training-artifacts.sh" >/dev/null; then fail "scripts contienen comandos prohibidos"; else pass "scripts sin comandos GCS prohibidos"; fi
  grep -q -- '--execute' "$REPO_CHECKOUT_ROOT/infra/gcp/download-training-data.sh" && grep -q -- '--execute' "$REPO_CHECKOUT_ROOT/infra/gcp/sync-training-artifacts.sh" && pass "scripts requieren --execute" || fail "falta control --execute"
  bash -n "$REPO_CHECKOUT_ROOT/infra/gcp/run-final-training.sh" && pass "runner sintaxis bash" || fail "runner sintaxis bash"
  mode="$(git -C "$REPO_CHECKOUT_ROOT" ls-files --stage infra/gcp/run-final-training.sh | awk '{print $1}')"
  [[ "$mode" == 100755 ]] && pass "runner modo 100755" || fail "runner no esta trackeado como 100755"
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
mods=['numpy','scipy','pandas','matplotlib','skimage','sklearn','torch','torchvision','pytest','SimpleITK','pydicom','PIL','nbclient','nbformat','jupyter_client','nbconvert','jupyter','ipykernel']
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
  out="$(PYTHONPATH="$PFI_REPO_ROOT/ai_service" "$PFI_VENV_DIR/bin/python" - <<'PY' "$AXIAL_E9_CURATED_SPLIT_CSV" "$AXIAL_IMAGES_DIR" "$AXIAL_MASKS_DIR" "$AXIAL_IMAGE_COL" "$AXIAL_MASK_COL" "$AXIAL_PATIENT_COL" "$AXIAL_SPLIT_COL"
import sys
from pathlib import Path
import pandas as pd
from pfi_ai_service.training.cloud_runtime import PortablePathError, resolve_portable_axial_path
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
missing_images=[]; missing_masks=[]; rebased=0; direct=0
for _, row in df.iterrows():
    try:
        img=resolve_portable_axial_path(row[img_col], Path(img_root), Path(csv).parent)
        msk=resolve_portable_axial_path(row[mask_col], Path(mask_root), Path(csv).parent)
    except PortablePathError as exc:
        print(f'path_error={exc}'); raise SystemExit(1)
    for original, resolved in [(str(row[img_col]), img), (str(row[mask_col]), msk)]:
        if Path(original).as_posix() == resolved.as_posix():
            direct += 1
        else:
            rebased += 1
    if not img.exists() and len(missing_images) < 10:
        missing_images.append(str(img))
    if not msk.exists() and len(missing_masks) < 10:
        missing_masks.append(str(msk))
print(f'portable_paths_direct={direct}')
print(f'portable_paths_rebased={rebased}')
print(f'missing_images_total={len(missing_images)}')
print(f'missing_masks_total={len(missing_masks)}')
if missing_images: print('missing_images_examples='+' | '.join(missing_images))
if missing_masks: print('missing_masks_examples='+' | '.join(missing_masks))
if len(df)==0 or df[[img_col,mask_col]].isna().any().any() or missing_images or missing_masks:
    raise SystemExit(1)
PY
)"; rc=$?; set -e
  printf '%s\n' "$out" | sed 's/^/[INFO] AXIAL /'
  [[ "$rc" -eq 0 ]] && pass "AXIAL CSV y paths" || fail "AXIAL CSV/paths invalidos"
  [[ "$FAIL" -gt "$before" ]] && S_AXIAL=FAIL || S_AXIAL=OK
}
model_import(){ [[ -x "$PFI_VENV_DIR/bin/python" ]] || { fail "venv python faltante para import"; return; }; PYTHONPATH="$PFI_REPO_ROOT/ai_service" "$PFI_VENV_DIR/bin/python" -c 'from pfi_ai_service.model_architectures import SagittalUNet2D, AxialUNet2D, build_checkpoint_model; import nbclient, nbformat, jupyter_client, ipykernel; import pfi_ai_service.training.notebook_executor' && pass "imports model_architectures/notebook_executor" || fail "imports model_architectures/notebook_executor fallan"; }
summary(){
  local result
  [[ "$FAIL" -eq 0 ]] && result=READY || result='NOT READY'
  printf '\n[INFO] Resumen preflight\n'
  info "modo: $MODE"; info "PASS: $PASS WARN: $WARN FAIL: $FAIL"; info "project: $S_PROJECT"; info "VM: $S_VM"; info "zone: $S_ZONE"; info "service account: $S_SA"; info "repo HEAD: $S_HEAD"; info "Python: $S_PY"; info "torch: $S_TORCH"; info "CUDA: $S_CUDA"; info "GPU: $S_GPU"; info "disco disponible: $S_DISK"; info "acceso bucket: $S_BUCKET"; info "SPIDER: $S_SPIDER"; info "AXIAL: $S_AXIAL"; info "run ID: $S_RUN_ID"
  if [[ "$result" == READY ]]; then printf '[PASS] READY\n'; else printf '[FAIL] NOT READY\n'; fi
}
main(){ load_env; validate_structural_vars; static_checks; if [[ "$MODE" == vm ]]; then vm_system; disk_check; python_vm; nvidia_check; metadata_check; gcloud_check; spider_check; axial_check; model_import; fi; summary; [[ "$FAIL" -eq 0 ]] && exit 0 || exit 1; }
main "$@"