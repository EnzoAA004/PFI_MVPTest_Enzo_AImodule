#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_CHECKOUT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO_CHECKOUT_ROOT" ]] || REPO_CHECKOUT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"

ENV_FILE=""
EXECUTE=0
DRY_RUN=1
SKIP_DOWNLOAD=0
FORCE_PULL_RESUME=0
REQUIRE_RESUME_FLAG=0
LOCK_DIR=""
LOCK_ACQUIRED=0
CHILD_PID=""
EXIT_CODE=0
FINAL_SYNC_ATTEMPTED=0
SYNC_ON_FAILURE_ATTEMPTED=0
STATUS_FILE=""
LOG_FILE=""
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CURRENT_STATUS="starting"
CURRENT_STEP="init"
FAILURE_MESSAGE=""
FINAL_VERIFICATION="null"

log(){ printf '[%s] %s\n' "$1" "$2"; }
info(){ log INFO "$*"; }
pass(){ log PASS "$*"; }
warn(){ log WARN "$*"; }
dry(){ log DRY-RUN "$*"; }
error(){ log ERROR "$*" >&2; }

usage(){ cat <<'USAGE'
Uso: run-final-training.sh [--env-file PATH] [--dry-run|--execute] [--skip-download] [--force-pull-resume] [--require-resume]

Orquesta el entrenamiento final en foreground. Default: --dry-run.
USAGE
}

select_env(){ if [[ -n "$ENV_FILE" ]]; then printf '%s\n' "$ENV_FILE"; elif [[ -f "$SCRIPT_DIR/training-vm.env" ]]; then printf '%s\n' "$SCRIPT_DIR/training-vm.env"; else printf '%s\n' "$SCRIPT_DIR/training-vm.env.example"; fi; }
norm(){ if command -v realpath >/dev/null 2>&1; then realpath -m "$1"; else python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; fi; }
script_dir(){ if [[ "$EXECUTE" -eq 0 && -n "${PFI_RUNNER_STUB_DIR:-}" ]]; then printf '%s\n' "$PFI_RUNNER_STUB_DIR"; else printf '%s\n' "$SCRIPT_DIR"; fi; }
runner_python(){ if [[ -x "$PFI_VENV_DIR/bin/python" ]]; then printf '%s\n' "$PFI_VENV_DIR/bin/python"; elif [[ "$EXECUTE" -eq 0 && -n "${PFI_RUNNER_PYTHON_BIN:-}" ]]; then printf '%s\n' "$PFI_RUNNER_PYTHON_BIN"; elif [[ "$EXECUTE" -eq 0 ]] && command -v python >/dev/null 2>&1; then command -v python; else printf '%s\n' "$PFI_VENV_DIR/bin/python"; fi; }
redact(){ sed -E 's/(access[_]token|private[_]key|client[_]secret|authorization:)[^[:space:]]*/[REDACTED]/Ig; s/BEGIN[[:space:]]+PRIVATE[[:space:]]+KEY/[REDACTED]/Ig; s/BEGIN[[:space:]]+RSA/[REDACTED]/Ig; s/BEGIN[[:space:]]+OPENSSH/[REDACTED]/Ig'; }

invalid_usage(){ error "$1"; usage >&2; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) [[ $# -ge 2 ]] || invalid_usage "--env-file requiere PATH"; ENV_FILE="$2"; shift 2;;
    --dry-run) [[ "$EXECUTE" -eq 0 ]] || invalid_usage "--dry-run y --execute son incompatibles"; DRY_RUN=1; shift;;
    --execute) [[ "$DRY_RUN" -eq 1 ]] || invalid_usage "--dry-run y --execute son incompatibles"; EXECUTE=1; DRY_RUN=0; shift;;
    --skip-download) SKIP_DOWNLOAD=1; shift;;
    --force-pull-resume) FORCE_PULL_RESUME=1; shift;;
    --require-resume) REQUIRE_RESUME_FLAG=1; shift;;
    --help) usage; exit 0;;
    *) invalid_usage "argumento desconocido: $1";;
  esac
done
[[ "$SKIP_DOWNLOAD" -eq 0 || "$EXECUTE" -eq 1 ]] || invalid_usage "--skip-download solo se permite con --execute"
[[ "$FORCE_PULL_RESUME" -eq 0 || "$EXECUTE" -eq 1 ]] || invalid_usage "--force-pull-resume solo se permite con --execute"
[[ "$REQUIRE_RESUME_FLAG" -eq 0 || "$EXECUTE" -eq 1 ]] || invalid_usage "--require-resume solo se permite con --execute"

load_env(){ ENV_FILE="$(select_env)"; [[ -f "$ENV_FILE" ]] || { error "env-file inexistente: $ENV_FILE"; exit 2; }; set -a; source "$ENV_FILE"; set +a; }
need(){ local n="$1"; local -n ref="$n"; [[ -n "${ref-}" ]] || { error "variable requerida ausente: $n"; exit 2; }; }
boolv(){ local n="$1"; local -n ref="$n"; [[ "${ref-}" == 0 || "${ref-}" == 1 ]] || { error "$n debe ser 0/1"; exit 2; }; }
path_ok(){ local n="$1" v root val; local -n ref="$n"; v="${ref-}"; [[ -n "$v" && "$v" == /* && "$v" != / ]] || { error "$n ruta insegura"; exit 2; }; root="$(norm "$PFI_VM_ROOT")"; val="$(norm "$v")"; [[ "$val" == "$root" || "$val" == "$root"/* ]] || { error "$n fuera de PFI_VM_ROOT"; exit 2; }; }
uri_ok(){ local n="$1" u bucket; local -n ref="$n"; u="${ref-}"; bucket="${PFI_GCS_BUCKET_URI%/}"; [[ "$u" == gs://* ]] || { error "$n no es gs://"; exit 2; }; [[ "${u%/}" != "$bucket" ]] || { error "$n no puede ser raiz bucket"; exit 2; }; [[ "${u%/}" == "$bucket"/* ]] || { error "$n fuera de bucket"; exit 2; }; [[ "${u%/}" == *"/$PFI_RUN_ID" || "${u%/}" == *"/$PFI_RUN_ID/"* ]] || { error "$n no contiene PFI_RUN_ID"; exit 2; }; }

validate_env(){
  local v env_norm expected_norm
  for v in PFI_CLOUD_MODE PFI_PREFLIGHT_ONLY PFI_SYNC_DRY_RUN PFI_SYNC_RESUME PFI_SYNC_FINAL_ARTIFACTS PFI_SYNC_FAILURE_IS_FATAL PFI_DOWNLOAD_DATASETS PFI_DOWNLOAD_RESUME PFI_RUN_ID PFI_GCP_PROJECT_ID PFI_GCP_ZONE PFI_VM_NAME PFI_VM_SERVICE_ACCOUNT PFI_REPO_ROOT PFI_VENV_DIR PFI_TRAIN_OUTPUT_DIR PFI_LOCAL_MODELS_DIR PFI_LOCAL_RESUME_DIR PFI_LOCAL_MANIFESTS_DIR PFI_LOCAL_LOGS_DIR PFI_TRAINING_ENV_FILE PFI_GCS_BUCKET_URI PFI_GCS_RUN_MODELS_URI PFI_GCS_RUN_RESUME_URI PFI_GCS_RUN_MANIFESTS_URI PFI_GCS_RUN_OUTPUTS_URI PFI_SYNC_MIN_FILE_AGE_SECONDS RUN_SAGITTAL RUN_AXIAL REQUIRE_GPU; do need "$v"; done
  for v in PFI_CLOUD_MODE PFI_PREFLIGHT_ONLY PFI_SYNC_DRY_RUN PFI_SYNC_RESUME PFI_SYNC_FINAL_ARTIFACTS PFI_SYNC_FAILURE_IS_FATAL PFI_DOWNLOAD_DATASETS PFI_DOWNLOAD_RESUME RUN_SAGITTAL RUN_AXIAL REQUIRE_GPU; do boolv "$v"; done
  [[ "$PFI_RUN_ID" =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]] || { error "PFI_RUN_ID invalido"; exit 2; }
  [[ "$RUN_SAGITTAL" == 1 || "$RUN_AXIAL" == 1 ]] || { error "al menos un plano debe estar activo"; exit 2; }
  [[ "$PFI_SYNC_MIN_FILE_AGE_SECONDS" =~ ^[0-9]+$ ]] || { error "PFI_SYNC_MIN_FILE_AGE_SECONDS invalido"; exit 2; }
  [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]] || { error "GOOGLE_APPLICATION_CREDENTIALS debe estar vacio"; exit 2; }
  for v in PFI_REPO_ROOT PFI_VENV_DIR PFI_TRAIN_OUTPUT_DIR PFI_LOCAL_MODELS_DIR PFI_LOCAL_RESUME_DIR PFI_LOCAL_MANIFESTS_DIR PFI_LOCAL_LOGS_DIR; do path_ok "$v"; done
  for v in PFI_GCS_RUN_MODELS_URI PFI_GCS_RUN_RESUME_URI PFI_GCS_RUN_MANIFESTS_URI PFI_GCS_RUN_OUTPUTS_URI; do uri_ok "$v"; done
  env_norm="$(norm "$ENV_FILE")"; expected_norm="$(norm "$PFI_TRAINING_ENV_FILE")"
  [[ "$env_norm" == "$expected_norm" || "$EXECUTE" -eq 0 ]] || { error "PFI_TRAINING_ENV_FILE no coincide con --env-file"; exit 2; }
}

active_planes_json(){ local items=(); [[ "$RUN_SAGITTAL" == 1 ]] && items+=("sagittal"); [[ "$RUN_AXIAL" == 1 ]] && items+=("axial"); printf '%s\n' "${items[*]}"; }
write_status(){
  local status="$1" step="$2" exit_code="${3:-null}" failure="${4:-}" finished="null" planes commit branch log_name final_json
  CURRENT_STATUS="$status"; CURRENT_STEP="$step"; [[ -n "$failure" ]] && FAILURE_MESSAGE="$failure"
  [[ "$status" == success || "$status" == failed || "$status" == interrupted ]] && finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  [[ -n "${PFI_LOCAL_LOGS_DIR:-}" ]] || return 0
  mkdir -p "$PFI_LOCAL_LOGS_DIR"
  STATUS_FILE="$PFI_LOCAL_LOGS_DIR/runner_status_${PFI_RUN_ID}.json"
  LOG_FILE="$PFI_LOCAL_LOGS_DIR/final_training_${PFI_RUN_ID}.log"
  commit="$(git -C "$REPO_CHECKOUT_ROOT" rev-parse HEAD 2>/dev/null || printf NO_OBTENIDO)"
  branch="$(git -C "$REPO_CHECKOUT_ROOT" branch --show-current 2>/dev/null || printf NO_OBTENIDO)"
  planes="$(active_planes_json)"
  final_json="$FINAL_VERIFICATION"
  PYTHONPATH="$PFI_REPO_ROOT/ai_service" "$(runner_python)" - <<'PY' "$STATUS_FILE" "$PFI_RUN_ID" "$STARTED_AT" "$finished" "$status" "$step" "$$" "$commit" "$branch" "$PFI_VM_NAME" "$PFI_GCP_ZONE" "$planes" "$(basename "$LOG_FILE")" "$exit_code" "$FAILURE_MESSAGE" "$SYNC_ON_FAILURE_ATTEMPTED" "$FINAL_SYNC_ATTEMPTED" "$final_json"
import json, sys
from pathlib import Path
from pfi_ai_service.training.cloud_runtime import atomic_write_json
(path, run_id, started, finished, status, step, pid, commit, branch, vm, zone, planes, log_file, exit_code, failure, sync_fail, final_sync, final_verification) = sys.argv[1:]
data={
 'schema_version':1,'run_id':run_id,'started_at_utc':started,'updated_at_utc':__import__('datetime').datetime.utcnow().replace(microsecond=0).isoformat()+'Z','finished_at_utc':None if finished=='null' else finished,
 'status':status,'current_step':step,'runner_pid':int(pid),'git_commit':commit,'git_branch':branch,'vm_name':vm,'zone':zone,'active_planes':planes.split() if planes else [],'notebook':'train_final_models_v5_cloud_portable.ipynb','log_file':log_file,'exit_code':None if exit_code=='null' else int(exit_code),'failure':None if not failure else failure,'sync_on_failure_attempted':sync_fail=='1','final_sync_attempted':final_sync=='1','final_verification':None if final_verification=='null' else json.loads(final_verification)}
atomic_write_json(data, Path(path))
PY
}

append_log_header(){ mkdir -p "$PFI_LOCAL_LOGS_DIR"; LOG_FILE="$PFI_LOCAL_LOGS_DIR/final_training_${PFI_RUN_ID}.log"; { printf '\n===== nueva invocacion %s =====\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; printf 'commit=%s run_id=%s vm=%s zone=%s\n' "$(git -C "$REPO_CHECKOUT_ROOT" rev-parse HEAD 2>/dev/null || true)" "$PFI_RUN_ID" "$PFI_VM_NAME" "$PFI_GCP_ZONE"; } >> "$LOG_FILE"; }
execute_logged(){ local step="$1"; shift; info "$step: $*"; { printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$step"; "$@" 2>&1 | redact; } | tee -a "$LOG_FILE"; return "${PIPESTATUS[0]}"; }

validate_execute_guards(){
  local v st branch local_head remote_head
  [[ "$EXECUTE" -eq 1 ]] || return 0
  [[ "$PFI_CLOUD_MODE" == 1 && "$PFI_PREFLIGHT_ONLY" == 0 && "$PFI_SYNC_DRY_RUN" == 0 && "$PFI_SYNC_RESUME" == 1 && "$PFI_SYNC_FINAL_ARTIFACTS" == 1 && "$PFI_SYNC_FAILURE_IS_FATAL" == 1 && "$REQUIRE_GPU" == 1 ]] || { error "guards --execute invalidos"; exit 2; }
  [[ "$(uname -s)" == Linux ]] || { error "--execute requiere Linux"; exit 2; }
  [[ "$(id -u)" -ne 0 ]] || { error "--execute no debe correr como root"; exit 2; }
  [[ -x "$PFI_VENV_DIR/bin/python" ]] || { error "venv python faltante"; exit 2; }
  [[ -f "$PFI_REPO_ROOT/notebooks/train_final_models_v5_cloud_portable.ipynb" ]] || { error "notebook v5 faltante"; exit 2; }
  [[ -x "$SCRIPT_DIR/sync-training-artifacts.sh" && -x "$SCRIPT_DIR/download-training-data.sh" && -x "$SCRIPT_DIR/preflight-training-vm.sh" ]] || { error "scripts GCP faltantes/no ejecutables"; exit 2; }
  [[ -d "$PFI_REPO_ROOT/.git" ]] || { error "PFI_REPO_ROOT no es repo Git"; exit 2; }
  branch="$(git -C "$PFI_REPO_ROOT" branch --show-current)"; [[ "$branch" == main ]] || { error "rama debe ser main"; exit 2; }
  st="$(git -C "$PFI_REPO_ROOT" status --porcelain)"; [[ -z "$st" ]] || { error "worktree dirty"; exit 2; }
  if git -C "$PFI_REPO_ROOT" rev-parse --verify origin/main >/dev/null 2>&1; then local_head="$(git -C "$PFI_REPO_ROOT" rev-parse HEAD)"; remote_head="$(git -C "$PFI_REPO_ROOT" rev-parse origin/main)"; [[ "$local_head" == "$remote_head" ]] || { error "HEAD local difiere de origin/main"; exit 2; }; fi
}

metadata_get(){ curl -fsS --max-time 2 -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/$1"; }
verify_gce_identity(){ [[ "$EXECUTE" -eq 1 ]] || return 0; local p i z e; p="$(metadata_get project/project-id 2>/dev/null || true)"; i="$(metadata_get instance/name 2>/dev/null || true)"; z="$(metadata_get instance/zone 2>/dev/null || true)"; e="$(metadata_get instance/service-accounts/default/email 2>/dev/null || true)"; [[ "$p" == "$PFI_GCP_PROJECT_ID" && "$i" == "$PFI_VM_NAME" && "${z##*/}" == "$PFI_GCP_ZONE" && "$e" == "$PFI_VM_SERVICE_ACCOUNT" ]] || { error "metadata GCE no coincide"; exit 1; }; }

acquire_lock(){ [[ "$EXECUTE" -eq 1 ]] || return 0; LOCK_DIR="/tmp/pfi-final-training-${PFI_RUN_ID}.lock"; if mkdir "$LOCK_DIR" 2>/dev/null; then LOCK_ACQUIRED=1; printf '%s\n' "$$" > "$LOCK_DIR/pid"; date -u +%Y-%m-%dT%H:%M:%SZ > "$LOCK_DIR/created_at_utc"; return; fi; if [[ -f "$LOCK_DIR/pid" ]]; then local pid; pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"; if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then error "lock activo para run_id $PFI_RUN_ID pid=$pid"; exit 1; fi; fi; error "lock stale existente; revisar manualmente: $LOCK_DIR"; exit 1; }
release_lock(){ if [[ "$LOCK_ACQUIRED" -eq 1 && -n "$LOCK_DIR" && "$LOCK_DIR" == /tmp/pfi-final-training-*.lock && -d "$LOCK_DIR" ]]; then rm -f "$LOCK_DIR/pid" "$LOCK_DIR/created_at_utc"; rmdir "$LOCK_DIR" 2>/dev/null || true; fi; }

count_resume(){ find "$PFI_LOCAL_RESUME_DIR" -maxdepth 1 -type f \( -name '*.last_checkpoint.pt' -o -name '*.best_checkpoint.pt' \) 2>/dev/null | wc -l | tr -d ' '; }
sync_resume_on_failure(){ [[ "$EXECUTE" -eq 1 ]] || return 0; [[ "$SYNC_ON_FAILURE_ATTEMPTED" -eq 0 ]] || return 0; [[ "$(count_resume)" -gt 0 ]] || { warn "sin checkpoints locales para push-resume"; return 0; }; SYNC_ON_FAILURE_ATTEMPTED=1; write_status interrupted push-resume null "$FAILURE_MESSAGE" || true; sleep "$PFI_SYNC_MIN_FILE_AGE_SECONDS"; execute_logged "sync push-resume" bash "$SCRIPT_DIR/sync-training-artifacts.sh" --mode push-resume --execute --env-file "$ENV_FILE" || warn "push-resume fallo; se conserva checkpoint local"; }

handle_int(){ FAILURE_MESSAGE="SIGINT recibido"; [[ -n "$CHILD_PID" ]] && kill -INT "$CHILD_PID" 2>/dev/null || true; [[ -n "$CHILD_PID" ]] && wait "$CHILD_PID" 2>/dev/null || true; sync_resume_on_failure; write_status interrupted sigint 130 "$FAILURE_MESSAGE" || true; release_lock; exit 130; }
handle_term(){ FAILURE_MESSAGE="SIGTERM recibido"; [[ -n "$CHILD_PID" ]] && kill -TERM "$CHILD_PID" 2>/dev/null || true; sleep 5; [[ -n "$CHILD_PID" ]] && kill -TERM "$CHILD_PID" 2>/dev/null || true; [[ -n "$CHILD_PID" ]] && wait "$CHILD_PID" 2>/dev/null || true; sync_resume_on_failure; write_status interrupted sigterm 143 "$FAILURE_MESSAGE" || true; release_lock; exit 143; }
on_exit(){ local code=$?; if [[ "$code" -ne 0 && -n "$STATUS_FILE" && "$CURRENT_STATUS" != interrupted && "$CURRENT_STATUS" != failed ]]; then write_status failed "$CURRENT_STEP" "$code" "runner exit $code" || true; fi; release_lock; }
trap handle_int INT
trap handle_term TERM
trap on_exit EXIT

run_preflight_static(){ local dir; dir="$(script_dir)"; execute_logged "preflight static" bash "$dir/preflight-training-vm.sh" --mode static --env-file "$ENV_FILE"; }
run_download(){ local dir; dir="$(script_dir)"; [[ "$SKIP_DOWNLOAD" -eq 1 ]] && { info "download skip solicitado"; return 0; }; if [[ "$EXECUTE" -eq 1 ]]; then execute_logged "download datasets" bash "$dir/download-training-data.sh" --component datasets --execute --env-file "$ENV_FILE"; else execute_logged "download datasets dry-run" bash "$dir/download-training-data.sh" --component datasets --dry-run --env-file "$ENV_FILE"; fi; }
restore_resume(){ local dir local_count; dir="$(script_dir)"; local_count="$(count_resume)"; if [[ "$EXECUTE" -eq 0 ]]; then [[ "$PFI_DOWNLOAD_RESUME" == 1 ]] && execute_logged "pull-resume dry-run" bash "$dir/sync-training-artifacts.sh" --mode pull-resume --dry-run --env-file "$ENV_FILE" || info "pull-resume dry-run skip"; return 0; fi; if [[ "$FORCE_PULL_RESUME" -eq 1 ]]; then warn "--force-pull-resume: se restaurara remoto sin borrar checkpoints locales"; execute_logged "pull-resume forced" bash "$SCRIPT_DIR/sync-training-artifacts.sh" --mode pull-resume --execute --env-file "$ENV_FILE"; elif [[ "$local_count" -gt 0 ]]; then info "Resume local presente; se preserva para evitar reemplazar progreso local con una copia remota anterior."; elif [[ "$PFI_DOWNLOAD_RESUME" == 1 ]]; then execute_logged "pull-resume" bash "$SCRIPT_DIR/sync-training-artifacts.sh" --mode pull-resume --execute --env-file "$ENV_FILE"; else info "pull-resume SKIP por PFI_DOWNLOAD_RESUME=0"; fi; [[ "$REQUIRE_RESUME_FLAG" -eq 0 || "$(count_resume)" -gt 0 ]] || { error "--require-resume solicitado pero no hay checkpoints"; exit 1; }; }
run_preflight_vm(){ [[ "$EXECUTE" -eq 1 ]] || return 0; execute_logged "preflight vm" bash "$SCRIPT_DIR/preflight-training-vm.sh" --mode vm --env-file "$ENV_FILE"; }
model_names(){ [[ "$RUN_SAGITTAL" == 1 ]] && printf '%s\n' sagittal_spider_multiclass_final_best.pt; [[ "$RUN_AXIAL" == 1 ]] && printf '%s\n' axial_t2_alkafri_final_best.pt; }
check_completed_local(){ local missing=0 name; while IFS= read -r name; do [[ -f "$PFI_LOCAL_MODELS_DIR/$name" ]] || missing=1; done < <(model_names); [[ "$missing" -eq 0 && -f "$PFI_LOCAL_MANIFESTS_DIR/training_run_${PFI_RUN_ID}.json" ]]; }
remote_exists(){ gcloud storage ls "$1" >/dev/null 2>&1; }
check_completed_remote(){ [[ "$EXECUTE" -eq 1 ]] || return 1; local name; while IFS= read -r name; do remote_exists "${PFI_GCS_RUN_MODELS_URI%/}/$name" || return 1; done < <(model_names); remote_exists "${PFI_GCS_RUN_MANIFESTS_URI%/}/training_run_${PFI_RUN_ID}.json"; }
prevent_completed_run(){ if check_completed_local || check_completed_remote; then error "corrida ya completada; usar PFI_RUN_ID nuevo"; exit 1; fi; }
run_notebook(){ local output; output="$PFI_LOCAL_LOGS_DIR/train_final_models_v5_${PFI_RUN_ID}.executed.ipynb"; write_status training notebook; info "notebook foreground iniciado"; PYTHONPATH="$PFI_REPO_ROOT/ai_service" "$PFI_VENV_DIR/bin/python" -m pfi_ai_service.training.notebook_executor --input "$PFI_REPO_ROOT/notebooks/train_final_models_v5_cloud_portable.ipynb" --output "$output" --log "$LOG_FILE" --kernel-python "$PFI_VENV_DIR/bin/python" --timeout 0 & CHILD_PID=$!; wait "$CHILD_PID"; local rc=$?; CHILD_PID=""; return "$rc"; }
verify_local_final(){ local name; while IFS= read -r name; do [[ -f "$PFI_LOCAL_MODELS_DIR/$name" ]] || { error "falta modelo final local: $name"; exit 1; }; done < <(model_names); [[ -f "$PFI_LOCAL_MANIFESTS_DIR/training_run_${PFI_RUN_ID}.json" ]] || { error "falta training_run manifest local"; exit 1; }; }
final_sync(){ FINAL_SYNC_ATTEMPTED=1; write_status syncing push-all; sleep "$PFI_SYNC_MIN_FILE_AGE_SECONDS"; execute_logged "sync push-all" bash "$SCRIPT_DIR/sync-training-artifacts.sh" --mode push-all --execute --env-file "$ENV_FILE"; }
verify_remote_final(){ local missing=0 name; for name in $(model_names); do if remote_exists "${PFI_GCS_RUN_MODELS_URI%/}/$name"; then pass "remoto OK $name"; else missing=1; fi; done; remote_exists "${PFI_GCS_RUN_RESUME_URI%/}/" || missing=1; remote_exists "${PFI_GCS_RUN_MANIFESTS_URI%/}/training_run_${PFI_RUN_ID}.json" || missing=1; remote_exists "${PFI_GCS_RUN_OUTPUTS_URI%/}/logs/final_training_${PFI_RUN_ID}.log" || missing=1; remote_exists "${PFI_GCS_RUN_OUTPUTS_URI%/}/logs/runner_status_${PFI_RUN_ID}.json" || missing=1; FINAL_VERIFICATION="{\"checked\":true,\"missing\":$missing}"; [[ "$missing" -eq 0 ]] || { error "verificacion remota incompleta; no borrar VM"; exit 1; }; }

main(){
  load_env; validate_env; append_log_header; validate_execute_guards; write_status starting init
  info "commit=$(git -C "$REPO_CHECKOUT_ROOT" rev-parse --short HEAD 2>/dev/null || true) branch=$(git -C "$REPO_CHECKOUT_ROOT" branch --show-current 2>/dev/null || true) run_id=$PFI_RUN_ID vm=$PFI_VM_NAME zone=$PFI_GCP_ZONE planes=$(active_planes_json)"
  if [[ "$EXECUTE" -eq 0 ]]; then dry "default dry-run; no entrenamiento, no GCS real si se usan stubs"; write_status preflight static; run_preflight_static; write_status downloading dry-run; run_download; write_status restoring_resume dry-run; restore_resume; PYTHONPATH="$PFI_REPO_ROOT/ai_service" "$(runner_python)" -m pfi_ai_service.training.notebook_executor --help >/dev/null; pass "notebook_executor CLI"; write_status success dry-run 0; pass "SUCCESS dry-run"; return 0; fi
  acquire_lock; verify_gce_identity; write_status preflight static; run_preflight_static; write_status downloading datasets; run_download; write_status restoring_resume resume; restore_resume; write_status preflight vm; run_preflight_vm; prevent_completed_run
  if run_notebook; then verify_local_final; final_sync; verify_remote_final; write_status success done 0; pass "SUCCESS execute"; return 0; else EXIT_CODE=$?; FAILURE_MESSAGE="notebook fallo rc=$EXIT_CODE"; sync_resume_on_failure; write_status failed notebook "$EXIT_CODE" "$FAILURE_MESSAGE"; return "$EXIT_CODE"; fi
}
main "$@"
