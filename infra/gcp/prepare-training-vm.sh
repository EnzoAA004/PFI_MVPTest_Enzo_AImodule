#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_CHECKOUT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_CHECKOUT_ROOT" ]]; then
  REPO_CHECKOUT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
fi
ENV_FILE=""; DRY_RUN=0; SKIP_APT=0; SKIP_PYTHON=0; APT_UPDATED=0
log(){ printf '[%s] %s\n' "$1" "$2"; }
info(){ log INFO "$*"; }; warn(){ log WARN "$*"; }; err(){ log ERROR "$*" >&2; }; dry(){ log DRY-RUN "$*"; }
trap 'err "fallo en linea $LINENO: $BASH_COMMAND"' ERR
usage(){ cat <<'USAGE'
Uso: prepare-training-vm.sh [--env-file PATH] [--dry-run] [--skip-apt] [--skip-python]
Prepara filesystem, repo y venv de la futura VM. No descarga datasets, no entrena,
no ejecuta notebooks, no usa Docker, no modifica Google Cloud y no instala torch/CUDA/drivers.
USAGE
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) [[ $# -ge 2 ]] || { err "--env-file requiere PATH"; exit 2; }; ENV_FILE="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --skip-apt) SKIP_APT=1; shift;;
    --skip-python) SKIP_PYTHON=1; shift;;
    --help) usage; exit 0;;
    *) err "argumento desconocido: $1"; usage >&2; exit 2;;
  esac
done
select_env(){ if [[ -n "$ENV_FILE" ]]; then printf '%s\n' "$ENV_FILE"; elif [[ -f "$SCRIPT_DIR/training-vm.env" ]]; then printf '%s\n' "$SCRIPT_DIR/training-vm.env"; else printf '%s\n' "$SCRIPT_DIR/training-vm.env.example"; fi; }
load_env(){ ENV_FILE="$(select_env)"; [[ -f "$ENV_FILE" ]] || { err "env-file inexistente: $ENV_FILE"; exit 2; }; info "cargando $ENV_FILE"; set -a; source "$ENV_FILE"; set +a; }
req(){ local n="$1"; [[ -n "${!n:-}" ]] || { err "variable requerida ausente: $n"; exit 2; }; }
abs(){ [[ "$1" == /* ]]; }
norm(){ python3 -c 'import os,sys; print(os.path.normpath(sys.argv[1]))' "$1" 2>/dev/null || printf '%s\n' "$1"; }
check_path(){ local n="$1" v="${!n:-}" r pv; [[ -n "$v" && "$v" != "/" ]] || { err "$n invalida"; exit 2; }; abs "$v" || { err "$n debe ser absoluta: $v"; exit 2; }; r="$(norm "$PFI_VM_ROOT")"; pv="$(norm "$v")"; if [[ "$n" != PFI_VM_ROOT && "$pv" != "$r" && "$pv" != "$r"/* ]]; then err "$n debe estar dentro de PFI_VM_ROOT: $v"; exit 2; fi; }
validate_env(){ local vars=(PFI_VM_ROOT PFI_REPO_URL PFI_REPO_ROOT PFI_DATA_ROOT PFI_OUTPUT_ROOT PFI_TRAIN_OUTPUT_DIR PFI_LOCAL_RESUME_DIR PFI_LOCAL_MODELS_DIR PFI_LOCAL_MANIFESTS_DIR PFI_LOCAL_LOGS_DIR PFI_PYTHON_BIN PFI_VENV_DIR); local n; for n in "${vars[@]}"; do req "$n"; done; for n in PFI_VM_ROOT PFI_REPO_ROOT PFI_DATA_ROOT PFI_OUTPUT_ROOT PFI_TRAIN_OUTPUT_DIR PFI_LOCAL_RESUME_DIR PFI_LOCAL_MODELS_DIR PFI_LOCAL_MANIFESTS_DIR PFI_LOCAL_LOGS_DIR PFI_VENV_DIR; do check_path "$n"; done; }
is_linux(){ [[ "$(uname -s 2>/dev/null || printf unknown)" == Linux ]]; }
run_or_dry(){ if [[ "$DRY_RUN" -eq 1 ]]; then dry "$*"; else "$@"; fi; }
ensure_linux(){ if [[ "$DRY_RUN" -eq 0 ]] && ! is_linux; then err "prepare real debe ejecutarse en Linux; use --dry-run localmente"; exit 2; fi; }
ensure_commands(){
  local missing=() c; for c in git curl jq rsync "$PFI_PYTHON_BIN"; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
  if [[ ${#missing[@]} -eq 0 ]]; then info "comandos base disponibles"; return; fi
  if [[ "$DRY_RUN" -eq 1 ]]; then dry "faltan comandos: ${missing[*]}; en real se instalaria lo minimo si no hay --skip-apt"; return; fi
  [[ "$SKIP_APT" -eq 0 ]] || { err "faltan comandos con --skip-apt: ${missing[*]}"; exit 1; }
  is_linux || { err "apt-get solo se usa en Linux"; exit 1; }
  command -v apt-get >/dev/null 2>&1 || { err "apt-get no disponible"; exit 1; }
  local pkgs=(); for c in "${missing[@]}"; do case "$c" in git|curl|jq|rsync) pkgs+=("$c");; python3) pkgs+=(python3 python3-venv);; *) err "sin paquete apt permitido para $c"; exit 1;; esac; done
  if ! "$PFI_PYTHON_BIN" -m venv --help >/dev/null 2>&1; then pkgs+=(python3-venv); fi
  if [[ "$APT_UPDATED" -eq 0 ]]; then info "apt-get update"; if [[ "$(id -u)" -eq 0 ]]; then apt-get update; else sudo apt-get update; fi; APT_UPDATED=1; fi
  info "instalando apt minimo: ${pkgs[*]}"; if [[ "$(id -u)" -eq 0 ]]; then apt-get install -y "${pkgs[@]}"; else sudo apt-get install -y "${pkgs[@]}"; fi
}
make_dir(){
  local d="$1" u g parent; u="$(id -un 2>/dev/null || printf root)"; g="$(id -gn 2>/dev/null || printf root)"; parent="$(dirname "$d")"
  if [[ "$DRY_RUN" -eq 1 ]]; then dry "crear directorio 0755 $d"; return; fi
  [[ -d "$d" ]] && { info "directorio existente: $d"; return; }
  if [[ -w "$parent" || "$(id -u)" -eq 0 ]]; then install -d -m 0755 "$d"; else sudo install -d -m 0755 -o "$u" -g "$g" "$d"; fi
}
ensure_dirs(){ local dirs=("$PFI_VM_ROOT" "$PFI_DATA_ROOT" "$PFI_OUTPUT_ROOT" "$PFI_TRAIN_OUTPUT_DIR" "$PFI_LOCAL_RESUME_DIR" "$PFI_LOCAL_MODELS_DIR" "$PFI_LOCAL_MANIFESTS_DIR" "$PFI_LOCAL_LOGS_DIR" "$(dirname "$PFI_REPO_ROOT")"); local d; for d in "${dirs[@]}"; do make_dir "$d"; done; }
strip_git(){ local x="$1"; x="${x%.git}"; printf '%s\n' "$x"; }
ensure_repo(){
  local checkout target origin; checkout="$(cd "$REPO_CHECKOUT_ROOT" && pwd -P)"; target="$PFI_REPO_ROOT"; [[ -d "$PFI_REPO_ROOT" ]] && target="$(cd "$PFI_REPO_ROOT" && pwd -P)"
  if [[ "$checkout" == "$target" ]]; then info "reutilizando checkout actual: $checkout"; git -C "$checkout" branch --show-current | sed 's/^/[INFO] rama: /'; git -C "$checkout" rev-parse --short HEAD | sed 's/^/[INFO] HEAD: /'; return; fi
  if [[ -e "$PFI_REPO_ROOT" ]]; then [[ -d "$PFI_REPO_ROOT/.git" ]] || { err "PFI_REPO_ROOT no es repo Git"; exit 1; }; origin="$(git -C "$PFI_REPO_ROOT" remote get-url origin)"; [[ "$(strip_git "$origin")" == "$(strip_git "$PFI_REPO_URL")" ]] || { err "origin no coincide: $origin"; exit 1; }; info "repo existente valido: $PFI_REPO_ROOT"; return; fi
  if [[ "$DRY_RUN" -eq 1 ]]; then dry "clonar $PFI_REPO_URL en $PFI_REPO_ROOT"; else git clone "$PFI_REPO_URL" "$PFI_REPO_ROOT"; fi
}
base_python_info(){ if [[ "$DRY_RUN" -eq 1 || "$SKIP_PYTHON" -eq 1 ]]; then dry "comprobar Python base y torch/CUDA reportada"; return; fi; "$PFI_PYTHON_BIN" - <<'PY'
import sys
print(f"[INFO] Python base: {sys.version.split()[0]}")
try:
 import torch; print(f"[INFO] torch base: {torch.__version__}"); print(f"[INFO] CUDA reportada: {torch.version.cuda}")
except Exception as exc: print(f"[WARN] torch no importable desde Python base: {exc}")
PY
}
ensure_venv(){
  [[ "$SKIP_PYTHON" -eq 0 ]] || { info "--skip-python activo"; return; }
  base_python_info
  if [[ "$DRY_RUN" -eq 1 ]]; then dry "crear venv --system-site-packages en $PFI_VENV_DIR"; dry "instalar solo paquetes faltantes no torch/torchvision"; return; fi
  command -v "$PFI_PYTHON_BIN" >/dev/null 2>&1 || { err "Python no disponible: $PFI_PYTHON_BIN"; exit 1; }
  if [[ ! -x "$PFI_VENV_DIR/bin/python" ]]; then make_dir "$(dirname "$PFI_VENV_DIR")"; "$PFI_PYTHON_BIN" -m venv --system-site-packages "$PFI_VENV_DIR"; fi
  [[ -x "$PFI_VENV_DIR/bin/python" ]] || { err "venv invalido: $PFI_VENV_DIR"; exit 1; }
  local miss pkgs=() m; miss="$($PFI_VENV_DIR/bin/python - <<'PY'
mods=['numpy','scipy','pandas','matplotlib','skimage','sklearn','torch','torchvision','pytest','SimpleITK','pydicom','PIL','nbformat','nbconvert','jupyter','ipykernel']
missing=[]
for m in mods:
 try: __import__(m)
 except Exception: missing.append(m)
print(' '.join(missing))
PY
)"
  for m in $miss; do case "$m" in torch|torchvision) err "$m falta; prepare no lo instala ni actualiza"; exit 1;; numpy) pkgs+=(numpy);; scipy) pkgs+=(scipy);; pandas) pkgs+=(pandas);; matplotlib) pkgs+=(matplotlib);; skimage) pkgs+=(scikit-image);; sklearn) pkgs+=(scikit-learn);; pytest) pkgs+=(pytest);; SimpleITK) pkgs+=(SimpleITK);; pydicom) pkgs+=(pydicom);; PIL) pkgs+=(pillow);; nbformat) pkgs+=(nbformat);; nbconvert) pkgs+=(nbconvert);; jupyter) pkgs+=(jupyter);; ipykernel) pkgs+=(ipykernel);; *) err "sin mapping pip: $m"; exit 1;; esac; done
  if [[ ${#pkgs[@]} -gt 0 ]]; then info "pip install faltantes: ${pkgs[*]}"; "$PFI_VENV_DIR/bin/python" -m pip install "${pkgs[@]}"; fi
}
final_report(){
  info "repo: $PFI_REPO_ROOT"; info "venv: $PFI_VENV_DIR"; info "data root: $PFI_DATA_ROOT"; info "output root: $PFI_OUTPUT_ROOT"; info "Python: $PFI_PYTHON_BIN"
  if [[ "$DRY_RUN" -eq 0 && "$SKIP_PYTHON" -eq 0 && -x "$PFI_VENV_DIR/bin/python" ]]; then "$PFI_VENV_DIR/bin/python" - <<'PY'
try:
 import torch; print(f"[INFO] PyTorch: {torch.__version__}"); print(f"[INFO] CUDA reportada: {torch.version.cuda}")
except Exception as exc: print(f"[WARN] PyTorch no importable: {exc}")
PY
  fi
  warn "todavia no se descargaron datasets"; warn "todavia no se valido una GPU"; warn "ejecutar preflight-training-vm.sh --mode vm en la VM real"
}
main(){ load_env; validate_env; ensure_linux; info "usuario/grupo: $(id -un 2>/dev/null || printf unknown)/$(id -gn 2>/dev/null || printf unknown)"; ensure_commands; ensure_dirs; ensure_repo; ensure_venv; final_report; }
main "$@"