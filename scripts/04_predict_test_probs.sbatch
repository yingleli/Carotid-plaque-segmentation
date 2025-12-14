#!/bin/bash
#SBATCH --job-name=04_predict_parts
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --array=0-7%8
#SBATCH --output=/cluster2/yingleli/logs/%x_%A_%a.out
#SBATCH --error=/cluster2/yingleli/logs/%x_%A_%a.err

set -euo pipefail
set -E
export PYTHONUNBUFFERED=1
ulimit -n 4096 || true

log(){ echo "[$(date '+%F %T')] $*"; }
heartbeat(){ log "heartbeat: still working... (host=$(hostname))"; }
run(){ log "CMD> $*"; "$@"; }

# —— 出错追踪（定位早退原因）——
trap 'log "ERR at line $LINENO: $BASH_COMMAND (exit=$?)"' ERR

# -------- 临时目录 --------
choose_job_tmp(){
  if [[ -n "${SLURM_JOB_TMPDIR:-}" && -d "$SLURM_JOB_TMPDIR" ]]; then echo "$SLURM_JOB_TMPDIR"; return; fi
  for d in /scratch /localscratch /local /tmp; do
    if [[ -d "$d" && -w "$d" ]]; then mkdir -p "$d/$USER"; mktemp -d -p "$d/$USER" jobtmp_${SLURM_JOB_ID:-$$}_XXXX; return; fi
  done
  mkdir -p "/cluster2/$USER/tmp"; mktemp -d -p "/cluster2/$USER/tmp" jobtmp_${SLURM_JOB_ID:-$$}_XXXX
}
JOB_TMP="$(choose_job_tmp)"
export HOME="$JOB_TMP/home"; mkdir -p "$HOME"
export TMPDIR="$JOB_TMP"
export PYTHONPYCACHEPREFIX="$JOB_TMP/pycache"
export TORCHINDUCTOR_CACHE_DIR="$JOB_TMP/torchinductor_cache"
mkdir -p "$PYTHONPYCACHEPREFIX" "$TORCHINDUCTOR_CACHE_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cleanup(){
  set +e
  log "cleanup…"
  for pid in "${HEARTBEAT_PID:-}" "${WATCHDOG_PID:-}" "${PROGRESS_PID:-}"; do
    [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null || true
  done
  sync; sleep 1; rm -rf "$JOB_TMP" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

log "job=${SLURM_JOB_NAME:-NA} id=${SLURM_JOB_ID:-NA} user=$USER node=$(hostname)"
log "workdir=$(pwd) tmp=$JOB_TMP"
( while true; do sleep 300; heartbeat; done ) & HEARTBEAT_PID=$!

# -------- 环境 --------
source /cluster2/yingleli/.bashrc
source /cluster2/yingleli/miniconda3/etc/profile.d/conda.sh
conda activate myenv
log "python=$(command -v python)  version=$(python -V 2>&1)"

# -------- 路径 --------
export nnUNet_raw="/cluster2/yingleli/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/cluster2/yingleli/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/cluster2/yingleli/nnUNet/nnUNet_results2"

DSET="Dataset001_CT"
DATASET=1
CONFIG=3d_fullres
TR=nnUNetTrainer_FTVOversample
MODEL_ROOT="$nnUNet_results/$DSET/${TR}__nnUNetPlans__${CONFIG}"
IN_DIR="$nnUNet_raw/$DSET/imagesTs"
OUT_DIR="$nnUNet_results/$DSET/$CONFIG/preds_test_ens"
mkdir -p "$OUT_DIR"

log "IN_DIR=$IN_DIR"
log "OUT_DIR=$OUT_DIR"
log "MODEL_ROOT=$MODEL_ROOT"

# -------- 分片参数 --------
PART_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_PARTS=$(( (${SLURM_ARRAY_TASK_MAX:-0}) + 1 ))
log "Array part: $PART_ID / $NUM_PARTS"

# -------- 折与 checkpoint --------
BEST_FOLDS=(); FINAL_FOLDS=()
for k in 0 1 2 3 4; do
  [[ -f "$MODEL_ROOT/fold_${k}/checkpoint_best.pth" ]]  && BEST_FOLDS+=("$k") && continue
  [[ -f "$MODEL_ROOT/fold_${k}/checkpoint_final.pth" ]] && FINAL_FOLDS+=("$k")
done
if [[ ${#BEST_FOLDS[@]} -gt 0 ]]; then
  CHK_FILE="checkpoint_best.pth"; AVAIL_FOLDS=("${BEST_FOLDS[@]}")
else
  [[ ${#FINAL_FOLDS[@]} -gt 0 ]] || { log "ERROR: no checkpoints found in $MODEL_ROOT"; exit 3; }
  CHK_FILE="checkpoint_final.pth"; AVAIL_FOLDS=("${FINAL_FOLDS[@]}")
fi
F_ARG=(-f "${AVAIL_FOLDS[@]}")
log "FOLDS=${AVAIL_FOLDS[*]}  CHK=$CHK_FILE"

# -------- 保存概率 / TTA / 续跑 --------
if nnUNetv2_predict -h 2>&1 | grep -q -- "--save_probabilities"; then SAVEFLAG="--save_probabilities"; else SAVEFLAG="--save_npz"; fi
TTA_FLAG=""; [[ "${DISABLE_TTA:-0}" == "1" ]] && TTA_FLAG="--disable_tta" && log "TTA: OFF" || log "TTA: ON"
CONT_FLAG=""; nnUNetv2_predict -h 2>&1 | grep -q -- "--continue_prediction" && CONT_FLAG="--continue_prediction"

# -------- 工具函数（稳健版）--------
list_outputs(){  # 列出 OUT_DIR 下所有 npz/npy（按名字排序）
  shopt -s nullglob
  local f
  for f in "$OUT_DIR"/*.npz "$OUT_DIR"/*.npy; do
    [[ -e "$f" ]] || continue
    basename "$f"
  done | sort
}
last_written(){  # 取最近 mtime 的5个文件名
  shopt -s nullglob
  local files=( "$OUT_DIR"/*.npz "$OUT_DIR"/*.npy )
  (( ${#files[@]} )) || return 0
  local f t
  for f in "${files[@]}"; do
    t=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null || echo 0)
    printf "%s\t%s\n" "$t" "$(basename "$f")"
  done | sort -nr | head -n 5 | cut -f2 | tr '\n' ' '
}
latest_mtime_epoch(){  # OUT_DIR 最新文件时间戳（无文件返回0）
  shopt -s nullglob
  local files=( "$OUT_DIR"/*.npz "$OUT_DIR"/*.npy )
  (( ${#files[@]} )) || { echo 0; return; }
  local f t max=0
  for f in "${files[@]}"; do
    t=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null || echo 0)
    (( t>max )) && max=$t
  done
  echo "$max"
}

# -------- 组装本分片子集（symlink 失败自动复制）--------
shopt -s nullglob
SUB_IN="$JOB_TMP/in_subset"; mkdir -p "$SUB_IN"
ALL_0000=( "$IN_DIR"/*_0000.nii.gz )
IFS=$'\n' ALL_0000=( $(printf '%s\n' "${ALL_0000[@]}" | sort) ); unset IFS
total=${#ALL_0000[@]}
(( total>0 )) || { log "ERROR: no *_0000.nii.gz under $IN_DIR"; exit 2; }

SELECTED_IDS="$JOB_TMP/selected_ids.txt"; : > "$SELECTED_IDS"
LINK_ERR="$JOB_TMP/link_errors.log"; : > "$LINK_ERR"
sel=0
for ((i=0; i<total; i++)); do
  (( i % NUM_PARTS == PART_ID )) || continue
  base="$(basename "${ALL_0000[$i]}")"
  id="${base%_0000.nii.gz}"
  for m in "$IN_DIR/${id}"_*.nii.gz; do
    [[ -e "$m" ]] || continue
    ln -s "$m" "$SUB_IN/" 2>>"$LINK_ERR" || {  # symlink 失败则复制
      log "WARN: symlink failed → cp $(basename "$m")"
      cp -n "$m" "$SUB_IN/" 2>>"$LINK_ERR" || cp "$m" "$SUB_IN/" 2>>"$LINK_ERR" || true
    }
  done
  echo "$id" >> "$SELECTED_IDS"
  ((sel+=1))  # ← 修复：避免 ((sel++)) 在 sel=0 时因 set -e 退出
done
log "subset: selected=$sel / total=$total  → $SUB_IN"
(( sel>0 )) || { log "WARN: this part has no assigned cases, exiting gracefully."; exit 0; }
cp "$SELECTED_IDS" "$OUT_DIR/selected_ids.part${PART_ID}.txt" 2>/dev/null || true

# -------- 进度/看门狗参数 --------
LOG_INTERVAL_MIN=${LOG_INTERVAL_MIN:-10}
WD_CHECK_MIN=${WD_CHECK_MIN:-10}
WD_STALL_MIN=${WD_STALL_MIN:-60}
WD_EXIT_ON_STALL=${WD_EXIT_ON_STALL:-0}

# 预测前快照
PRE_LIST="$(mktemp)"; list_outputs > "$PRE_LIST"

# 后台：进度日志（按时间展示最近5个）
( while true; do
    sleep $(( LOG_INTERVAL_MIN*60 ))
    CNT=$(list_outputs | wc -l | awk '{print $1}')
    LAST5=$(last_written || true)
    log "progress: outputs=$CNT  last5=[ $LAST5 ]"
  done ) & PROGRESS_PID=$!

# 后台：看门狗
( last_epoch="$(latest_mtime_epoch)"; [[ "$last_epoch" == "0" ]] && last_epoch="$(date +%s)"
  while true; do
    sleep $(( WD_CHECK_MIN*60 ))
    now="$(date +%s)"
    cur_epoch="$(latest_mtime_epoch)"; [[ "$cur_epoch" == "0" ]] && cur_epoch="$last_epoch"
    awk "BEGIN{exit !($cur_epoch > $last_epoch)}" && last_epoch="$cur_epoch"
    idle_min=$(( (now - ${last_epoch%.*}) / 60 ))
    if (( idle_min >= WD_STALL_MIN )); then
      log "WATCHDOG: no new outputs for ${idle_min} min (≥ $WD_STALL_MIN)."
      if (( WD_EXIT_ON_STALL == 1 )); then
        log "WATCHDOG: exiting due to stall (WD_EXIT_ON_STALL=1)."
        pkill -P $$ || true; exit 99
      fi
    fi
  done ) & WATCHDOG_PID=$!

# -------- 正式预测 --------
nvidia-smi || true
run nnUNetv2_predict \
  -i "$SUB_IN" -o "$OUT_DIR" \
  -d $DATASET -c $CONFIG "${F_ARG[@]}" \
  -tr "$TR" -chk "$CHK_FILE" \
  -device cuda \
  -npp 1 -nps 1 \
  $TTA_FLAG $CONT_FLAG $SAVEFLAG

# -------- 记录完成清单（仅本分片）--------
POST_LIST="$(mktemp)"; list_outputs > "$POST_LIST"
NEW_LIST="$(mktemp)"; comm -13 "$PRE_LIST" "$POST_LIST" > "$NEW_LIST"

PART_LIST_ALL="$OUT_DIR/completed_cases.part${PART_ID}.all.txt"; : > "$PART_LIST_ALL"
PART_LIST_NEW="$OUT_DIR/completed_cases.part${PART_ID}.new.txt"; : > "$PART_LIST_NEW"

while read -r id; do
  [[ -n "$id" ]] || continue
  [[ -f "$OUT_DIR/${id}.npz" ]] && echo "${id}.npz" >> "$PART_LIST_ALL"
  [[ -f "$OUT_DIR/${id}.npy" ]] && echo "${id}.npy" >> "$PART_LIST_ALL"
done < "$SELECTED_IDS"

awk '{print $0".npz"; print $0".npy"}' "$SELECTED_IDS" \
| grep -Ff - "$NEW_LIST" > "$PART_LIST_NEW" || true

GLOBAL_LIST="$OUT_DIR/completed_cases.txt"
if command -v flock >/dev/null 2>&1; then
  ( flock -w 60 9 || { log "WARN: cannot acquire lock to update completed_cases.txt"; exit 0; }
    cat "$PART_LIST_NEW" >> "$GLOBAL_LIST"
  ) 9>>"$OUT_DIR/.completed_cases.lock"
else
  cat "$PART_LIST_NEW" >> "$GLOBAL_LIST"
fi

ADDED=$(wc -l < "$PART_LIST_NEW" | awk '{print $1}')
TOTAL=$(wc -l < "$POST_LIST" | awk '{print $1}')
log "completed: added_by_this_job=$ADDED  total_outputs=$TOTAL"
log "lists: PART_ALL=$PART_LIST_ALL  PART_NEW=$PART_LIST_NEW  GLOBAL=$GLOBAL_LIST"

log "[OK] part $PART_ID/$NUM_PARTS done → $OUT_DIR"
