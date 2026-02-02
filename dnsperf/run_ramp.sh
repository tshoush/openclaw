#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_ramp.py"

if command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="python"
else
  echo "Error: python3/python not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "Error: missing script: $PY_SCRIPT" >&2
  exit 1
fi

prompt_default() {
  local prompt="$1"
  local default="$2"
  local input
  read -r -p "$prompt [$default]: " input
  if [[ -z "$input" ]]; then
    printf "%s" "$default"
  else
    printf "%s" "$input"
  fi
}

prompt_yes_no() {
  local prompt="$1"
  local default="$2"
  local display
  local input
  if [[ "$default" == "true" ]]; then
    display="Y/n"
  else
    display="y/N"
  fi
  read -r -p "$prompt ($display): " input
  input="${input:-}"
  input="$(printf "%s" "$input" | tr '[:upper:]' '[:lower:]')"

  if [[ -z "$input" ]]; then
    printf "%s" "$default"
    return
  fi

  case "$input" in
    y|yes|true|1) printf "true" ;;
    n|no|false|0) printf "false" ;;
    *)
      echo "Invalid input: $input. Use y/n." >&2
      exit 1
      ;;
  esac
}

echo "=== dnsperf Ramp Wrapper ==="
echo "Press Enter to accept defaults."
echo

DEFAULT_FILES=(
  "$SCRIPT_DIR/sports-leagues-a.txt"
  "$SCRIPT_DIR/news-mainstream-a.txt"
  "$SCRIPT_DIR/tech-cloud-a.txt"
  "$SCRIPT_DIR/public-edu-a.txt"
)
DEFAULT_FILES_STR="${DEFAULT_FILES[*]}"

server="$(prompt_default "F5 DNS VIP / server IP" "127.0.0.1")"
dnsperf_bin="$(prompt_default "dnsperf binary path" "dnsperf")"
port="$(prompt_default "Port" "53")"
mode="$(prompt_default "Transport mode (udp|tcp|dot|doh)" "udp")"
files_raw="$(prompt_default "Query files (space-separated)" "$DEFAULT_FILES_STR")"
shuffle_files="$(prompt_yes_no "Shuffle file order each QPS step" "true")"

step_seconds="$(prompt_default "Seconds per step" "60")"
qps_start="$(prompt_default "Start QPS" "2000")"
qps_step="$(prompt_default "Step QPS increment" "2000")"
qps_max="$(prompt_default "Max QPS" "30000")"

clients="$(prompt_default "Clients (-c)" "4")"
threads="$(prompt_default "Threads (-T)" "2")"
outstanding="$(prompt_default "Outstanding queries (-q)" "20000")"
timeout="$(prompt_default "Per-query timeout seconds (-t)" "3")"
stats_interval="$(prompt_default "Stats interval seconds (-S, 0 to disable)" "0")"
latency_histogram="$(prompt_yes_no "Enable latency histogram for p50/p95" "true")"
pause_seconds="$(prompt_default "Pause between runs (seconds)" "2")"
seed="$(prompt_default "Shuffle seed" "42")"
stop_loss_pct="$(prompt_default "Stop if loss %% exceeds" "2.0")"
stop_p95_ms="$(prompt_default "Stop if p95 ms exceeds (-1 disables)" "-1.0")"
output_prefix="$(prompt_default "Output prefix" "$SCRIPT_DIR/ramp-$(date +%Y%m%d-%H%M%S)")"
extra_args_raw="$(prompt_default "Extra dnsperf args (optional, space-separated)" "")"

if ! command -v "$dnsperf_bin" >/dev/null 2>&1; then
  echo "Error: dnsperf binary not found: $dnsperf_bin" >&2
  exit 1
fi

files=()
extra_args=()
if [[ -n "$files_raw" ]]; then
  IFS=' ' read -r -a files <<< "$files_raw"
fi
if [[ -n "$extra_args_raw" ]]; then
  IFS=' ' read -r -a extra_args <<< "$extra_args_raw"
fi

if [[ "${#files[@]}" -eq 0 ]]; then
  echo "Error: at least one query file is required." >&2
  exit 1
fi

cmd=(
  "$PY_BIN" "$PY_SCRIPT"
  --dnsperf-bin "$dnsperf_bin"
  --server "$server"
  --port "$port"
  --mode "$mode"
  --step-seconds "$step_seconds"
  --qps-start "$qps_start"
  --qps-step "$qps_step"
  --qps-max "$qps_max"
  --clients "$clients"
  --threads "$threads"
  --outstanding "$outstanding"
  --timeout "$timeout"
  --stats-interval "$stats_interval"
  --pause-seconds "$pause_seconds"
  --seed "$seed"
  --stop-loss-pct "$stop_loss_pct"
  --stop-p95-ms "$stop_p95_ms"
  --output-prefix "$output_prefix"
  --files "${files[@]}"
)

if [[ "$shuffle_files" == "true" ]]; then
  cmd+=(--shuffle-files)
fi

if [[ "$latency_histogram" == "true" ]]; then
  cmd+=(--latency-histogram)
fi

if [[ "${#extra_args[@]}" -gt 0 ]]; then
  for arg in "${extra_args[@]}"; do
    [[ -z "$arg" ]] && continue
    cmd+=(--extra-arg "$arg")
  done
fi

echo
echo "Command to run:"
printf '  %q' "${cmd[@]}"
echo
echo

confirm="$(prompt_yes_no "Start test now" "true")"
if [[ "$confirm" != "true" ]]; then
  echo "Cancelled."
  exit 0
fi

exec "${cmd[@]}"
