#!/usr/bin/env python3
"""
Run dnsperf in controlled QPS ramps and export per-step metrics.

Example:
  python dnsperf/run_ramp.py \
    --server 10.0.0.53 \
    --files dnsperf/sports-leagues-a.txt dnsperf/news-mainstream-a.txt \
    --qps-start 2000 --qps-step 2000 --qps-max 30000 \
    --step-seconds 60 --clients 8 --threads 4 --outstanding 20000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_FILES = [
    "dnsperf/sports-leagues-a.txt",
    "dnsperf/news-mainstream-a.txt",
    "dnsperf/tech-cloud-a.txt",
    "dnsperf/public-edu-a.txt",
]

QTYPE_ORDER = ["A", "AAAA", "CNAME", "MX", "NS", "SOA", "TXT", "CAA", "HTTPS", "SVCB"]

RE_SENT = re.compile(r"^\s*Queries sent:\s*([0-9]+)\s*$", re.IGNORECASE | re.MULTILINE)
RE_COMPLETED = re.compile(
    r"^\s*Queries completed:\s*([0-9]+)(?:\s+\([^)]+\))?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
RE_LOST = re.compile(r"^\s*Queries lost:\s*([0-9]+)(?:\s+\([^)]+\))?\s*$", re.IGNORECASE | re.MULTILINE)
RE_QPS = re.compile(r"^\s*Queries per second:\s*([0-9.]+)\s*$", re.IGNORECASE | re.MULTILINE)
RE_AVG_LAT = re.compile(
    r"^\s*Average Latency \(s\):\s*([0-9.eE+-]+)\s+\(min\s+([0-9.eE+-]+),\s*max\s+([0-9.eE+-]+)\)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
RE_STDDEV = re.compile(
    r"^\s*Latency StdDev \(s\):\s*([0-9.eE+-]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class StepResult:
    query_file: str
    qps_target: int
    sent: int
    completed: int
    lost: int
    loss_pct: float
    qps_achieved: float | None
    lat_avg_ms: float | None
    lat_min_ms: float | None
    lat_max_ms: float | None
    lat_stddev_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    rc: int
    elapsed_s: float
    command: str
    stderr_tail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dnsperf ramp tests and export metrics.")
    parser.add_argument("--dnsperf-bin", default="dnsperf", help="Path to dnsperf binary.")
    parser.add_argument("--server", required=True, help="DNS server/VIP address.")
    parser.add_argument("--port", type=int, default=53, help="Server port.")
    parser.add_argument("--mode", default="udp", choices=["udp", "tcp", "dot", "doh"], help="Transport mode.")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES, help="Query files.")
    parser.add_argument("--shuffle-files", action="store_true", help="Shuffle file order each QPS step.")
    parser.add_argument("--step-seconds", type=int, default=60, help="Duration per run in seconds.")
    parser.add_argument("--qps-start", type=int, default=2000, help="Ramp start QPS.")
    parser.add_argument("--qps-step", type=int, default=2000, help="Ramp step QPS.")
    parser.add_argument("--qps-max", type=int, default=30000, help="Ramp max QPS.")
    parser.add_argument("--clients", type=int, default=4, help="dnsperf -c")
    parser.add_argument("--threads", type=int, default=2, help="dnsperf -T")
    parser.add_argument("--outstanding", type=int, default=20000, help="dnsperf -q")
    parser.add_argument("--timeout", type=int, default=3, help="dnsperf -t")
    parser.add_argument("--stats-interval", type=int, default=0, help="dnsperf -S value, 0 disables.")
    parser.add_argument(
        "--latency-histogram",
        action="store_true",
        help="Add -O latency-histogram and compute p50/p95 from bins when available.",
    )
    parser.add_argument("--pause-seconds", type=float, default=2.0, help="Pause between runs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=2.0,
        help="Stop ramp if any run exceeds this loss percent; negative disables.",
    )
    parser.add_argument(
        "--stop-p95-ms",
        type=float,
        default=-1.0,
        help="Stop ramp if any run exceeds this p95 latency in ms; negative disables.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument to pass to dnsperf (repeatable).",
    )
    parser.add_argument(
        "--output-prefix",
        default="dnsperf/ramp",
        help="Prefix for output files. Writes <prefix>.json, <prefix>.csv.",
    )
    return parser.parse_args()


def ensure_inputs(args: argparse.Namespace) -> None:
    if shutil.which(args.dnsperf_bin) is None:
        raise SystemExit(
            f"dnsperf binary not found: {args.dnsperf_bin}\n"
            "Install dnsperf or pass --dnsperf-bin /path/to/dnsperf"
        )
    for file_path in args.files:
        if not Path(file_path).is_file():
            raise SystemExit(f"Query file not found: {file_path}")
    if args.qps_start <= 0 or args.qps_step <= 0 or args.qps_max < args.qps_start:
        raise SystemExit("Invalid QPS ramp settings.")
    if args.step_seconds <= 0:
        raise SystemExit("--step-seconds must be > 0")


def qps_targets(start: int, step: int, max_qps: int) -> list[int]:
    values = []
    qps = start
    while qps <= max_qps:
        values.append(qps)
        qps += step
    return values


def percentile_from_histogram(histogram: list[list[float]], percentile: float) -> float | None:
    if not histogram:
        return None
    total = 0.0
    for bucket in histogram:
        if len(bucket) != 3:
            continue
        total += float(bucket[2])
    if total <= 0:
        return None

    threshold = total * (percentile / 100.0)
    seen = 0.0
    for bucket in histogram:
        if len(bucket) != 3:
            continue
        low, high, count = float(bucket[0]), float(bucket[1]), float(bucket[2])
        if count <= 0:
            continue
        next_seen = seen + count
        if next_seen >= threshold:
            ratio = (threshold - seen) / count
            ratio = min(1.0, max(0.0, ratio))
            return low + (high - low) * ratio
        seen = next_seen
    return float(histogram[-1][1])


def parse_dnsperf_json(stdout: str) -> dict[str, Any] | None:
    stats_obj = None
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "statistics" not in obj:
            continue
        stats = obj["statistics"]
        # Prefer the final aggregate statistics object.
        if not stats.get("interval", False):
            stats_obj = stats
        elif stats_obj is None:
            stats_obj = stats
    return stats_obj


def text_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def parse_dnsperf_text(stdout: str) -> dict[str, Any] | None:
    sent = text_match(RE_SENT, stdout)
    completed = text_match(RE_COMPLETED, stdout)
    lost = text_match(RE_LOST, stdout)
    if sent is None or completed is None or lost is None:
        return None

    stats: dict[str, Any] = {
        "sent": int(sent),
        "completed": int(completed),
        "lost": int(lost),
        "qps": None,
        "latency": {},
    }
    qps = text_match(RE_QPS, stdout)
    if qps is not None:
        stats["qps"] = float(qps)

    avg_match = RE_AVG_LAT.search(stdout)
    if avg_match:
        stats["latency"]["avg"] = float(avg_match.group(1))
        stats["latency"]["min"] = float(avg_match.group(2))
        stats["latency"]["max"] = float(avg_match.group(3))

    stddev_match = RE_STDDEV.search(stdout)
    if stddev_match:
        stats["latency"]["stddev"] = float(stddev_match.group(1))
    return stats


def ms(value: float | None) -> float | None:
    return None if value is None else value * 1000.0


def dnsperf_supports_json(dnsperf_bin: str) -> bool:
    try:
        proc = subprocess.run([dnsperf_bin, "-H"], text=True, capture_output=True, check=False)
    except OSError:
        return False
    combined = f"{proc.stdout}\n{proc.stderr}"
    return " -j " in combined or "\n-j " in combined or "\n  -j" in combined


def build_command(args: argparse.Namespace, query_file: str, qps_target: int, use_json: bool) -> list[str]:
    cmd = [
        args.dnsperf_bin,
        "-s",
        args.server,
        "-p",
        str(args.port),
        "-m",
        args.mode,
        "-d",
        query_file,
        "-Q",
        str(qps_target),
        "-l",
        str(args.step_seconds),
        "-q",
        str(args.outstanding),
        "-c",
        str(args.clients),
        "-T",
        str(args.threads),
        "-t",
        str(args.timeout),
    ]
    if use_json:
        cmd.append("-j")
    if args.stats_interval > 0:
        cmd.extend(["-S", str(args.stats_interval)])
    if args.latency_histogram:
        cmd.extend(["-O", "latency-histogram"])
    for extra in args.extra_arg:
        cmd.append(extra)
    return cmd


def run_step(args: argparse.Namespace, query_file: str, qps_target: int, use_json: bool) -> StepResult:
    cmd = build_command(args, query_file, qps_target, use_json=use_json)
    started = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.time() - started

    stats = parse_dnsperf_json(proc.stdout)
    if stats is None:
        stats = parse_dnsperf_text(proc.stdout)
    if stats is None:
        raise RuntimeError(
            f"Unable to parse dnsperf output for {query_file} at {qps_target} QPS.\n"
            f"STDOUT tail:\n{proc.stdout[-1200:]}\n\nSTDERR tail:\n{proc.stderr[-1200:]}"
        )

    sent = int(stats.get("sent", 0))
    completed = int(stats.get("completed", 0))
    lost = int(stats.get("lost", max(sent - completed, 0)))
    loss_pct = (lost / sent * 100.0) if sent > 0 else 0.0

    lat = stats.get("latency") or {}
    histogram = lat.get("histogram") or []
    p50 = percentile_from_histogram(histogram, 50.0)
    p95 = percentile_from_histogram(histogram, 95.0)

    return StepResult(
        query_file=query_file,
        qps_target=qps_target,
        sent=sent,
        completed=completed,
        lost=lost,
        loss_pct=loss_pct,
        qps_achieved=float(stats["qps"]) if stats.get("qps") is not None else None,
        lat_avg_ms=ms(float(lat["avg"])) if lat.get("avg") is not None else None,
        lat_min_ms=ms(float(lat["min"])) if lat.get("min") is not None else None,
        lat_max_ms=ms(float(lat["max"])) if lat.get("max") is not None else None,
        lat_stddev_ms=ms(float(lat["stddev"])) if lat.get("stddev") is not None else None,
        p50_ms=ms(p50),
        p95_ms=ms(p95),
        rc=proc.returncode,
        elapsed_s=elapsed,
        command=" ".join(cmd),
        stderr_tail=(proc.stderr[-800:] if proc.stderr else ""),
    )


def print_result(row: StepResult) -> None:
    p50 = "n/a" if row.p50_ms is None else f"{row.p50_ms:.2f}"
    p95 = "n/a" if row.p95_ms is None else f"{row.p95_ms:.2f}"
    lat = "n/a" if row.lat_avg_ms is None else f"{row.lat_avg_ms:.2f}"
    qps = "n/a" if row.qps_achieved is None else f"{row.qps_achieved:.1f}"
    print(
        f"qps={row.qps_target:>7} file={Path(row.query_file).name:<24} "
        f"sent={row.sent:>7} lost={row.lost:>6} loss={row.loss_pct:>6.2f}% "
        f"achieved={qps:>8} avg_ms={lat:>7} p50_ms={p50:>7} p95_ms={p95:>7}"
    )


def export_results(rows: list[StepResult], prefix: str) -> tuple[Path, Path]:
    out_base = Path(prefix)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix(".json")
    csv_path = out_base.with_suffix(".csv")

    payload = [asdict(r) for r in rows]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload[0].keys()) if payload else [])
        if payload:
            writer.writeheader()
            for row in payload:
                writer.writerow(row)
    return json_path, csv_path


def summarize_rows(rows: list[StepResult]) -> None:
    if not rows:
        print("No results.")
        return
    max_good = None
    for row in rows:
        if row.loss_pct <= 1.0:
            max_good = row
    if max_good:
        p95_str = "n/a" if max_good.p95_ms is None else f"{max_good.p95_ms:.2f}"
        print(
            f"\nBest <=1% loss run: qps={max_good.qps_target}, file={Path(max_good.query_file).name}, "
            f"loss={max_good.loss_pct:.2f}%, p95_ms={p95_str}"
        )


def maybe_stop(args: argparse.Namespace, row: StepResult) -> str | None:
    if args.stop_loss_pct >= 0 and row.loss_pct > args.stop_loss_pct:
        return f"loss {row.loss_pct:.2f}% exceeded threshold {args.stop_loss_pct:.2f}%"
    if args.stop_p95_ms >= 0 and row.p95_ms is not None and row.p95_ms > args.stop_p95_ms:
        return f"p95 {row.p95_ms:.2f}ms exceeded threshold {args.stop_p95_ms:.2f}ms"
    if row.rc != 0:
        return f"dnsperf exited with code {row.rc}"
    return None


def print_mix_report(files: list[str]) -> None:
    print("RR-type mix check:")
    for file_path in files:
        counts = {k: 0 for k in QTYPE_ORDER}
        total = 0
        with Path(file_path).open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                qtype = parts[1].upper()
                counts[qtype] = counts.get(qtype, 0) + 1
                total += 1
        line = ", ".join(
            f"{qtype}:{(counts.get(qtype, 0) / total * 100 if total else 0):.1f}%"
            for qtype in QTYPE_ORDER
        )
        print(f"  {Path(file_path).name}: {line}")


def main() -> int:
    args = parse_args()
    ensure_inputs(args)
    print_mix_report(args.files)
    json_supported = dnsperf_supports_json(args.dnsperf_bin)
    if json_supported:
        print("dnsperf JSON output: enabled (-j)")
    else:
        print("dnsperf JSON output: unavailable; using text parser")

    rng = random.Random(args.seed)
    rows: list[StepResult] = []
    ramp = qps_targets(args.qps_start, args.qps_step, args.qps_max)

    print("\nStarting ramp...")
    for qps_target in ramp:
        files = list(args.files)
        if args.shuffle_files:
            rng.shuffle(files)

        for query_file in files:
            row = run_step(args, query_file, qps_target, use_json=json_supported)
            rows.append(row)
            print_result(row)
            reason = maybe_stop(args, row)
            if reason:
                print(f"\nStopping ramp: {reason}")
                json_path, csv_path = export_results(rows, args.output_prefix)
                print(f"Saved: {json_path}")
                print(f"Saved: {csv_path}")
                return 0
            if args.pause_seconds > 0:
                time.sleep(args.pause_seconds)

    json_path, csv_path = export_results(rows, args.output_prefix)
    print("\nRamp complete.")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    summarize_rows(rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
