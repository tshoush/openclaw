# DNSPerf Load Test Kit

This folder contains:

- `sports-leagues-a.txt`
- `news-mainstream-a.txt`
- `tech-cloud-a.txt`
- `public-edu-a.txt`
- `run_ramp.py` (non-interactive ramp runner)
- `run_ramp.sh` (interactive wrapper with prompts + defaults)

Each query file has 12,000 lines and 12,000 unique `qname qtype` queries, with mixed RR types (`A`, `AAAA`, `CNAME`, `MX`, `NS`, `SOA`, `TXT`, `CAA`, `HTTPS`, `SVCB`).

## Prerequisites

- `dnsperf` installed and in `PATH`
- `python3` (or `python`) in `PATH`

## Quick Start (Interactive)

Run:

```bash
./dnsperf/run_ramp.sh
```

The script prompts for all major settings and shows defaults. Press Enter to keep defaults, or type a new value.

## Non-Interactive Example

```bash
python3 dnsperf/run_ramp.py \
  --server 10.0.0.53 \
  --latency-histogram \
  --shuffle-files \
  --qps-start 2000 \
  --qps-step 2000 \
  --qps-max 60000 \
  --step-seconds 60 \
  --clients 12 \
  --threads 6 \
  --outstanding 50000 \
  --stop-loss-pct 1.5 \
  --stop-p95-ms 40 \
  --output-prefix dnsperf/f5-ramp-$(date +%Y%m%d-%H%M%S)
```

## Outputs

`run_ramp.py` writes:

- `<output-prefix>.json`
- `<output-prefix>.csv`

Each row includes target QPS, achieved QPS, sent/completed/lost, loss %, and latency metrics (including p50/p95 when histogram is enabled).

## Suggested F5 Ramp Plan

1. Start low (`2k` QPS, `60s`).
2. Increase by fixed steps (`+2k` QPS each step).
3. Stop automatically with guardrails (for example, loss `>1.5%` or p95 `>40ms`).
4. Hold a soak run near the highest stable QPS.
5. Run a short burst above the stable point to check recovery behavior.
