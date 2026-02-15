#!/usr/bin/env python3
"""
crib_wind_archive_processor.py
================================
Level 3 Archive Pattern Processor for Harrison-Dever Crib Wind Prediction
Lake Michigan Chicago Racing · NOAA GLERL CHII2 Station

Downloads and processes 25+ years (2000–2025) of 2-minute wind observations
from the Harrison-Dever Crib into a compact lookup table for pattern-matching
prediction. Outputs two files:

  crib_patterns.json   — Main lookup: {month}_{dirBucket}_{spdBucket}_{hourBucket}
                         → { p60_spd, p60_dir, p60_gust, n_samples, confidence }
  crib_seasonal.json   — Monthly statistics: avg speed, direction rose,
                         lake breeze frequency, shift statistics

Usage:
    pip install requests numpy scipy tqdm
    python3 crib_wind_archive_processor.py

    # Only download specific years:
    python3 crib_wind_archive_processor.py --years 2020 2021 2022 2023 2024

    # Use already-downloaded .txt files:
    python3 crib_wind_archive_processor.py --local-dir ./glerl_data/

    # Query the lookup table after building:
    python3 crib_wind_archive_processor.py --query --month 7 --dir 200 --spd 9 --hour 13

Author: Generated for Chicago sailboat racing wind prediction tool
Data source: https://www.glerl.noaa.gov/metdata/chi/archive/
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("WARNING: numpy not found — using pure Python fallback (slower)")
    HAS_NUMPY = False

# ─── CONFIG ──────────────────────────────────────────────────────────────────

BASE_URL   = "https://www.glerl.noaa.gov/metdata/chi/archive/"
YEARS      = list(range(2000, 2026))   # 2000–2025 inclusive
OUTPUT_DIR = Path(".")

# Observation interval is 2 minutes → 30 obs = 60 min ahead
STEPS_AHEAD = 30

# Buckets for lookup key
DIR_BUCKET_SIZE = 45    # 8 compass octants (N, NE, E, SE, S, SW, W, NW)
SPD_BUCKET_SIZE = 5     # 0–4, 5–9, 10–14, 15–19, 20–24, 25+ kts
HOUR_BUCKET_SIZE = 3    # 0–2, 3–5, 6–8, 9–11, 12–14, 15–17, 18–20, 21–23 CST

MIN_SAMPLES = 10        # minimum obs to include a bucket in output

# ─── DATA PARSING ─────────────────────────────────────────────────────────────

def ms_to_kts(v: float) -> float:
    return v * 1.94384

def parse_line(line: str) -> Optional[dict]:
    """Parse one 2-min observation line from GLERL .txt archive file."""
    parts = line.strip().split()
    # Format: ID Year DOY UTC AirTempC WindSpdMs WindGstMs WindDirDeg RelHum%
    if len(parts) < 8:
        return None
    if parts[0] != '4':      # station ID 4 = Harrison-Dever Crib
        return None
    try:
        year  = int(parts[1])
        doy   = int(parts[2])
        utc_s = parts[3].zfill(4)
        air_c = float(parts[4])
        spd   = ms_to_kts(float(parts[5]))
        gst   = ms_to_kts(float(parts[6]))
        dirg  = float(parts[7])
        rh    = float(parts[8]) if len(parts) > 8 else None

        # Skip bad data flags
        if any(abs(v) > 990 for v in [spd, gst, dirg] if v is not None):
            return None
        if spd < 0 or gst < 0:
            return None
        if not (0 <= dirg <= 360):
            return None

        # Convert DOY + UTC to month and CST hour
        # DOY 1 = Jan 1; add offset. Use datetime for accuracy.
        try:
            dt_utc = datetime(year, 1, 1, tzinfo=timezone.utc)
            # Add DOY-1 days + UTC hours/minutes
            utc_h = int(utc_s[:2])
            utc_m = int(utc_s[2:])
            dt_utc = dt_utc.replace(
                month=1, day=1
            )
            # Simpler: compute from DOY
            import calendar
            # Jan 1 = DOY 1
            month = 1
            day_count = doy
            while day_count > 0:
                days_in_month = calendar.monthrange(year, month)[1]
                if day_count <= days_in_month:
                    break
                day_count -= days_in_month
                month += 1
                if month > 12:
                    month = 12
                    break

            # CST = UTC - 6
            cst_h = (utc_h - 6) % 24

        except Exception:
            return None

        return {
            'year': year,
            'month': month,
            'doy': doy,
            'hour_cst': cst_h,
            'spd_kts': round(spd, 2),
            'gst_kts': round(gst, 2),
            'dir_deg': round(dirg, 1),
            'air_f': round(air_c * 9 / 5 + 32, 1),
            'rh': round(rh, 1) if rh else None,
        }
    except (ValueError, IndexError, ZeroDivisionError):
        return None


def parse_file(filepath: Path) -> list[dict]:
    """Parse all valid observations from a GLERL archive .txt file."""
    obs = []
    try:
        with open(filepath, 'r', errors='replace') as f:
            for line in f:
                if line.strip().startswith('4'):
                    r = parse_line(line)
                    if r:
                        obs.append(r)
    except Exception as e:
        print(f"  WARNING: Could not parse {filepath}: {e}")
    return obs


# ─── DOWNLOAD ─────────────────────────────────────────────────────────────────

def download_year(year: int, dest_dir: Path) -> Optional[Path]:
    """Download one year's raw data from GLERL. Returns local path or None."""
    fname = f"chi{year}.04t.txt"
    url   = f"{BASE_URL}{fname}"
    dest  = dest_dir / fname

    if dest.exists() and dest.stat().st_size > 50000:
        print(f"  {year}: already cached ({dest.stat().st_size:,} bytes)")
        return dest

    print(f"  {year}: downloading {url} ...", end='', flush=True)
    try:
        r = requests.get(url, timeout=60, headers={'User-Agent': 'CribWindProcessor/1.0'})
        if r.status_code == 200 and len(r.content) > 1000:
            dest.write_bytes(r.content)
            print(f" {len(r.content):,} bytes")
            time.sleep(0.5)   # polite rate-limit
            return dest
        else:
            print(f" FAILED (HTTP {r.status_code})")
            return None
    except Exception as e:
        print(f" ERROR: {e}")
        return None


# ─── CIRCULAR STATISTICS ──────────────────────────────────────────────────────

def circular_mean(angles_deg: list[float]) -> float:
    """Mean direction of a list of angles (degrees), handles wraparound."""
    if not angles_deg:
        return 0.0
    sin_sum = sum(math.sin(math.radians(a)) for a in angles_deg)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles_deg)
    mean_rad = math.atan2(sin_sum / len(angles_deg), cos_sum / len(angles_deg))
    return (math.degrees(mean_rad) + 360) % 360


def circular_std(angles_deg: list[float]) -> float:
    """Circular standard deviation in degrees."""
    if len(angles_deg) < 2:
        return 0.0
    sin_sum = sum(math.sin(math.radians(a)) for a in angles_deg)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles_deg)
    n = len(angles_deg)
    R = math.sqrt((sin_sum/n)**2 + (cos_sum/n)**2)
    R = min(R, 1.0)
    return math.degrees(math.sqrt(-2 * math.log(R))) if R > 0 else 180.0


def circ_diff(a: float, b: float) -> float:
    """Signed angular difference b - a, in (-180, 180]."""
    return ((b - a + 540) % 360) - 180


# ─── BUCKETING ────────────────────────────────────────────────────────────────

def dir_bucket(deg: float) -> int:
    """Return octant bucket index 0–7 (N=0, NE=1, …, NW=7)."""
    return int((deg + DIR_BUCKET_SIZE/2) / DIR_BUCKET_SIZE) % (360 // DIR_BUCKET_SIZE)


def spd_bucket(kts: float) -> int:
    """Speed bucket: 0=calm(0-4), 1=light(5-9), 2=mod(10-14), 3=fresh(15-19), 4=strong(20+)."""
    return min(int(kts / SPD_BUCKET_SIZE), 5)


def hour_bucket(h: int) -> int:
    """3-hour block bucket."""
    return h // HOUR_BUCKET_SIZE


def lookup_key(month: int, dir_deg: float, spd_kts: float, hour_cst: int) -> str:
    return f"{month}_{dir_bucket(dir_deg)}_{spd_bucket(spd_kts)}_{hour_bucket(hour_cst)}"


DIR_BUCKET_NAMES = ['N','NE','E','SE','S','SW','W','NW']
SPD_BUCKET_NAMES = ['Calm(0-4)','Light(5-9)','Mod(10-14)','Fresh(15-19)','Strong(20-24)','Gale(25+)']


# ─── PATTERN BUILDING ─────────────────────────────────────────────────────────

def build_patterns(all_obs: list[dict]) -> tuple[dict, dict]:
    """
    Build two output structures:

    patterns: {lookup_key: {spd_samples, dir_samples, gust_samples,
                            p60_spd_mean, p60_spd_std,
                            p60_dir_mean, p60_dir_std,
                            p60_gust_mean, p60_gust_std,
                            spd_delta_mean, dir_delta_mean,
                            n_samples}}

    seasonal: {month: {avg_spd, avg_dir, dir_rose, avg_gust,
                       shift_freq_per_hr, lb_freq_by_hour,
                       spd_percentiles, temp_f_range}}
    """
    print(f"\nBuilding pattern lookup from {len(all_obs):,} observations...")

    # Group observations into sliding windows of STEPS_AHEAD
    # For each obs[i], pair with obs[i + STEPS_AHEAD] if same day
    patterns: dict[str, dict] = defaultdict(lambda: {
        'curr_spd': [], 'pred_spd': [],
        'curr_dir': [], 'pred_dir': [],
        'curr_gst': [], 'pred_gst': [],
        'dir_deltas': [], 'spd_deltas': [],
    })

    # Seasonal accumulators
    seasonal: dict[int, dict] = {
        m: {
            'spds': [], 'dirs': [], 'gusts': [],
            'temps': [], 'rhs': [],
            'dir_changes_per_hour': [],   # abs direction change per pair
            'lb_hours': defaultdict(int), # hour → count of NE flow obs
            'lb_total': defaultdict(int), # hour → total obs
        }
        for m in range(1, 13)
    }

    n = len(all_obs)
    skipped = 0
    matched = 0

    for i in range(n - STEPS_AHEAD):
        curr = all_obs[i]
        fwd  = all_obs[i + STEPS_AHEAD]

        # Accumulate seasonal stats regardless of forward match
        m = curr['month']
        seasonal[m]['spds'].append(curr['spd_kts'])
        seasonal[m]['dirs'].append(curr['dir_deg'])
        seasonal[m]['gusts'].append(curr['gst_kts'])
        if curr['air_f']:
            seasonal[m]['temps'].append(curr['air_f'])
        if curr['rh']:
            seasonal[m]['rhs'].append(curr['rh'])
        # Lake breeze proxy: NE/ENE/NNE flow in summer during afternoon
        h = curr['hour_cst']
        seasonal[m]['lb_total'][h] += 1
        if curr['dir_deg'] and 20 <= curr['dir_deg'] <= 90 and curr['spd_kts'] > 3:
            seasonal[m]['lb_hours'][h] += 1

        # Only pair obs from the same year+DOY (same day — avoids midnight wrapping)
        if curr['year'] != fwd['year'] or curr['doy'] != fwd['doy']:
            skipped += 1
            continue

        matched += 1
        key = lookup_key(curr['month'], curr['dir_deg'], curr['spd_kts'], curr['hour_cst'])
        p   = patterns[key]

        p['curr_spd'].append(curr['spd_kts'])
        p['pred_spd'].append(fwd['spd_kts'])
        p['curr_dir'].append(curr['dir_deg'])
        p['pred_dir'].append(fwd['dir_deg'])
        p['curr_gst'].append(curr['gst_kts'])
        p['pred_gst'].append(fwd['gst_kts'])
        p['spd_deltas'].append(fwd['spd_kts'] - curr['spd_kts'])
        p['dir_deltas'].append(circ_diff(curr['dir_deg'], fwd['dir_deg']))

    print(f"  Matched {matched:,} pairs | Skipped {skipped:,} cross-day | {len(patterns):,} unique buckets")

    # ── Compress pattern buckets ──
    compressed_patterns = {}
    thin_buckets = 0
    for key, p in patterns.items():
        n_s = len(p['pred_spd'])
        if n_s < MIN_SAMPLES:
            thin_buckets += 1
            continue

        def mean(lst): return sum(lst) / len(lst) if lst else 0
        def std(lst):
            if len(lst) < 2: return 0
            m = mean(lst)
            return math.sqrt(sum((x-m)**2 for x in lst) / len(lst))
        def pct(lst, p):
            if not lst: return 0
            s = sorted(lst)
            i = int(len(s) * p / 100)
            return s[min(i, len(s)-1)]

        pred_spd_vals = p['pred_spd']
        pred_dir_vals = p['pred_dir']
        pred_gst_vals = p['pred_gst']
        spd_deltas    = p['spd_deltas']
        dir_deltas    = p['dir_deltas']

        # Direction using circular stats
        p60_dir_mean = circular_mean(pred_dir_vals)
        p60_dir_std  = circular_std(pred_dir_vals)
        dir_delta_mean = mean(dir_deltas)

        # Speed
        p60_spd_mean  = mean(pred_spd_vals)
        p60_spd_std   = std(pred_spd_vals)
        spd_delta_mean = mean(spd_deltas)

        # Gust
        p60_gst_mean  = mean(pred_gst_vals)

        # Confidence: based on sample size + std deviations
        # High n, low std → high confidence
        n_factor  = min(1.0, math.log(n_s + 1) / math.log(200))
        spd_factor = max(0, 1 - p60_spd_std / 8)
        dir_factor = max(0, 1 - p60_dir_std / 45)
        base_conf  = round((n_factor * 0.3 + spd_factor * 0.4 + dir_factor * 0.3) * 100)

        # Percentile distribution for speed (useful for risk bands)
        spd_p10 = pct(pred_spd_vals, 10)
        spd_p90 = pct(pred_spd_vals, 90)

        compressed_patterns[key] = {
            'n':               n_s,
            'p60_spd_mean':    round(p60_spd_mean, 2),
            'p60_spd_std':     round(p60_spd_std, 2),
            'p60_spd_p10':     round(spd_p10, 1),
            'p60_spd_p90':     round(spd_p90, 1),
            'p60_dir_mean':    round(p60_dir_mean, 1),
            'p60_dir_std':     round(p60_dir_std, 1),
            'p60_gst_mean':    round(p60_gst_mean, 2),
            'spd_delta_mean':  round(spd_delta_mean, 2),
            'dir_delta_mean':  round(dir_delta_mean, 1),
            'base_confidence': base_conf,
        }

    print(f"  Patterns: {len(compressed_patterns):,} buckets (dropped {thin_buckets:,} thin buckets < {MIN_SAMPLES} samples)")

    # ── Compress seasonal ──
    compressed_seasonal = {}
    for m, s in seasonal.items():
        if not s['spds']:
            continue

        def mean(lst): return sum(lst)/len(lst) if lst else 0
        def pct(lst, p):
            if not lst: return 0
            sl = sorted(lst)
            return sl[int(len(sl)*p/100)]

        # Direction rose: count obs per 16-point compass
        rose = defaultdict(int)
        for d in s['dirs']:
            sector = int((d + 11.25) / 22.5) % 16
            rose[sector] += 1
        total_dir_obs = len(s['dirs'])
        dir_rose_pct = {
            ['N','NNE','NE','ENE','E','ESE','SE','SSE',
             'S','SSW','SW','WSW','W','WNW','NW','NNW'][k]: round(v/total_dir_obs*100, 1)
            for k, v in rose.items()
        }

        # Shift frequency: mean abs direction change per adjacent 2-min pair
        # Approximate from dirs array
        if len(s['dirs']) > 1:
            abs_changes = [abs(circ_diff(s['dirs'][i], s['dirs'][i+1]))
                           for i in range(len(s['dirs'])-1)]
            avg_change_per_2min = mean(abs_changes)
            shift_freq_per_hr = round(avg_change_per_2min * 30 / 15, 2)
        else:
            shift_freq_per_hr = 0

        # Lake breeze frequency by hour (NE obs / total obs)
        lb_by_hour = {}
        for h in range(24):
            tot = s['lb_total'].get(h, 0)
            lb  = s['lb_hours'].get(h, 0)
            lb_by_hour[str(h)] = round(lb / tot * 100, 1) if tot > 0 else 0

        compressed_seasonal[str(m)] = {
            'month_name':       ['','Jan','Feb','Mar','Apr','May','Jun',
                                  'Jul','Aug','Sep','Oct','Nov','Dec'][m],
            'n_obs':            len(s['spds']),
            'avg_spd_kts':      round(mean(s['spds']), 2),
            'spd_p25_kts':      round(pct(s['spds'], 25), 1),
            'spd_p75_kts':      round(pct(s['spds'], 75), 1),
            'avg_gust_kts':     round(mean(s['gusts']), 2),
            'avg_dir_deg':      round(circular_mean(s['dirs']), 1),
            'dir_rose_pct':     dir_rose_pct,
            'shift_freq_per_hr': shift_freq_per_hr,
            'lb_prob_by_hour':  lb_by_hour,
            'avg_temp_f':       round(mean(s['temps']), 1),
            'avg_rh_pct':       round(mean(s['rhs']), 1),
        }

    return compressed_patterns, compressed_seasonal


# ─── QUERY MODE ───────────────────────────────────────────────────────────────

def query_patterns(patterns: dict, seasonal: dict,
                   month: int, dir_deg: float,
                   spd_kts: float, hour_cst: int):
    """Look up prediction for given conditions. Falls back to broader buckets."""

    key = lookup_key(month, dir_deg, spd_kts, hour_cst)
    result = patterns.get(key)

    fallback_used = False
    if result is None:
        # Try relaxing hour bucket
        for h_fuzz in [0, -1, 1, -2, 2, -3, 3]:
            h_try = (hour_cst + h_fuzz * HOUR_BUCKET_SIZE) % 24
            k2 = lookup_key(month, dir_deg, spd_kts, h_try)
            if k2 in patterns:
                result = patterns[k2]
                fallback_used = True
                break

    if result is None:
        print(f"\n  No pattern found for key: {key} (month={month}, dir={dir_deg}°, spd={spd_kts}kts, hour={hour_cst})")
        return

    db = DIR_BUCKET_NAMES[dir_bucket(dir_deg)]
    sb = SPD_BUCKET_NAMES[spd_bucket(spd_kts)]
    hb = f"{hour_bucket(hour_cst)*HOUR_BUCKET_SIZE:02d}–{hour_bucket(hour_cst)*HOUR_BUCKET_SIZE+HOUR_BUCKET_SIZE-1:02d}"

    print(f"\n{'═'*60}")
    print(f"  ARCHIVE PATTERN QUERY RESULT")
    print(f"{'═'*60}")
    print(f"  Input: Month={month}, Dir={dir_deg}° ({db}), Spd={spd_kts}kts ({sb}), Hour={hour_cst:02d}:00 CST ({hb})")
    if fallback_used: print(f"  ⚠ Fallback: relaxed hour bucket")
    print(f"  Samples: {result['n']:,} historical observations")
    print(f"{'─'*60}")
    print(f"  60-min prediction:")
    print(f"    Speed:     {result['p60_spd_mean']:.1f} kts ± {result['p60_spd_std']:.1f}  "
          f"  (P10: {result['p60_spd_p10']:.1f}, P90: {result['p60_spd_p90']:.1f})")
    print(f"    Direction: {result['p60_dir_mean']:.0f}° ± {result['p60_dir_std']:.0f}°")
    print(f"    Gust:      {result['p60_gst_mean']:.1f} kts")
    print(f"    Δ Speed:   {result['spd_delta_mean']:+.1f} kts vs current")
    print(f"    Δ Dir:     {result['dir_delta_mean']:+.0f}°  "
          f"({'veering' if result['dir_delta_mean'] > 0 else 'backing'})")
    print(f"    Confidence: {result['base_confidence']}%")
    print(f"{'─'*60}")

    # Seasonal context
    sm = seasonal.get(str(month), {})
    if sm:
        print(f"  {sm.get('month_name','')} seasonal context:")
        print(f"    Avg speed: {sm['avg_spd_kts']} kts  |  Avg dir: {sm['avg_dir_deg']}°")
        print(f"    Shift freq: {sm['shift_freq_per_hr']}×/hr")
        lb_this_hour = sm.get('lb_prob_by_hour', {}).get(str(hour_cst), 0)
        print(f"    Lake breeze prob at {hour_cst:02d}:00 CST: {lb_this_hour}%")
    print(f"{'═'*60}\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='GLERL Crib Wind Archive Processor — builds a 25-year pattern lookup table'
    )
    parser.add_argument('--years', nargs='+', type=int, default=YEARS,
                        help='Years to process (default: 2000–2025)')
    parser.add_argument('--local-dir', type=str, default=None,
                        help='Directory with pre-downloaded .txt files (skip download)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory for output JSON files (default: current dir)')
    parser.add_argument('--query', action='store_true',
                        help='Run a query against existing crib_patterns.json')
    parser.add_argument('--month',  type=int, help='Query: month (1–12)')
    parser.add_argument('--dir',    type=float, help='Query: wind direction (degrees)')
    parser.add_argument('--spd',    type=float, help='Query: wind speed (knots)')
    parser.add_argument('--hour',   type=int, help='Query: CST hour (0–23)')
    parser.add_argument('--racing-months-only', action='store_true',
                        help='Only process May–September (faster)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns_file = out_dir / 'crib_patterns.json'
    seasonal_file = out_dir / 'crib_seasonal.json'

    # ── QUERY MODE ──
    if args.query:
        if not patterns_file.exists():
            print(f"ERROR: {patterns_file} not found. Run without --query first.")
            sys.exit(1)
        print(f"Loading {patterns_file} ...", end='', flush=True)
        patterns = json.loads(patterns_file.read_text())
        seasonal = json.loads(seasonal_file.read_text()) if seasonal_file.exists() else {}
        print(f" {len(patterns):,} buckets loaded")

        m   = args.month or int(input("Month (1–12): "))
        d   = args.dir   if args.dir is not None else float(input("Wind direction (°): "))
        s   = args.spd   if args.spd is not None else float(input("Speed (kts): "))
        h   = args.hour  if args.hour is not None else int(input("CST Hour (0–23): "))
        query_patterns(patterns, seasonal, m, d, s, h)
        return

    # ── PROCESSING MODE ──
    years = args.years
    if args.racing_months_only:
        print("Racing months only (May–Sep) — will filter after loading")
    print(f"\nCrib Wind Archive Processor")
    print(f"Years: {years[0]}–{years[-1]} ({len(years)} years)")
    print(f"Output: {out_dir.resolve()}")
    print(f"{'─'*50}")

    # Step 1: Download or locate files
    data_dir = Path(args.local_dir) if args.local_dir else out_dir / 'glerl_data'
    data_dir.mkdir(parents=True, exist_ok=True)

    files = []
    if args.local_dir:
        # Use local files
        txt_files = list(Path(args.local_dir).glob('chi*.txt'))
        print(f"\nFound {len(txt_files)} local .txt files in {args.local_dir}")
        files = sorted(txt_files)
    else:
        print(f"\nDownloading archive files to {data_dir}/ ...")
        for year in years:
            f = download_year(year, data_dir)
            if f:
                files.append(f)

    if not files:
        print("ERROR: No data files found or downloaded.")
        sys.exit(1)

    # Step 2: Parse all files
    print(f"\nParsing {len(files)} files...")
    all_obs = []
    for f in sorted(files):
        obs = parse_file(f)
        # Racing months filter
        if args.racing_months_only:
            obs = [o for o in obs if 5 <= o['month'] <= 9]
        all_obs.extend(obs)
        year_tag = f.stem[-4:] if len(f.stem) >= 4 else f.stem
        print(f"  {year_tag}: {len(obs):,} obs → total {len(all_obs):,}")

    if not all_obs:
        print("ERROR: No observations parsed.")
        sys.exit(1)

    print(f"\nTotal observations: {len(all_obs):,}")
    print(f"Date range: {all_obs[0]['year']} DOY {all_obs[0]['doy']} "
          f"→ {all_obs[-1]['year']} DOY {all_obs[-1]['doy']}")

    # Step 3: Build patterns
    patterns, seasonal = build_patterns(all_obs)

    # Step 4: Add metadata
    metadata = {
        '_meta': {
            'generated':    datetime.utcnow().isoformat() + 'Z',
            'source':       'NOAA GLERL Harrison-Dever Crib (CHII2)',
            'years':        f"{years[0]}–{years[-1]}",
            'n_obs_total':  len(all_obs),
            'n_buckets':    len(patterns),
            'steps_ahead':  STEPS_AHEAD,
            'obs_interval_min': 2,
            'prediction_horizon_min': STEPS_AHEAD * 2,
            'dir_bucket_deg': DIR_BUCKET_SIZE,
            'spd_bucket_kts': SPD_BUCKET_SIZE,
            'hour_bucket_hr': HOUR_BUCKET_SIZE,
            'min_samples':  MIN_SAMPLES,
            'bucket_key_format': '{month}_{dir_octant}_{spd_band}_{hour_block}',
            'dir_octants':  DIR_BUCKET_NAMES,
            'spd_bands':    SPD_BUCKET_NAMES,
        }
    }
    patterns.update(metadata)

    # Step 5: Write outputs
    patterns_file.write_text(json.dumps(patterns, indent=2))
    seasonal_file.write_text(json.dumps(seasonal, indent=2))

    p_size = patterns_file.stat().st_size
    s_size = seasonal_file.stat().st_size
    print(f"\n{'═'*50}")
    print(f"  ✓ crib_patterns.json — {len(patterns)-1:,} buckets — {p_size/1024:.0f} KB")
    print(f"  ✓ crib_seasonal.json — {len(seasonal)} months  — {s_size/1024:.0f} KB")
    print(f"{'═'*50}")
    print(f"\nNext steps:")
    print(f"  1. Host these JSON files on a CDN / static server")
    print(f"  2. In the dashboard JS, fetch crib_patterns.json on load")
    print(f"  3. In computePrediction(), call lookupArchivePattern(month, dir, spd, hour)")
    print(f"     and blend archive result into the ensemble with weight ~0.25")
    print(f"  4. Mark chip-archive as 'active' in the method chips")
    print(f"\nQuery example:")
    print(f"  python3 {sys.argv[0]} --query --month 7 --dir 200 --spd 9 --hour 13")
    print()


if __name__ == '__main__':
    main()
