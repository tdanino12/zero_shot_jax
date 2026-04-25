"""
read_msgpack_debug.py
---------------------
Reads counter_circuit debug files (MessagePack format, length-prefixed records)
and displays them in human-readable form.

Usage:
    python read_msgpack_debug.py                        # full results report for each file
    python read_msgpack_debug.py --summary              # compact per-record action table
    python read_msgpack_debug.py --record 5             # pretty-print a specific record
    python read_msgpack_debug.py --all                  # pretty-print every record
    python read_msgpack_debug.py --file path/to/file    # specify a single file

Requirements:
    pip install msgpack numpy
"""

import struct
import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

try:
    import msgpack
except ImportError:
    print("Install msgpack first:  pip install msgpack")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Install numpy first:  pip install numpy")
    sys.exit(1)


# ── Default file paths ──────────────────────────────────────────────────────────
DEFAULT_FILES = [ "user=1440183103_name=counter_circuit_debug=0.json", "user=2446854346_name=counter_circuit_debug=0.json", ]

SURVEY_QUESTIONS = [
    "The agent adapted to me when making decisions.",
    "The agent was consistent in its actions.",
    "The agent's actions were human-like.",
    "The agent frequently got in my way.",
    "The agent's behavior was frustrating.",
    "Overall, I enjoyed playing with the agent.",
    "Overall, I felt that the agent's ability to coordinate with me was:",
]


# ── File loading ────────────────────────────────────────────────────────────────

def load_records(path):
    """
    The files use MessagePack serialisation despite the .json extension.
    Each record is prefixed with a 4-byte big-endian length:
        [ uint32 length ][ msgpack bytes ] ...
    """
    records = []
    with open(path, "rb") as f:
        raw = f.read()

    pos = 0
    while pos < len(raw):
        if pos + 4 > len(raw):
            break
        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        pos += 4
        if pos + length > len(raw):
            print(f"  [WARNING] Truncated record at byte {pos}; stopping.")
            break
        chunk = raw[pos : pos + length]
        try:
            obj = msgpack.unpackb(chunk, raw=False, strict_map_key=False)
            records.append(obj)
        except Exception as e:
            print(f"  [WARNING] Could not decode record {len(records)}: {e}")
        pos += length

    return records


# ── JAX array decoding ──────────────────────────────────────────────────────────

def decode_jax_ext(ext):
    """
    JAX arrays in the timestep blobs are stored as msgpack ExtType(code=1).
    The ext payload is itself a msgpack list: [shape, dtype_str, raw_bytes].
    """
    inner = msgpack.unpackb(ext.data, raw=False, strict_map_key=False)
    shape, dtype_str, data = inner
    arr = np.frombuffer(data, dtype=np.dtype(dtype_str))
    return float(arr[0]) if arr.size == 1 else arr.tolist()


def decode_timestep(ts_blob):
    """Decode a timestep msgpack blob and return reward and step_type as Python scalars."""
    ts = msgpack.unpackb(ts_blob, raw=False, strict_map_key=False)
    result = {}
    for key in ("reward", "step_type", "discount"):
        val = ts.get(key)
        if hasattr(val, "data"):          # ExtType
            result[key] = decode_jax_ext(val)
        else:
            result[key] = val
    return result


# ── Per-block aggregation ───────────────────────────────────────────────────────

def aggregate_blocks(records):
    """
    Walk all EnvStage records and accumulate reward + episode counts per block.
    step_type: 0=FIRST, 1=MID, 2=LAST (episode terminal).
    """
    blocks = defaultdict(lambda: {"rewards": [], "episodes": 0})
    for r in records:
        meta = r.get("metadata", {})
        if not isinstance(meta, dict) or meta.get("type") != "EnvStage":
            continue
        name = r.get("name", "unknown")
        ts_blob = r.get("data", {}).get("timestep")
        if not isinstance(ts_blob, bytes):
            continue
        try:
            ts = decode_timestep(ts_blob)
            blocks[name]["rewards"].append(ts.get("reward", 0.0))
            if ts.get("step_type", -1) == 2:      # LAST = episode end
                blocks[name]["episodes"] += 1
        except Exception:
            pass
    return blocks


def get_feedback(records):
    """Return a dict of {survey_name: data_dict} for all FeedbackStage records."""
    feedback = {}
    for r in records:
        meta = r.get("metadata", {})
        if not isinstance(meta, dict) or meta.get("type") != "FeedbackStage":
            continue
        feedback[r.get("name", "")] = r.get("data", {})
    return feedback


def match_survey(block_name, feedback):
    """Find the survey whose name best matches a block name (exact match preferred)."""
    candidates = []
    for survey_name, data in feedback.items():
        method = survey_name.replace(" Counter Circuit Survey", "").strip().replace(" ", "_")
        if method == block_name:
            return data   # exact match — return immediately
        if block_name.startswith(method) or method in block_name:
            candidates.append((len(method), data))
    if candidates:
        # Pick the longest (most specific) match
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


# ── Display helpers ─────────────────────────────────────────────────────────────

def clean_bytes(obj, max_bytes=32):
    """Replace raw bytes with a readable placeholder."""
    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>" if len(obj) > max_bytes else obj.hex()
    if isinstance(obj, dict):
        return {k: clean_bytes(v, max_bytes) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_bytes(i, max_bytes) for i in obj]
    return obj


def pretty(obj, indent=2):
    return json.dumps(clean_bytes(obj), indent=indent, ensure_ascii=False, default=str)


def print_results_report(records, path):
    """Main human-readable report: session info + per-method reward + survey answers."""
    # ── Session / user info ──────────────────────────────────────────────────
    user_data = next((r.get("user_data") for r in records if r.get("user_data")), {})
    uid       = user_data.get("user_id", "?")
    age       = user_data.get("age", "?")
    sex       = user_data.get("sex", "?")

    last = records[-1] if records else {}
    us = last.get("user_storage", {})
    session_start = us.get("session_start", "?")
    duration_min  = round(us.get("session_duration", 0.0), 2)
    finished      = us.get("experiment_finished", False)
    session_id    = next((r.get("session_id") for r in records if r.get("session_id")), "?")

    seed = us.get("seed", "?")

    # ── Layout: collect unique block desc values (environment descriptions) ──
    layout_descs = []
    seen_descs = set()
    for r in records:
        meta = r.get("metadata", {})
        if not isinstance(meta, dict):
            continue
        bm = meta.get("block_metadata", {})
        if isinstance(bm, dict):
            desc = bm.get("desc", "").strip()
            if desc and desc not in seen_descs and desc.lower() != "instructions":
                seen_descs.add(desc)
                layout_descs.append(desc)
    layout_str = ", ".join(layout_descs) if layout_descs else "N/A"

    p = Path(path)
    print(f"\n{'='*68}")
    print(f"  FILE    : {p.name}  ({p.stat().st_size:,} bytes)")
    print(f"  USER ID : {uid}  |  age={age}  sex={sex}")
    print(f"  Session : {session_id}")
    print(f"  Seed    : {seed}")
    print(f"  Layout  : {layout_str}")
    print(f"  Start   : {session_start}")
    print(f"  Duration: {duration_min} min")
    print(f"  Records : {len(records)}")
    print(f"  Done    : {'Yes' if finished else 'No (incomplete)'}")
    print(f"{'='*68}")

    # ── Tutorial summary ─────────────────────────────────────────────────────
    blocks   = aggregate_blocks(records)
    feedback = get_feedback(records)

    tutorial = blocks.pop("tutorial", None)
    if tutorial:
        total_r = sum(tutorial["rewards"])
        print(f"\n  [Tutorial]  reward={total_r:.0f}  steps={len(tutorial['rewards'])}  "
              f"episodes={tutorial['episodes']}")

    # ── Per-method performance table ─────────────────────────────────────────
    print(f"\n  {'-'*64}")
    print(f"  {'METHOD':<35} {'REWARD':>8}  {'STEPS':>6}  {'EPISODES':>8}")
    print(f"  {'-'*64}")

    method_details = []
    for block_name, data in blocks.items():
        total_r = sum(data["rewards"])
        steps   = len(data["rewards"])
        eps     = data["episodes"]
        survey  = match_survey(block_name, feedback)
        print(f"  {block_name:<35} {total_r:>8.0f}  {steps:>6}  {eps:>8}")
        method_details.append((block_name, total_r, steps, eps, survey))

    # ── Survey answers per method ────────────────────────────────────────────
    print(f"\n\n  {'='*64}")
    print(f"  SURVEY ANSWERS PER METHOD")
    print(f"  {'='*64}")

    for block_name, total_r, steps, eps, survey in method_details:
        print(f"\n  +-- {block_name}  (reward={total_r:.0f}, steps={steps}, episodes={eps})")
        if survey:
            for q in SURVEY_QUESTIONS:
                ans = survey.get(q, "N/A")
                q_short = (q[:60] + "..") if len(q) > 62 else q
                print(f"  |   {q_short:<62}  ->  {ans}")
        else:
            print(f"  |   (no survey recorded for this block)")
        print(f"  +{'-'*67}")


def print_summary(records):
    """Compact one-line-per-record action table."""
    print(f"\n{'#':<6} {'stage_idx':<12} {'type':<16} {'name':<35} {'action':<12}")
    print("-" * 82)
    for i, rec in enumerate(records):
        stage_idx = rec.get("stage_idx", "")
        name      = rec.get("name", "")[:34]
        rec_type  = rec.get("metadata", {}).get("type", "") if isinstance(rec.get("metadata"), dict) else ""
        data      = rec.get("data", {})
        action    = data.get("action_name", "") if isinstance(data, dict) else ""
        print(f"{i:<6} {str(stage_idx):<12} {rec_type:<16} {name:<35} {action:<12}")


def print_record(rec, index):
    """Pretty-print a single record."""
    stage_idx  = rec.get("stage_idx", "?")
    session_id = rec.get("session_id", "?")
    name       = rec.get("name", "?")
    rec_type   = rec.get("metadata", {}).get("type", "?") if isinstance(rec.get("metadata"), dict) else "?"

    print(f"\n{'='*70}")
    print(f"  Record #{index}  |  stage_idx={stage_idx}  |  type={rec_type}  |  name={name}")
    print(f"  session: {session_id}")
    print(f"{'='*70}")

    data = rec.get("data", {})
    if isinstance(data, dict):
        for key in ("image_seen_time", "action_taken_time", "computer_interaction",
                    "action_name", "action_idx", "timelimit"):
            if key in data:
                print(f"  {key:<26} {data[key]}")

    ud = rec.get("user_data", {})
    if isinstance(ud, dict):
        print(f"\n  User: id={ud.get('user_id')}  age={ud.get('age')}  sex={ud.get('sex')}")

    meta = rec.get("metadata", {})
    if isinstance(meta, dict):
        bm = meta.get("block_metadata", {})
        print(f"\n  Block : {bm.get('name', '?')}  desc=\"{bm.get('desc', '')}\"")
        print(f"  Layout: {bm.get('desc', 'N/A')}")
        print(f"  nsteps={meta.get('nsteps')}  nepisodes={meta.get('nepisodes')}  nsuccesses={meta.get('nsuccesses')}")

    us = rec.get("user_storage", {})
    if isinstance(us, dict) and "seed" in us:
        print(f"  Seed  : {us.get('seed', '?')}")

    if rec_type == "FeedbackStage" and isinstance(data, dict):
        print("\n  Feedback responses:")
        skip = {"image_seen_time", "action_taken_time", "computer_interaction",
                "action_name", "action_idx", "timelimit", "prolific_id"}
        for q, ans in data.items():
            if q not in skip and isinstance(ans, str):
                print(f"    Q: {q}")
                print(f"    A: {ans}")

    print(f"\n  Full record (bytes truncated):\n")
    for line in pretty(rec).splitlines():
        print("    " + line)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Display counter_circuit debug files.")
    parser.add_argument("--file",    type=str, default=None,  help="Path to a specific file")
    parser.add_argument("--all",     action="store_true",      help="Print every record in full")
    parser.add_argument("--summary", action="store_true",      help="Print a compact action table")
    parser.add_argument("--record",  type=int, default=None,   help="Print a single record by index")
    args = parser.parse_args()

    files = [args.file] if args.file else DEFAULT_FILES

    for path in files:
        p = Path(path)
        if not p.exists():
            print(f"\n[SKIP] File not found: {path}")
            continue

        records = load_records(str(p))

        if args.summary:
            print(f"\n{'#'*70}\n  {p.name}  ({len(records)} records)\n{'#'*70}")
            print_summary(records)
        elif args.all:
            print(f"\n{'#'*70}\n  {p.name}  ({len(records)} records)\n{'#'*70}")
            for i, rec in enumerate(records):
                print_record(rec, i)
        elif args.record is not None:
            if args.record < len(records):
                print_record(records[args.record], args.record)
            else:
                print(f"  [ERROR] Record {args.record} out of range (max {len(records)-1})")
        else:
            # Default: full results report
            print_results_report(records, str(p))


if __name__ == "__main__":
    main()