"""
src/data_stream.py
------------------
Simulates an incoming data stream by feeding the held-out 50% of data
into the system in chunks. After each chunk, PSI drift is checked
and retraining is triggered automatically if drift is detected.

The stream pool is created by train.py on first run.

Usage:
    python src/data_stream.py                  # stream one chunk manually
    python src/data_stream.py --chunk-size 500 # custom chunk size
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.append("/app/src")
from drift import check_drift_psi, MONITOR_FEATURES

# ── Constants ─────────────────────────────────────────────────────────────────
STREAM_POOL_PATH  = Path("data/processed/stream_pool.csv")   # held-out 50%
STREAM_DATA_PATH  = Path("data/processed/stream_data.csv")   # accumulated so far
STREAM_INDEX_PATH = Path("data/processed/stream_index.txt")  # current position
DEFAULT_CHUNK     = 1000


def get_current_index() -> int:
    """Read current stream position."""
    if STREAM_INDEX_PATH.exists():
        return int(STREAM_INDEX_PATH.read_text().strip())
    return 0


def save_current_index(idx: int):
    """Save current stream position."""
    STREAM_INDEX_PATH.write_text(str(idx))


def stream_next_chunk(chunk_size: int = DEFAULT_CHUNK) -> dict:
    """
    Feed the next chunk of data from the stream pool into the system.
    Returns drift check result.
    """
    if not STREAM_POOL_PATH.exists():
        return {"error": "Stream pool not found. Run python src/train.py first."}

    df_pool = pd.read_csv(STREAM_POOL_PATH)
    current_idx = get_current_index()

    if current_idx >= len(df_pool):
        return {
            "status"       : "stream_complete",
            "message"      : "All stream data has been consumed.",
            "total_streamed": current_idx,
        }

    # ── Get next chunk ─────────────────────────────────────────────────────────
    next_idx   = min(current_idx + chunk_size, len(df_pool))
    df_chunk   = df_pool.iloc[current_idx:next_idx]
    save_current_index(next_idx)

    # ── Accumulate stream data ─────────────────────────────────────────────────
    STREAM_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if STREAM_DATA_PATH.exists():
        df_chunk.to_csv(STREAM_DATA_PATH, mode="a", header=False, index=False)
    else:
        df_chunk.to_csv(STREAM_DATA_PATH, index=False)

    print(f"Streamed rows {current_idx} → {next_idx} ({len(df_chunk)} new rows)")
    print(f"Total accumulated: {next_idx} rows")

    # ── PSI Drift Check — use all accumulated data for a stable distribution ───
    df_accumulated = pd.read_csv(STREAM_DATA_PATH)
    drift_result = check_drift_psi(df_accumulated)
    print(f"Drift check: {drift_result}")

    return {
        "status"         : "ok",
        "rows_streamed"  : len(df_chunk),
        "total_streamed" : next_idx,
        "stream_remaining": len(df_pool) - next_idx,
        "drift"          : drift_result,
    }


def reset_stream():
    """Reset stream to beginning — for demo purposes."""
    if STREAM_INDEX_PATH.exists():
        STREAM_INDEX_PATH.unlink()
    if STREAM_DATA_PATH.exists():
        STREAM_DATA_PATH.unlink()
    print("Stream reset to beginning.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--reset", action="store_true", help="Reset stream to beginning")
    args = parser.parse_args()

    if args.reset:
        reset_stream()
    else:
        result = stream_next_chunk(chunk_size=args.chunk_size)
        print(result)