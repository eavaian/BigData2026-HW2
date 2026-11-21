"""Download GH Archive hour files (YYYY-MM-DD-H.json.gz) for a UTC date range.

    python src/download.py --start 2026-10-01 --end 2026-11-20T16 --out ../raw_final

Files already downloaded (and readable) are skipped, so the command can be re-run to resume.
Hours that GH Archive does not have (HTTP 404) are listed at the end.
"""
import argparse
import gzip
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

URL = "https://data.gharchive.org/{name}"
HEADERS = {"User-Agent": "gharchive-downloader/1.0"}  # the default Python agent gets HTTP 403


def hour_names(start, end):
    h = start
    while h < end:
        yield f"{h:%Y-%m-%d}-{h.hour}.json.gz"
        h += timedelta(hours=1)


def is_complete(path):
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except Exception:
        return False


def fetch(name, out, retries=4):
    path = out / name
    if path.exists() and is_complete(path):
        return name, "skip"
    tmp = path.with_suffix(".part")
    for _ in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL.format(name=name), headers=HEADERS), timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(path)
            return name, "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return name, "missing"
        except Exception:
            pass
    return name, "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="first UTC day, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="UTC time to stop before, YYYY-MM-DD or YYYY-MM-DDTHH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    parse = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    end = min(parse(a.end), datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
              - timedelta(hours=1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(hour_names(parse(a.start), end))
    status = {}
    with ThreadPoolExecutor(a.workers) as pool:
        futures = [pool.submit(fetch, n, out) for n in names]
        for i, fut in enumerate(as_completed(futures), 1):
            n, s = fut.result()
            status[n] = s
            if i % 24 == 0 or i == len(names):
                print(f"{i}/{len(names)} done", flush=True)
    for kind in ("missing", "failed"):
        bad = sorted(n for n, s in status.items() if s == kind)
        print(f"{kind}: {len(bad)}", *bad[:50])


if __name__ == "__main__":
    main()
