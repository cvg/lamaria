from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL_DEFAULT = "https://cvg-data.inf.ethz.ch/lamaria/"

FOLDERS = [
    "raw_data",
    "aria_calibrations",
    "asl_folder",
    "pinhole_calibrations",
    "rosbag",
    "ground_truth",
]
DEFAULT_SUBFOLDERS = ["training/", "test/"]
GT_SUBFOLDERS = ["pseudo_dense/", "sparse/"]

PAYLOADS = {
    "raw": ["raw_data", "aria_calibrations"],
    "pinhole": ["asl_folder", "pinhole_calibrations", "rosbag"],
    "both": [
        "raw_data",
        "aria_calibrations",
        "asl_folder",
        "pinhole_calibrations",
        "rosbag",
    ],
}


def fetch_index(full_url: str) -> list[str]:
    """Return list of hrefs on an Apache index page (files and folders)."""
    r = requests.get(full_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    hrefs = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # skip parent directory, sort/query links, and anchors
        if href == "../":
            continue
        if href.startswith("?") or "?" in href or href.startswith("#"):
            continue
        hrefs.append(href)
    return hrefs


def build_catalog(base_url: str) -> dict[str, list[tuple[str, str]]]:
    catalog = {}
    for folder in FOLDERS:
        entries = []
        subfolders = (
            GT_SUBFOLDERS if folder == "ground_truth" else DEFAULT_SUBFOLDERS
        )
        for sub in subfolders:
            url = base_url.rstrip("/") + "/" + folder + "/" + sub
            try:
                items = fetch_index(url)
            except Exception as e:
                print(f"[warn] Could not read {url}: {e}", file=sys.stderr)
                items = []
            for it in items:
                if it.endswith("/"):
                    continue
                entries.append((sub, it))
        catalog[folder] = entries
    return catalog


def names_from_listing(files: Iterable[str]) -> list[str]:
    names = set()
    pattern = r"\.(zip|tar|tar\.gz|tgz|7z|bag|vrs|json|txt|csv|tar\.xz)$"
    for f in files:
        f = f.rstrip("/")
        name = re.split(pattern, f, maxsplit=1, flags=re.IGNORECASE)[0]
        names.add(name)
    return sorted(names)


def derive_splits(
    catalog: dict[str, list[tuple[str, str]]],
) -> tuple[list[str], list[str]]:
    raw_entries = catalog.get("raw_data", [])
    train_files = [fname for (sub, fname) in raw_entries if sub == "training/"]
    test_files = [fname for (sub, fname) in raw_entries if sub == "test/"]
    train = sorted(names_from_listing(train_files))
    test = sorted(names_from_listing(test_files))
    return train, test


def raw_split_map(catalog: dict[str, list[tuple[str, str]]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for sub, fname in catalog.get("raw_data", []):
        if sub not in ("training/", "test/"):
            continue
        if fname.startswith("?") or "?" in fname:
            continue
        base = names_from_listing([fname])[0]
        if base in mapping and mapping[base] == "training/":
            continue
        mapping[base] = sub
    return mapping


def pick_files_for_sequence(
    catalog: dict[str, list[tuple[str, str]]],
    sequence: str,
    folders: list[str],
    split: str | None,
) -> list[tuple[str, str, str]]:
    matches: list[tuple[str, str, str]] = []
    for folder in folders:
        for sub, fname in catalog.get(folder, []):
            if folder != "ground_truth" and split is not None and sub != split:
                continue
            if fname.startswith(sequence):
                matches.append((folder, sub, fname))
    return matches


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def download_file(
    url: str, dest: Path, session: requests.Session, chunk_size: int = 1 << 20
) -> None:
    headers = {}
    existing = dest.stat().st_size if dest.exists() else 0

    head = session.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    total = (
        int(head.headers.get("Content-Length", "0"))
        if head.headers.get("Content-Length")
        else None
    )

    if total is not None and existing > 0 and existing < total:
        headers["Range"] = f"bytes={existing}-"
    elif total is not None and existing == total:
        print(f"[skip] {dest} (already complete, {human_size(total)})")
        return

    with session.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        mode = "ab" if "Range" in headers else "wb"
        total_to_show = None
        if total is not None:
            total_to_show = total - existing if "Range" in headers else total

        with (
            open(dest, mode) as f,
            tqdm(
                total=total_to_show,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                initial=0,
                desc=dest.name,
                leave=False,
            ) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    if total is not None and dest.stat().st_size != total:
        print(
            f"[warn] Size mismatch for {dest} (expected {total}, got {dest.stat().st_size})"
        )


def main():
    parser = argparse.ArgumentParser(description="LaMAria dataset downloader")
    parser.add_argument(
        "--set",
        choices=["training", "test", "specific"],
        required=True,
        help="What to download: training sequences, test sequences, or a specific list",
    )
    parser.add_argument(
        "--type",
        choices=["raw", "pinhole", "both"],
        required=True,
        help="Which data to fetch",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Sequence names (required for --set specific). Example: R_01_easy sequence_3_17",
    )
    parser.add_argument(
        "--out-dir",
        default="out_dir",
        help="Root output folder (script will create out_dir/lamaria)",
    )
    args = parser.parse_args()

    # Root: out_dir/lamaria
    root = Path(args.out_dir) / "lamaria"
    ensure_dir(root)

    print("[info] Building catalog from server…")
    catalog = build_catalog(BASE_URL_DEFAULT)
    train_names, test_names = derive_splits(catalog)
    split_lookup = raw_split_map(catalog)  # seq -> 'training/' or 'test/'

    if args.set == "training":
        target_sequences = train_names
        global_split = "training/"
    elif args.set == "test":
        target_sequences = test_names
        global_split = "test/"
    else:
        if not args.sequences:
            print(
                "[error] --set specific requires --sequences", file=sys.stderr
            )
            sys.exit(2)
        target_sequences = args.sequences
        global_split = None  # per-sequence

    # To fetch
    folders = PAYLOADS[args.type].copy()
    plan = []

    created_split_dirs = set()

    for seq in target_sequences:
        if global_split is not None:
            seq_split = global_split
        else:
            seq_split = split_lookup.get(seq)
            if seq_split is None:
                raise ValueError(
                    f"Could not determine split for sequence: {seq}"
                )

        if seq_split in ("training/", "test/"):
            split_folder = "training" if seq_split == "training/" else "test"
        else:
            split_folder = "unknown"

        seq_root = root / split_folder / seq
        if split_folder not in created_split_dirs:
            ensure_dir(root / split_folder)
            created_split_dirs.add(split_folder)
        ensure_dir(seq_root)

        selected = pick_files_for_sequence(
            catalog,
            seq,
            folders,
            split=(seq_split if seq_split in ("training/", "test/") else None),
        )

        if seq_split == "training/":
            selected += pick_files_for_sequence(
                catalog, seq, ["ground_truth"], split=None
            )

        if not selected:
            print(f"[warn] No files found for: {seq}")
            continue

        for folder, sub, fname in selected:
            url = BASE_URL_DEFAULT + folder + "/" + sub + fname
            if folder == "ground_truth":
                dest_dir = seq_root / folder / sub.rstrip("/")
            else:
                dest_dir = seq_root / folder

            ensure_dir(dest_dir)
            dest = dest_dir / fname
            plan.append((folder, sub, fname, url, dest))

    if not plan:
        print("[warn] Nothing to download with the given filters.")
        return

    print("\n[download] Starting downloads.")
    with requests.Session() as sess:
        for _area, _sub, fname, url, dest in plan:
            print(f"[get] {url}")
            try:
                download_file(url, dest, sess)
            except Exception as e:
                print(
                    f"[error] Failed: {url} -> {dest}\n        {e}",
                    file=sys.stderr,
                )

    print(
        "\n[done] All downloads attempted. Files stored under:", root.resolve()
    )


if __name__ == "__main__":
    main()
