"""
Download Astrodatabank c_sample from adbexport and convert to raw JSONL.

Uses the official public export (no scraping). Output matches the schema
expected by astrodatabank_to_bazi.py: name, date, time, longitude, latitude,
location, rodden_rating, category, biography, source, source_url.

Usage:
  python scripts/astrodatabank_export_to_jsonl.py --output data/astrodatabank_raw.jsonl
  python scripts/astrodatabank_export_to_jsonl.py --output data/astrodatabank_raw.jsonl --no-verify-ssl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

# Reuse coordinate parsing from API script (same DMS format in export)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.astrodatabank_api import parse_latitude, parse_longitude

ADBEXPORT_C_SAMPLE_URL = "https://www.astro.com/adbexport/c_sample.zip"
REQUEST_TIMEOUT = 120


def _tag(elt, name: str) -> ET.Element | None:
    if elt is None:
        return None
    for c in elt:
        if c.tag.split("}")[-1] == name:
            return c
    return None


def _text(elt: ET.Element | None) -> str:
    if elt is None:
        return ""
    return "".join(elt.itertext()).strip()


def _attr(elt: ET.Element | None, name: str, default: str = "") -> str:
    if elt is None:
        return default
    return elt.get(name) or elt.get(f"{name}") or default


def _parse_coord(s: str, is_lat: bool) -> float | None:
    if not s or not s.strip():
        return None
    s = s.strip()
    try:
        v = float(s)
        if is_lat and (-90 <= v <= 90):
            return v
        if not is_lat and (-180 <= v <= 180):
            return v
        return None
    except ValueError:
        pass
    return parse_latitude(s) if is_lat else parse_longitude(s)


def _rodden_ok(rating: str, min_rodden: str) -> bool:
    r = (rating or "").strip().upper()
    if min_rodden == "AA":
        return r == "AA"
    return r in ("AA", "A")


def _first_category(entry: ET.Element) -> str:
    research = _tag(entry, "research_data")
    if research is None:
        return "Unknown"
    cats = _tag(research, "categories")
    if cats is None:
        return "Unknown"
    for c in cats:
        if c.tag.split("}")[-1] == "category":
            text = (c.text or "").strip() or (c.get("catnotes") or "")
            if text:
                return text
    return "Unknown"


def entry_to_record(entry: ET.Element, min_rodden: str) -> dict | None:
    public = _tag(entry, "public_data")
    if public is None:
        return None
    name_elt = _tag(public, "name")
    name = _text(name_elt) if name_elt is not None else ""
    if not name:
        return None
    rodden_elt = _tag(public, "roddenrating")
    rating = _text(rodden_elt) if rodden_elt is not None else ""
    if not _rodden_ok(rating, min_rodden):
        return None
    bdata = _tag(public, "bdata")
    if bdata is None:
        return None
    place_elt = _tag(bdata, "place")
    if place_elt is None:
        return None
    slati = _attr(place_elt, "slati")
    slong = _attr(place_elt, "slong")
    lat = _parse_coord(slati, True)
    lng = _parse_coord(slong, False)
    if lat is None or lng is None:
        return None
    sbdate_dmy = _tag(bdata, "sbdate_dmy")
    date_str = _text(sbdate_dmy) if sbdate_dmy is not None else ""
    sbtime_elt = _tag(bdata, "sbtime")
    time_str = _text(sbtime_elt) if sbtime_elt is not None else ""
    place_str = _text(place_elt)
    country_elt = _tag(bdata, "country")
    country = _attr(country_elt, "sctr") if country_elt is not None else ""
    location = place_str or country or ""

    text_data = _tag(entry, "text_data")
    biography = ""
    adb_link = ""
    sourcenotes = ""
    if text_data is not None:
        bio_elt = _tag(text_data, "shortbiography")
        biography = _text(bio_elt) if bio_elt is not None else ""
        link_elt = _tag(text_data, "adb_link")
        adb_link = _text(link_elt) if link_elt is not None else ""
        src_elt = _tag(text_data, "sourcenotes")
        sourcenotes = _text(src_elt) if src_elt is not None else ""

    category = _first_category(entry)
    source_url = adb_link if adb_link.startswith("http") else f"https://www.astro.com/astro-databank/{adb_link}" if adb_link else ""

    return {
        "name": name,
        "date": date_str,
        "time": time_str,
        "longitude": lng,
        "latitude": lat,
        "location": location,
        "rodden_rating": rating.strip().upper(),
        "category": category,
        "biography": biography,
        "source": sourcenotes,
        "source_url": source_url,
    }


def download_c_sample(verify_ssl: bool = True) -> bytes:
    r = requests.get(
        ADBEXPORT_C_SAMPLE_URL,
        timeout=REQUEST_TIMEOUT,
        verify=verify_ssl,
    )
    r.raise_for_status()
    return r.content


def parse_c_sample_zip(zip_bytes: bytes, min_rodden: str) -> list[dict]:
    records = []
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
        names = [n for n in z.namelist() if n.endswith(".xml")]
        if not names:
            raise ValueError("No XML file in c_sample.zip")
        with z.open(names[0]) as f:
            tree = ET.parse(f)
    root = tree.getroot()
    for entry in root.iter():
        if entry.tag.split("}")[-1] == "adb_entry":
            rec = entry_to_record(entry, min_rodden)
            if rec is not None:
                records.append(rec)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Astrodatabank c_sample and convert to raw JSONL"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/astrodatabank_raw.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--input-zip",
        type=str,
        default="",
        help="Use local c_sample.zip instead of downloading (e.g. after manual download)",
    )
    parser.add_argument(
        "--rodden",
        type=str,
        default="A",
        choices=["AA", "A"],
        help="Minimum Rodden rating",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL verification for download",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    if args.input_zip:
        input_path = Path(args.input_zip)
        if not input_path.exists():
            print(f"Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Reading from {input_path}...")
        zip_bytes = input_path.read_bytes()
        print(f"Read {len(zip_bytes) / (1024*1024):.1f} MB")
    else:
        print("Downloading c_sample.zip from adbexport...")
        try:
            zip_bytes = download_c_sample(verify_ssl=not args.no_verify_ssl)
        except requests.RequestException as e:
            print(f"Download failed: {e}", file=sys.stderr)
            print(
                "Download c_sample.zip from https://www.astro.com/adbexport/c_sample.zip "
                "in a browser, then run with --input-zip /path/to/c_sample.zip",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Downloaded {len(zip_bytes) / (1024*1024):.1f} MB")

    print("Parsing XML and filtering records...")
    records = parse_c_sample_zip(zip_bytes, min_rodden=args.rodden)
    print(f"Records (min Rodden {args.rodden}): {len(records)}")

    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
