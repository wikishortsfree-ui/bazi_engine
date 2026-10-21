"""
Astrodatabank MediaWiki API Client
Retrieves high-quality birth data for bāzì training dataset

Usage:
    python astrodatabank_api.py --target 2500 --output astrodatabank_raw.jsonl

Note: astro.com may return HTML ("Checking your browser") instead of JSON for script
requests. If you see "API returned non-JSON" or connection errors, see
docs/astrodatabank_api_notes.md for causes and the adbexport (c_sample) alternative.
"""

import argparse
import csv
import json
import os
import re
import time
import urllib.parse
from collections import Counter
from html.parser import HTMLParser

import requests

ASTRODATABANK_API_URL = "https://www.astro.com/astro-databank/api.php"
REQUEST_TIMEOUT = 30


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_text(self):
        return " ".join(self.text)


def _api_get(params, verify_ssl=True):
    response = requests.get(
        ASTRODATABANK_API_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
        verify=verify_ssl,
    )
    response.raise_for_status()
    try:
        return response.json()
    except requests.JSONDecodeError as e:
        raise ValueError(
            f"API returned non-JSON (status {response.status_code}). "
            "astro.com may be serving a browser-check page. See docs/astrodatabank_api_notes.md."
        ) from e


def get_category_members(category, limit=500, verify_ssl=True):
    """
    Get all Astrodatabank entries in a category.

    Args:
        category: Category name (e.g., 'Technology', 'Business')
        limit: Max number of members to retrieve

    Returns:
        Dict with category members
    """
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": min(limit, 500),
        "format": "json",
    }
    return _api_get(params, verify_ssl=verify_ssl)


def get_birth_data(page_title, verify_ssl=True):
    """
    Extract structured birth data from Astrodatabank entry.

    Returns: Dict with name, date, time, location, coordinates, Rodden rating
    """
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "wikitext",
        "format": "json",
    }
    data = _api_get(params, verify_ssl=verify_ssl)
    parse = data.get("parse")
    if not parse:
        return None
    wikitext_obj = parse.get("wikitext")
    if not wikitext_obj or "*" not in wikitext_obj:
        return None
    wikitext = wikitext_obj["*"]

    patterns = {
        "name": r"\|name=([^\n\|]+)",
        "date": r"\|date=([^\n\|]+)",
        "time": r"\|time=([^\n\|]+)",
        "place": r"\|place=([^\n\|]+)",
        "longitude": r"\|long=([^\n\|]+)",
        "latitude": r"\|lat=([^\n\|]+)",
        "rodden": r"\|rodden=([^\n\|]+)",
        "source": r"\|source=([^\n\|]+)",
    }
    result = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, wikitext, re.IGNORECASE)
        if match:
            result[field] = match.group(1).strip()
    return result


def _parse_dms(s, negative_char, allowed_directions):
    if not s or not isinstance(s, str):
        return None
    match = re.match(r"(\d+)([a-z])(\d+)", s.strip().lower())
    if not match:
        return None
    degrees = int(match.group(1))
    direction = match.group(2)
    minutes = int(match.group(3))
    if direction not in allowed_directions or minutes < 0 or minutes >= 60:
        return None
    decimal = degrees + minutes / 60.0
    if direction == negative_char:
        decimal = -decimal
    return decimal


def parse_longitude(long_str):
    """
    Parse Astrodatabank longitude format.
    Examples: '122w25' = 122°25' West = -122.4167
    """
    value = _parse_dms(long_str, "w", ("e", "w"))
    if value is None:
        return None
    if value < -180 or value > 180:
        return None
    return value


def parse_latitude(lat_str):
    """
    Parse Astrodatabank latitude format.
    Examples: '37n47' = 37°47' North = +37.7833
    """
    value = _parse_dms(lat_str, "s", ("n", "s"))
    if value is None:
        return None
    if value < -90 or value > 90:
        return None
    return value


def get_biography_text(page_title, verify_ssl=True):
    """Extract biography text for event extraction."""
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "text",
        "format": "json",
    }
    data = _api_get(params, verify_ssl=verify_ssl)
    parse = data.get("parse")
    if not parse:
        return ""
    text_obj = parse.get("text")
    if not text_obj or "*" not in text_obj:
        return ""
    parser = _TextExtractor()
    parser.feed(text_obj["*"])
    return parser.get_text()


def _category_targets(target_count):
    base = [
        ("Technology", 300),
        ("Business", 300),
        ("Politics", 300),
        ("Sports", 300),
        ("Arts", 300),
        ("Science", 200),
        ("Entertainment", 300),
        ("Military", 100),
        ("Religion", 100),
        ("Crime", 100),
    ]
    total_base = sum(t for _, t in base)
    if target_count >= total_base:
        return base
    scale = target_count / total_base
    return [(cat, max(1, int(round(t * scale)))) for cat, t in base]


def collect_astrodatabank_dataset(target_count=2500, min_rodden="A", verify_ssl=True):
    """
    Collect diverse, high-quality charts from Astrodatabank.

    Args:
        target_count: Target number of charts to collect (stops when reached).
        min_rodden: Minimum Rodden rating ('AA' or 'A').

    Returns:
        List of chart dictionaries.
    """
    categories = _category_targets(target_count)
    all_charts = []

    for category, target in categories:
        if len(all_charts) >= target_count:
            break
        cap = min(target, target_count - len(all_charts))

        print(f"\n{'='*60}")
        print(f"Fetching {category} (target: {cap}, total so far: {len(all_charts)})...")
        print(f"{'='*60}")

        try:
            members = get_category_members(category, limit=cap * 2, verify_ssl=verify_ssl)
        except (requests.RequestException, ValueError) as e:
            print(f"Error fetching category {category}: {e}")
            continue

        if "query" not in members or "categorymembers" not in members["query"]:
            print(f"No members found for {category}")
            continue

        category_charts = []
        member_list = members["query"]["categorymembers"]
        total_members = len(member_list)

        for i, member in enumerate(member_list):
            title = member.get("title")
            if not title:
                continue
            if i % 10 == 0:
                print(
                    f"Progress: {i}/{total_members} ({len(category_charts)} valid)",
                    end="\r",
                )

            try:
                birth_data = get_birth_data(title, verify_ssl=verify_ssl)
                if not birth_data:
                    continue

                rodden = birth_data.get("rodden", "").strip().upper()
                if min_rodden == "AA" and rodden != "AA":
                    continue
                if min_rodden == "A" and rodden not in ("AA", "A"):
                    continue

                longitude = parse_longitude(birth_data.get("longitude", ""))
                latitude = parse_latitude(birth_data.get("latitude", ""))
                if longitude is None or latitude is None:
                    continue

                biography = get_biography_text(title, verify_ssl=verify_ssl)

                safe_title_url = urllib.parse.quote(title.replace(" ", "_"), safe="/")
                category_charts.append({
                    "name": birth_data.get("name", title),
                    "date": birth_data.get("date", ""),
                    "time": birth_data.get("time", ""),
                    "longitude": longitude,
                    "latitude": latitude,
                    "location": birth_data.get("place", ""),
                    "rodden_rating": rodden,
                    "category": category,
                    "biography": biography,
                    "source": birth_data.get("source", ""),
                    "source_url": f"https://www.astro.com/astro-databank/{safe_title_url}",
                })

                if len(category_charts) >= cap or len(all_charts) + len(category_charts) >= target_count:
                    break
            except (requests.RequestException, ValueError, KeyError) as e:
                print(f"\nError processing {title!r}: {e}")
                continue

        all_charts.extend(category_charts)
        print(f"\n✓ Collected {len(category_charts)} from {category} (total: {len(all_charts)})")
        time.sleep(2)

    return all_charts[:target_count]


def save_and_report(charts, output_file="data/astrodatabank_raw.jsonl"):
    """Save charts and print statistics."""
    summary_file = output_file.replace(".jsonl", "_summary.csv")
    tmp_file = output_file + ".tmp"

    with open(tmp_file, "w", encoding="utf-8") as f:
        for chart in charts:
            f.write(json.dumps(chart, ensure_ascii=False) + "\n")
    os.replace(tmp_file, output_file)

    print(f"\n{'='*60}")
    print("COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total collected: {len(charts)} charts")
    print(f"Saved to: {output_file}")

    category_dist = Counter(c["category"] for c in charts)
    rodden_dist = Counter(c["rodden_rating"] for c in charts)

    print("\nCategory distribution:")
    for cat, count in category_dist.most_common():
        print(f"  {cat:20s}: {count:4d}")
    print("\nRodden rating distribution:")
    for rating, count in rodden_dist.most_common():
        print(f"  {rating:5s}: {count:4d}")

    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "count"])
        writer.writeheader()
        for cat, count in category_dist.most_common():
            writer.writerow({"category": cat, "count": count})
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    import urllib3
    parser = argparse.ArgumentParser(
        description="Collect Astrodatabank charts via MediaWiki API"
    )
    parser.add_argument("--target", type=int, default=2500, help="Target number of charts")
    parser.add_argument(
        "--rodden",
        type=str,
        default="A",
        choices=["AA", "A"],
        help="Minimum Rodden rating",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/astrodatabank_raw.jsonl",
        help="Output file",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL verification (use only if certificate errors block requests)",
    )
    args = parser.parse_args()
    if args.no_verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    print("Astrodatabank Collection")
    print(f"Target: {args.target} charts")
    print(f"Min Rodden rating: {args.rodden}")
    print(f"Output: {args.output}")

    charts = collect_astrodatabank_dataset(
        target_count=args.target,
        min_rodden=args.rodden,
        verify_ssl=not args.no_verify_ssl,
    )
    save_and_report(charts, args.output)
