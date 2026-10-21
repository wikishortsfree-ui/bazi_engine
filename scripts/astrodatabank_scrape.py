"""
Scrape Astrodatabank via Playwright + MediaWiki API.

Uses a headless Chromium browser to bypass Astrodienst's custom JS challenge,
then batch-queries the MediaWiki API to enumerate and fetch birth data.

Phase 1: Enumerate all page titles via allpages API (500/call).
Phase 2: Batch-fetch wikitext via query&prop=revisions (50 pages/call).
Parse birth data from wikitext, filter by Rodden rating, save to JSONL.

Prerequisites:
    pip install playwright
    python -m playwright install chromium

Usage:
    python scripts/astrodatabank_scrape.py --output data/astrodatabank_scraped.jsonl
    python scripts/astrodatabank_scrape.py --output data/astrodatabank_scraped.jsonl --resume
    python scripts/astrodatabank_scrape.py --output data/astrodatabank_scraped.jsonl --skip-existing data/astrodatabank_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.astrodatabank_api import parse_latitude, parse_longitude

WIKI_MAIN_PAGE = "https://www.astro.com/astro-databank/Main_Page"
API_URL = "https://www.astro.com/wiki/astro-databank/api.php"

WIKITEXT_PATTERNS = {
    "name": r"\|Name=([^\n\|]+)",
    "date": r"\|sbdate_dmy=([^\n\|]+)",
    "time": r"\|sbtime=([^\n\|]+)",
    "place": r"\|Place=([^\n\|]+)",
    "longitude": r"\|slong=([^\n\|]+)",
    "latitude": r"\|slati=([^\n\|]+)",
    "rodden": r"\|sroddenrating=([^\n\|]+)",
    "source": r"\|sdatasource=([^\n\|]+)",
}

RODDEN_ACCEPTABLE = {"AA", "A"}


def _parse_wikitext(wikitext: str) -> dict | None:
    result = {}
    for field, pattern in WIKITEXT_PATTERNS.items():
        match = re.search(pattern, wikitext, re.IGNORECASE)
        if match:
            result[field] = match.group(1).strip()
    return result if result else None


def _extract_biography(wikitext: str) -> str:
    match = re.search(
        r"==\s*Biography\s*==\s*(.*?)(?:\n==|\Z)",
        wikitext,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        text = match.group(1).strip()
        text = re.sub(r"\[\[(?:[^\]]*\|)?([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"'{2,}", "", text)
        return text.strip()[:5000]
    return ""


def _build_record(title: str, wikitext: str, min_rodden: str) -> dict | None:
    fields = _parse_wikitext(wikitext)
    if not fields:
        return None

    rodden = (fields.get("rodden") or "").strip().upper()
    if min_rodden == "AA" and rodden != "AA":
        return None
    if min_rodden == "A" and rodden not in RODDEN_ACCEPTABLE:
        return None

    lng = parse_longitude(fields.get("longitude", ""))
    lat = parse_latitude(fields.get("latitude", ""))
    if lng is None or lat is None:
        return None

    biography = _extract_biography(wikitext)
    safe_title = urllib.parse.quote(title.replace(" ", "_"), safe="/:")

    return {
        "name": fields.get("name", title),
        "date": fields.get("date", ""),
        "time": fields.get("time", ""),
        "longitude": lng,
        "latitude": lat,
        "location": fields.get("place", ""),
        "rodden_rating": rodden,
        "category": "",
        "biography": biography,
        "source": fields.get("source", ""),
        "source_url": f"https://www.astro.com/astro-databank/{safe_title}",
    }


def _load_existing_names(path: str) -> set[str]:
    names = set()
    p = Path(path)
    if not p.exists():
        return names
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                names.add(r.get("name", ""))
            except json.JSONDecodeError:
                continue
    return names


class AstrodatabankScraper:
    def __init__(self, delay: float = 1.0, headless: bool = True):
        self.delay = delay
        self.headless = headless
        self._page = None
        self._browser = None
        self._pw = None

    def _is_alive(self) -> bool:
        try:
            if self._page is None or self._browser is None:
                return False
            self._page.title()
            return True
        except Exception:
            return False

    def _restart_browser(self):
        print("  Browser lost, restarting...")
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page()

    def start(self):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page()
        self._pass_challenge()

    def stop(self):
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def _pass_challenge(self, max_retries: int = 5):
        for attempt in range(max_retries):
            try:
                if not self._is_alive():
                    self._restart_browser()
                print(f"Navigating to wiki to pass JS challenge (attempt {attempt+1})...")
                self._page.goto(WIKI_MAIN_PAGE, wait_until="networkidle", timeout=60000)
                title = self._page.title()
                if "One moment" in title or "Checking" in title:
                    print("  Challenge page detected, waiting for resolution...")
                    self._page.wait_for_function(
                        "() => !document.title.includes('One moment')", timeout=30000
                    )
                title = self._page.title()
                print(f"  Page title: {title}")
                if "astro" in title.lower() or "birth data" in title.lower():
                    print("  Challenge passed.")
                    return
                print(f"  Unexpected title, retrying...")
            except Exception as e:
                print(f"  Error during challenge (attempt {attempt+1}): {e}")
            wait = 5 * (attempt + 1)
            print(f"  Waiting {wait}s before retry...")
            time.sleep(wait)
        raise RuntimeError("Failed to pass challenge after all retries.")

    def _api_call(self, params: dict, retries: int = 3) -> dict | None:
        qs = urllib.parse.urlencode(params)
        url = f"{API_URL}?{qs}"

        for attempt in range(retries):
            time.sleep(self.delay)
            if not self._is_alive():
                self._restart_browser()
                self._pass_challenge(max_retries=3)
            try:
                result = self._page.evaluate(
                    """async (url) => {
                        try {
                            const resp = await fetch(url, {credentials: 'include'});
                            const text = await resp.text();
                            return {ok: resp.ok, status: resp.status, body: text};
                        } catch(e) {
                            return {ok: false, status: 0, body: e.toString()};
                        }
                    }""",
                    url,
                )
            except Exception as e:
                print(f"  fetch() error (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    self._restart_browser()
                    self._pass_challenge(max_retries=3)
                continue

            if result and result.get("ok"):
                body = result.get("body", "")
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    if "One moment" in body or "Checking" in body:
                        print(f"  Challenge in fetch response (attempt {attempt+1}), re-solving...")
                        self._pass_challenge()
                        continue
                    print(f"  Non-JSON response (attempt {attempt+1}), status={result.get('status')}")
            else:
                status = result.get("status", "?") if result else "?"
                body_preview = (result.get("body", "")[:200] if result else "no result")
                print(f"  fetch failed (attempt {attempt+1}), status={status}: {body_preview}")
                if attempt < retries - 1:
                    self._pass_challenge()

        return None

    def enumerate_all_pages(self) -> list[str]:
        """Phase 1: Get all page titles via allpages API."""
        all_titles = []
        apcontinue = None
        call_count = 0

        while True:
            params = {
                "action": "query",
                "list": "allpages",
                "aplimit": "500",
                "format": "json",
            }
            if apcontinue:
                params["apcontinue"] = apcontinue

            call_count += 1
            data = self._api_call(params)
            if data is None:
                print(f"  Failed to fetch allpages (call #{call_count}), stopping enumeration.")
                break

            pages = data.get("query", {}).get("allpages", [])
            titles = [p["title"] for p in pages if "title" in p]
            all_titles.extend(titles)

            cont = data.get("continue", {})
            apcontinue = cont.get("apcontinue")
            print(f"  allpages call #{call_count}: +{len(titles)} titles (total: {len(all_titles)})", end="\r")

            if not apcontinue:
                break

        print(f"\nPhase 1 complete: {len(all_titles)} page titles in {call_count} calls.")
        return all_titles

    def batch_fetch_wikitext(self, titles: list[str], batch_size: int = 50) -> dict[str, str]:
        """Phase 2: Batch-fetch wikitext for multiple pages at once."""
        results = {}
        total = len(titles)

        for i in range(0, total, batch_size):
            batch = titles[i : i + batch_size]
            titles_param = "|".join(batch)

            params = {
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "titles": titles_param,
                "format": "json",
            }

            data = self._api_call(params)
            if data is None:
                print(f"\n  Failed batch at index {i}, skipping {len(batch)} titles.")
                continue

            pages = data.get("query", {}).get("pages", {})
            for _pid, page_data in pages.items():
                t = page_data.get("title", "")
                revisions = page_data.get("revisions")
                if not revisions:
                    continue
                slots = revisions[0].get("slots", {})
                main_slot = slots.get("main", {})
                content = main_slot.get("*", "")
                if content:
                    results[t] = content

            fetched = min(i + batch_size, total)
            print(f"  Fetched wikitext: {fetched}/{total} ({len(results)} with content)", end="\r")

        print(f"\nPhase 2 complete: {len(results)} pages with wikitext out of {total} requested.")
        return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape Astrodatabank via Playwright + MediaWiki API"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/astrodatabank_scraped.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--pages-cache",
        type=str,
        default="data/astrodatabank_pages.json",
        help="Cache file for page titles (Phase 1)",
    )
    parser.add_argument(
        "--skip-existing",
        type=str,
        default="",
        help="JSONL file with existing records to skip (by name)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume: skip names already in the output file",
    )
    parser.add_argument(
        "--rodden",
        type=str,
        default="A",
        choices=["AA", "A"],
        help="Minimum Rodden rating",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between API calls",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Pages per batch API call (max 50)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser in visible mode (for debugging)",
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Only run Phase 1 (enumerate pages), skip Phase 2",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages_cache = Path(args.pages_cache)
    pages_cache.parent.mkdir(parents=True, exist_ok=True)

    skip_names: set[str] = set()
    if args.skip_existing:
        skip_names |= _load_existing_names(args.skip_existing)
        print(f"Loaded {len(skip_names)} names to skip from {args.skip_existing}")
    if args.resume:
        skip_names |= _load_existing_names(args.output)
        print(f"Resume: {len(skip_names)} total names to skip")

    scraper = AstrodatabankScraper(
        delay=args.delay, headless=not args.no_headless
    )

    try:
        scraper.start()

        # Phase 1
        if pages_cache.exists():
            print(f"Loading cached page titles from {pages_cache}...")
            with open(pages_cache, "r") as f:
                all_titles = json.load(f)
            print(f"  Loaded {len(all_titles)} titles from cache.")
        else:
            print("Phase 1: Enumerating all page titles...")
            all_titles = scraper.enumerate_all_pages()
            with open(pages_cache, "w") as f:
                json.dump(all_titles, f)
            print(f"  Saved {len(all_titles)} titles to {pages_cache}")

        if args.phase1_only:
            print("Phase 1 only mode. Exiting.")
            return 0

        titles_to_fetch = [t for t in all_titles if t not in skip_names]
        print(f"Phase 2: {len(titles_to_fetch)} titles to fetch ({len(all_titles) - len(titles_to_fetch)} skipped)")

        if not titles_to_fetch:
            print("Nothing to fetch.")
            return 0

        # Phase 2: batch fetch and parse
        saved = 0
        batch_size = min(args.batch_size, 50)
        total = len(titles_to_fetch)

        with open(output_path, "a", encoding="utf-8") as f_out:
            for chunk_start in range(0, total, batch_size * 10):
                chunk_end = min(chunk_start + batch_size * 10, total)
                chunk_titles = titles_to_fetch[chunk_start:chunk_end]

                wikitext_map = scraper.batch_fetch_wikitext(
                    chunk_titles, batch_size=batch_size
                )

                for title, wikitext in wikitext_map.items():
                    record = _build_record(title, wikitext, args.rodden)
                    if record is None:
                        continue
                    if record["name"] in skip_names:
                        continue
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    skip_names.add(record["name"])
                    saved += 1

                f_out.flush()
                print(f"  Progress: {min(chunk_end, total)}/{total} processed, {saved} saved")

        print(f"\nDone. Saved {saved} new records to {output_path}")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")
        return 1
    finally:
        scraper.stop()


if __name__ == "__main__":
    sys.exit(main())
