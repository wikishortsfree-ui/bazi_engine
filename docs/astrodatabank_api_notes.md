# Astrodatabank API – Why Requests Fail and Alternatives

## Why the MediaWiki API requests fail

### 1. Browser / bot verification

- **Observed:** Requests to `https://www.astro.com/astro-databank/api.php` often return HTTP 200 with an **HTML** body (“One moment…”, “Checking your browser…”) instead of JSON.
- **Cause:** astro.com uses a browser-verification / bot-detection layer (e.g. Cloudflare or similar). Scripts and simple HTTP clients are treated as bots and get the HTML challenge page instead of the API response.
- **Result:** `response.json()` in our script fails with “API returned non-JSON (status 200)” because the body is HTML.

### 2. TLS / connection issues (environment-dependent)

- **Observed:** From some environments, `curl` and the in-IDE browser fail to connect to `www.astro.com:443` with:
  - `SSL_ERROR_SYSCALL` (curl)
  - Chrome error page / connection failure (browser)
- **Cause:** Can be due to local SSL trust chain, corporate proxy, firewall, or the server closing connections for non-browser traffic.
- **Result:** Requests never get a valid response, so the script cannot retrieve data.

### 3. Summary

| Check                    | Result |
|--------------------------|--------|
| `api.php?action=help`    | HTML “Checking your browser” (no JSON). |
| `api.php?action=query&...` | Same HTML gate when hit by script/bot. |
| curl to api.php          | SSL/connection error in tested environment. |
| Browser navigate to API  | Connection/error page in tested environment. |
| `adbexport/` index       | Reachable (directory listing and readme fetched). |

So the **MediaWiki API is both behind a browser check and, in some networks, hard or impossible to reach**. The script’s logic is correct; failures are due to environment and bot protection, not to bugs in our request construction.

---

## Official data source: adbexport

Astrodatabank’s **official** programmatic data is the **export area**, not the wiki API:

- **URL:** https://www.astro.com/adbexport/
- **Readme:** https://www.astro.com/adbexport/00_readme.htm

### Public (no license)

- **`c_sample.zip`** – All records with names starting with “C”. Unencrypted, ~5,866 records (e.g. ~3,692 AA-rated). Updated regularly (e.g. Feb 2026).
- **`adb_export_sample.xml`** / **`adb_export_sample.json`** – Small sample for format reference.
- **`c_sample.xml`** – Uncompressed C-sample XML (also listed on the index).

### Full export (license required)

- **`adb_export_YYYYMMDD_HHMM.zip`** – Full database, encrypted; requires a signed license and approval (see readme).

---

## Recommendations

1. **Use the adbexport script (implemented)**
   - **Script:** `scripts/astrodatabank_export_to_jsonl.py`
   - **With download:** Run `python scripts/astrodatabank_export_to_jsonl.py --output data/astrodatabank_raw.jsonl` when your network can reach astro.com. Uses the public **c_sample.zip** from https://www.astro.com/adbexport/c_sample.zip
   - **With local zip:** If the download fails (SSL/network), download **c_sample.zip** in a browser from the link above, then run:
     - `python scripts/astrodatabank_export_to_jsonl.py --output data/astrodatabank_raw.jsonl --input-zip /path/to/c_sample.zip`
   - **Test fixture:** For CI or offline testing, use `tests/fixtures/c_sample_mini.zip` (minimal XML in ADB export format):  
     `python scripts/astrodatabank_export_to_jsonl.py --output data/astrodatabank_raw.jsonl --input-zip tests/fixtures/c_sample_mini.zip`
   - Output matches the schema expected by `astrodatabank_to_bazi.py` (name, date, time, longitude, latitude, location, rodden_rating, category, biography, source, source_url).

2. **If you still want to use the wiki API**
   - Run from a context where a **real browser** can pass the “Checking your browser” step (e.g. same network as a successful browser session).
   - Options (not implemented in this repo):
     - Use a browser-automation stack that executes the verification (e.g. Playwright/Puppeteer), then reuse cookies/headers for `requests`; or
     - Manually copy cookies from a browser that has passed the check and send them with the script (fragile and may break when cookies expire).

3. **Script behavior**
   - `scripts/astrodatabank_api.py` is correct for the **documented** MediaWiki API. When requests fail, it is due to the above (bot check + possible TLS/network). The script already uses timeouts, `raise_for_status()`, and safe JSON parsing; optional `--no-verify-ssl` is only for environments where SSL verification itself is the blocker (e.g. corporate proxy).

---

## References

- Phase 2 data plan: `docs/phase2_data_detailed_plan_FINAL.docx` (Task 2.1 – Astrodatabank integration).
- Export readme: https://www.astro.com/adbexport/00_readme.htm
- Export index: https://www.astro.com/adbexport/
