import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import re

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
)
from dotenv import load_dotenv
from google import genai

from status_codes import STATUS_BY_NAME  # your cd_std / cd_name / cd_desc mapping


# ----------------------------------------------------------
# Paths & env
# ----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
URLS_FILE = BASE_DIR / "urls.txt"

load_dotenv()
load_dotenv(BASE_DIR / ".env")

HEADLESS = os.getenv("HEADLESS", "true").lower() == "true"
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "output.json")
DELAY_BEFORE_RETURN_HTML = float(os.getenv("DELAY_BEFORE_RETURN_HTML", "2.0"))
MAX_CONTENT_CHARS = int(os.getenv("MAX_CONTENT_CHARS", "20000"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_MAX_ATTEMPTS = int(os.getenv("GEMINI_MAX_ATTEMPTS", "2"))


# ----------------------------------------------------------
# Gemini client
# ----------------------------------------------------------
def configure_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. "
            "Put it in your .env file or export it in your shell."
        )
    return genai.Client(api_key=api_key)


# ----------------------------------------------------------
# Crawl4AI JS interactions (scroll + show more + show contact)
# ----------------------------------------------------------
def build_js_commands() -> list[str]:
    """
    1. Scroll top -> bottom to trigger lazy loading.
    2. Click all relevant "show more" / "show contact number" / "call"/"whatsapp" buttons.
       We handle multiple such buttons by iterating over them.
    """
    return [
        "window.scrollTo(0, 0);",
        "window.scrollTo(0, document.body.scrollHeight);",
        """
        (function () {
          const buttons = Array.from(
            document.querySelectorAll("button, a[role='button'], div[role='button']")
          );

          const wantShowMore = ["show more"];
          const wantContact = ["show contact number", "show contact", "show phone number"];
          const wantCall = ["call", "whatsapp", "chat"];

          function clickByKeywords(keywords, flagName) {
            buttons.forEach(btn => {
              const txt = (btn.innerText || btn.textContent || "").toLowerCase().trim();
              if (!txt) return;
              if (btn.dataset[flagName]) return;
              if (keywords.some(k => txt.includes(k))) {
                btn.dataset[flagName] = "1";
                try {
                  btn.scrollIntoView({behavior:"instant", block:"center"});
                } catch (e) {}
                btn.click();
              }
            });
          }

          clickByKeywords(wantShowMore, "__clickedShowMore");
          clickByKeywords(wantContact, "__clickedContact");
          clickByKeywords(wantCall, "__clickedCall");
        })();
        """,
    ]


async def fetch_page_text_and_html(
    url: str, crawler: AsyncWebCrawler
) -> tuple[str, str, float]:
    """
    Returns: (page_text_for_gemini, raw_html, crawl_duration_sec)
    """
    url = url.strip()
    if not url:
        raise ValueError("Empty URL")

    js_commands = build_js_commands()

    # Wait until body looks "listing-ish"
    wait_js = (
        "js:() => {"
        "  const txt = (document.body.innerText || '').replace(/\\s+/g, ' ');"
        "  if (txt.length < 1000) return false;"
        "  const hasListingWords = /(RM\\s*\\d[\\d,. ]*|sq\\.ft|bedroom|bathroom|for sale)/i.test(txt);"
        "  return hasListingWords;"
        "}"
    )

    run_config = CrawlerRunConfig(
        js_code=js_commands,
        wait_for=wait_js,
        wait_for_timeout=20000,
        delay_before_return_html=DELAY_BEFORE_RETURN_HTML,
        scan_full_page=True,
        cache_mode=CacheMode.BYPASS,
        verbose=True,
    )

    print(f"[STEP] Navigating & interacting with: {url}")
    t0 = time.perf_counter()
    result = await crawler.arun(url=url, config=run_config)
    crawl_duration = time.perf_counter() - t0

    print(f"[DEBUG] Crawl success={result.success}, status={getattr(result, 'status_code', None)}")
    print(f"[DEBUG] Final URL from crawler: {getattr(result, 'url', None)!r}")

    if not result.success:
        raise RuntimeError(f"Crawl failed for {url}: {result.error_message}")

    raw_html = result.html or ""
    cleaned_html = result.cleaned_html or ""
    raw_md = getattr(getattr(result, "markdown", None), "raw_markdown", "") or ""
    fit_md = getattr(getattr(result, "markdown", None), "fit_markdown", "") or ""

    print(
        f"[DEBUG] Lengths | html={len(raw_html)}, cleaned_html={len(cleaned_html)}, "
        f"raw_md={len(raw_md)}, fit_md={len(fit_md)}"
    )

    if raw_md:
        page_text = raw_md
        origin = "raw_markdown"
    elif cleaned_html:
        page_text = cleaned_html
        origin = "cleaned_html"
    else:
        page_text = raw_html
        origin = "html"

    snippet = page_text[:500].replace("\n", " ")
    print(f"[DEBUG] Text snippet ({origin}) for {url}: {snippet!r}")

    return page_text, raw_html, crawl_duration


# ----------------------------------------------------------
# Schema & helpers
# ----------------------------------------------------------
FIELD_NAMES = [
    "url",
    "listing_title",
    "project_name",
    "area",
    "state",
    "price",
    "sq_ft",
    "bedrooms",
    "bathrooms",
    "property_type",
    "carpark",
    "floor_range",
    "phone_number",
    "description",
]


def empty_record(url: str) -> dict:
    return {k: (url if k == "url" else None) for k in FIELD_NAMES}


def extract_phone_candidates_from_html(raw_html: str) -> list[str]:
    """
    Tiny helper to fish out possible phone numbers from raw HTML, including
    numbers in scripts / JSON / wasap.my links.
    """
    candidates: list[str] = []

    phone_patterns = [
        r"\b01\d[-\s]?\d{3}[-\s]?\d{4}\b",  # 012-345 6789 / 017 787 0260
        r"\b01\d\d{7,8}\b",                 # 0112345678 / 01123456789
        r"\b0\d{1,2}-\d{6,8}\b",            # 03-12345678
        r"\b6\d{8,11}\b",                   # 60123456789 (intl style)
    ]

    for pat in phone_patterns:
        for m in re.finditer(pat, raw_html):
            val = m.group(0).strip()
            if val not in candidates:
                candidates.append(val)

    return candidates


# ----------------------------------------------------------
# Meta logging
# ----------------------------------------------------------
@dataclass
class GeminiMeta:
    status_key: str
    status_code: str
    gemini_model: str | None = None
    gemini_attempts: int = 0
    gemini_prompt_tokens: int | None = None
    gemini_response_tokens: int | None = None
    gemini_total_tokens: int | None = None
    gemini_duration_sec: float | None = None
    crawl_duration_sec: float | None = None
    total_duration_sec: float | None = None
    timestamp_utc: str | None = None


def _status_row(status_name: str) -> dict:
    """Map cd_name -> row, fall back to UNEXPECTED_ERROR."""
    return STATUS_BY_NAME.get(status_name, STATUS_BY_NAME["UNEXPECTED_ERROR"])


def make_meta(
    status_name: str,
    crawl_duration_sec: float | None,
    total_duration_sec: float | None,
    gemini_model: str | None = None,
    attempts: int = 0,
    usage: dict | None = None,
    gemini_duration_sec: float | None = None,
) -> dict:
    row = _status_row(status_name)
    usage = usage or {}

    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    meta = GeminiMeta(
        status_key=row["cd_name"],
        status_code=row["cd_std"],
        gemini_model=gemini_model,
        gemini_attempts=attempts,
        gemini_prompt_tokens=usage.get("prompt_token_count"),
        gemini_response_tokens=usage.get("candidates_token_count"),
        gemini_total_tokens=usage.get("total_token_count"),
        gemini_duration_sec=gemini_duration_sec,
        crawl_duration_sec=crawl_duration_sec,
        total_duration_sec=total_duration_sec,
        timestamp_utc=ts,
    )
    return asdict(meta)


# ----------------------------------------------------------
# Gemini extraction
# ----------------------------------------------------------
def extract_with_gemini(
    client: genai.Client,
    url: str,
    page_text: str,
    raw_html: str,
    crawl_duration_sec: float,
) -> tuple[dict, dict]:
    t0_total = time.perf_counter()

    phone_candidates = extract_phone_candidates_from_html(raw_html)
    print(f"[DEBUG] Phone candidates from HTML for {url}: {phone_candidates}")

    content = page_text[:MAX_CONTENT_CHARS]

    hints_block = (
        "PHONE CANDIDATES (from full HTML, may include numbers from scripts, links, or hidden widgets):\n"
        f"{phone_candidates}\n\n"
    )

    base_prompt = f"""
You are an assistant that extracts structured data from a SINGLE property listing page on mudah.my.

Use BOTH the page content and the phone candidates list below.
The phone candidates may come from JavaScript, links (e.g. wasap.my/6012...), or other hidden parts of the HTML.

Rules for phone_number:
- You MUST search for phone numbers anywhere in the page, including:
  * main description text
  * "Contact" widgets
  * WhatsApp / tel: links
  * any other visible or hidden text in the HTML
- If there are both masked and full numbers (e.g. "017323****" and "0173238055"),
  you MUST choose the full digit number (0173238055).
- Prefer a single, Malaysian-style phone number (with or without country code).
- You may use the PHONE CANDIDATES list to resolve masked numbers.
- If you only see masked phone numbers (with asterisks) and NO full digit candidate at all,
  then you may return the masked number like "017323****".
- If genuinely no phone number is present anywhere, set phone_number to null.
- Never invent or guess digits that do not appear on the page.

General field rules:
- Extract ONLY what is clearly supported by the content.
- Prefer information specific to THIS listing, not generic area/project blurbs.
- If a field is not clearly present, set it to null.
- Do not hallucinate values.

{hints_block}
Here is the page content (after scrolling and clicking 'show more' and 'Show contact number'):

---------------- PAGE CONTENT START ----------------
{content}
---------------- PAGE CONTENT END ----------------
"""

    response_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "listing_title": {"type": ["string", "null"]},
            "project_name": {"type": ["string", "null"]},
            "area": {"type": ["string", "null"]},
            "state": {"type": ["string", "null"]},
            "price": {"type": ["string", "null"]},
            "sq_ft": {"type": ["string", "null"]},
            "bedrooms": {"type": ["string", "null"]},
            "bathrooms": {"type": ["string", "null"]},
            "property_type": {"type": ["string", "null"]},
            "carpark": {"type": ["string", "null"]},
            "floor_range": {"type": ["string", "null"]},
            "phone_number": {"type": ["string", "null"]},
            "description": {"type": ["string", "null"]},
        },
        "required": ["url"],
    }

    config = {
        "response_mime_type": "application/json",
        "response_json_schema": response_schema,
    }

    attempts = 0
    last_error: Exception | None = None
    usage: dict | None = None
    gemini_duration: float | None = None

    while attempts < GEMINI_MAX_ATTEMPTS:
        attempts += 1
        print(f"[DEBUG] Gemini attempt {attempts} for {url}")
        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=base_prompt,
                config=config,
            )
            gemini_duration = time.perf_counter() - t0

            usage_meta = getattr(response, "usage_metadata", None)
            if usage_meta:
                usage = {
                    "prompt_token_count": getattr(
                        usage_meta, "prompt_token_count", None
                    ),
                    "candidates_token_count": getattr(
                        usage_meta, "candidates_token_count", None
                    ),
                    "total_token_count": getattr(
                        usage_meta, "total_token_count", None
                    ),
                }
                print(
                    f"[DEBUG] Gemini usage for {url} | "
                    f"prompt={usage.get('prompt_token_count')}, "
                    f"response={usage.get('candidates_token_count')}, "
                    f"total={usage.get('total_token_count')}"
                )

            raw = (response.text or "").strip()
            print(f"[DEBUG] Gemini raw response for {url}: {raw[:300]!r}")

            data = json.loads(raw)

            record = empty_record(url)
            for k in FIELD_NAMES:
                if k == "url":
                    record[k] = url
                else:
                    record[k] = data.get(k, None)

            total_duration = time.perf_counter() - t0_total
            meta = make_meta(
                status_name="SUCCESS",
                crawl_duration_sec=crawl_duration_sec,
                total_duration_sec=total_duration,
                gemini_model=GEMINI_MODEL,
                attempts=attempts,
                usage=usage,
                gemini_duration_sec=gemini_duration,
            )
            return record, meta

        except Exception as e:
            last_error = e
            print(f"[WARN] Gemini attempt {attempts} failed for {url}: {e}")
            continue

    # If we’re here, all attempts failed
    total_duration = time.perf_counter() - t0_total
    print(f"[ERROR] Gemini failed after {GEMINI_MAX_ATTEMPTS} attempts for {url}: {last_error}")
    meta = make_meta(
        status_name="GEMINI_CALL_FAILED",
        crawl_duration_sec=crawl_duration_sec,
        total_duration_sec=total_duration,
        gemini_model=GEMINI_MODEL,
        attempts=attempts,
        usage=usage,
        gemini_duration_sec=gemini_duration,
    )
    return empty_record(url), meta


# ----------------------------------------------------------
# Per-URL pipeline
# ----------------------------------------------------------
async def process_url(url: str, crawler: AsyncWebCrawler, client: genai.Client) -> dict:
    print("\n" + "=" * 80)
    print(f"[INFO] Processing URL: {url}")

    try:
        page_text, raw_html, crawl_duration = await fetch_page_text_and_html(url, crawler)
    except Exception as e:
        print(f"[ERROR] Crawl error for {url}: {e}")
        meta = make_meta(
            status_name="CRAWL_FAILED",
            crawl_duration_sec=None,
            total_duration_sec=None,
            gemini_model=GEMINI_MODEL,
            attempts=0,
            usage=None,
            gemini_duration_sec=None,
        )
        rec = empty_record(url)
        rec["meta"] = meta
        return rec

    record, meta = extract_with_gemini(client, url, page_text, raw_html, crawl_duration)
    record["meta"] = meta

    print(
        "[RESULT] " + url + "\n"
        f"  status={meta['status_key']} ({meta['status_code']})\n"
        f"  title={record.get('listing_title')!r}\n"
        f"  price={record.get('price')!r}\n"
        f"  bedrooms={record.get('bedrooms')!r}, bathrooms={record.get('bathrooms')!r}\n"
        f"  property_type={record.get('property_type')!r}\n"
        f"  phone={record.get('phone_number')!r}\n"
        f"  crawl_duration={meta.get('crawl_duration_sec')}s, "
        f"gemini_duration={meta.get('gemini_duration_sec')}s, "
        f"total={meta.get('total_duration_sec')}s, "
        f"timestamp_utc={meta.get('timestamp_utc')}\n"
    )

    return record


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
async def main():
    if not URLS_FILE.exists():
        raise FileNotFoundError(f"urls.txt not found at {URLS_FILE}")

    urls = [
        line.strip()
        for line in URLS_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    if not urls:
        print("No URLs found in urls.txt")
        return

    print(f"[INFO] Loaded {len(urls)} URL(s) from {URLS_FILE}")
    print(f"[INFO] HEADLESS={HEADLESS}, DELAY_BEFORE_RETURN_HTML={DELAY_BEFORE_RETURN_HTML}")
    print(f"[INFO] GEMINI_MAX_ATTEMPTS={GEMINI_MAX_ATTEMPTS}")

    client = configure_gemini_client()

    browser_conf = BrowserConfig(
        headless=HEADLESS,
        verbose=True,
        viewport_width=1280,
        viewport_height=720,
    )

    results: list[dict] = []
    t0_all = time.perf_counter()

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        for url in urls:
            try:
                record = await process_url(url, crawler, client)
            except Exception as e:
                print(f"[ERROR] Unexpected error in pipeline for {url}: {e}")
                meta = make_meta(
                    status_name="SYSTEM_ERROR",
                    crawl_duration_sec=None,
                    total_duration_sec=None,
                    gemini_model=GEMINI_MODEL,
                    attempts=0,
                    usage=None,
                    gemini_duration_sec=None,
                )
                rec = empty_record(url)
                rec["meta"] = meta
                record = rec

            results.append(record)

    total_all = time.perf_counter() - t0_all
    output_path = BASE_DIR / OUTPUT_FILE
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[DONE] Saved {len(results)} records to {output_path}")
    print(f"[DONE] Total run time: {total_all:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
