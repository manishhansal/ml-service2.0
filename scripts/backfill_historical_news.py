#!/usr/bin/env python3
"""
scripts/backfill_historical_news.py
--------------------------------------
Fetch 5 years of historical India financial news from multiple free sources
and inject into SentinelPulse via BullMQ Redis queue for processing.

SentinelPulse has no HTTP ingest endpoint — we push directly to the
`news.raw` BullMQ queue in Redis.  Workers pick them up and run the full
12-stage NLP pipeline: normalize → dedup → entity → event → sentiment →
impact → feature → embed.

Data sources
-------------
  1. Economic Times (source: economic-times)
     • Monthly sitemaps: /sitemap/etmarkets/month-YYYY-MM.xml
     • RSS categories: markets/stocks, markets/nifty, markets/ipo
  2. Moneycontrol (source: moneycontrol)
     • RSS: /rss/marketsnews.xml, /rss/businesesnews.xml, /rss/economy.xml
  3. NSE Corporate Announcements (source: nse-india) — free official API
     • /api/corporate-announcements?index=equities&from_date=...&to_date=...
  4. BSE Corporate Announcements (source: bse-india) — free official API
     • api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w
  5. Business Standard (source: business-standard)
     • RSS: /rss/markets-109.rss, /rss/economy-policy-101.rss
  6. Google News RSS for NSE/India stocks (source: economic-times / reuters)
     • news.google.com/rss/search?q=...&hl=en-IN&gl=IN

Coverage goal
--------------
  ~5,000–50,000 articles from Jan 2021 → Sep 2026.
  The pipeline deduplicates via content hash — no double-processing.

Usage
------
  PYTHONPATH=. python3 scripts/backfill_historical_news.py
  PYTHONPATH=. python3 scripts/backfill_historical_news.py --from-year 2021 --to-year 2026
  PYTHONPATH=. python3 scripts/backfill_historical_news.py --source et
  PYTHONPATH=. python3 scripts/backfill_historical_news.py --source nse --dry-run
  PYTHONPATH=. python3 scripts/backfill_historical_news.py --resume

Note: NSE API requires setting User-Agent to avoid 403.
      ET sitemaps use HTTP/1.1 compression — requests handles transparently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from calendar import monthrange
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────
BASE             = Path(__file__).parent.parent
PROGRESS_FILE    = BASE / "logs" / "news_backfill_progress.json"
LOG_FILE         = BASE / "logs" / "news_backfill.log"
REDIS_HOST       = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT       = int(os.getenv("REDIS_PORT", "6379"))
RATE_SLEEP       = 0.5     # seconds between HTTP requests
BATCH_SLEEP      = 2.0     # seconds between source batches
NSE_HEADERS      = {       # NSE blocks default urllib UA
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
}
BSE_HEADERS      = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# SentinelPulse registered source IDs (must exist in news_sources table)
SP_SOURCES = {
    "economic-times": "Economic Times",
    "moneycontrol":   "Moneycontrol",
    "reuters":        "Reuters",
    "bloomberg":      "Bloomberg",
}
NSE_SOURCE_ID   = "economic-times"   # map NSE announcements to ET source
BSE_SOURCE_ID   = "moneycontrol"     # map BSE announcements to MC source
BS_SOURCE_ID    = "economic-times"   # Business Standard → map to ET

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("news_backfill")


# ── BullMQ Redis injector ─────────────────────────────────────────────────────

class BullMQInjector:
    """Push RawArticle jobs directly into SentinelPulse's news.raw queue."""

    QUEUE    = "news.raw"
    PREFIX   = "bull"
    WAIT_KEY = "bull:news.raw:wait"

    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT) -> None:
        try:
            import redis as _redis
            self._r = _redis.Redis(host=host, port=port, decode_responses=True)
            self._r.ping()
            log.info(f"Connected to Redis {host}:{port}")
        except Exception as exc:
            log.error(f"Redis connection failed: {exc}")
            raise

    def _job_key(self, source_id: str, external_id: str) -> str:
        return f"{self.PREFIX}:{self.QUEUE}:{source_id}:{external_id}"

    def already_queued(self, source_id: str, external_id: str) -> bool:
        return bool(self._r.exists(self._job_key(source_id, external_id)))

    def push(
        self,
        source_id: str,
        source_name: str,
        external_id: str,
        url: str,
        title: str,
        published_at: str,
        summary: str = "",
        content: str = "",
        author: str = "",
        category: str = "",
    ) -> bool:
        """
        Push one article to the news.raw queue.  Returns True if newly queued.

        Args:
            external_id:   Unique identifier within the source (usually the URL).
            published_at:  ISO-8601 UTC string, e.g. "2022-03-15T10:30:00.000Z"
        """
        job_id = f"{source_id}:{external_id}"
        key    = self._job_key(source_id, external_id)

        # Idempotent: skip if the job already exists in Redis
        if self._r.exists(key):
            return False

        now_ms = int(time.time() * 1000)
        raw_article = {
            "sourceId":       source_id,
            "sourceName":     source_name,
            "externalId":     external_id,
            "url":            url,
            "title":          title,
            "summary":        summary[:500] if summary else "",
            "content":        content[:5000] if content else (summary[:500] if summary else ""),
            "author":         author or None,
            "category":       category or None,
            "publishedAt":    published_at,
            "adapterVersion": "1.0.0",
            # Compat fields used by SentinelPulse normalize-worker
            "source_id":      source_id,
            "source_name":    source_name,
            "fetched_at":     datetime.now(tz=timezone.utc).isoformat(),
            "adapter_version": "1.0.0",
        }

        job_data = {
            "name":      self.QUEUE,
            "data":      json.dumps(raw_article, ensure_ascii=False),
            "opts":      json.dumps({
                "jobId":             job_id,
                "removeOnComplete":  {"age": 86400 * 7},
                "removeOnFail":      {"age": 86400 * 30},
                "attempts":          0,
            }),
            "priority":  "0",
            "delay":     "0",
            "timestamp": str(now_ms),
            "ats":       "0",
            "atm":       "1",
        }

        try:
            self._r.hset(key, mapping=job_data)
            self._r.rpush(self.WAIT_KEY, job_id)
            return True
        except Exception as exc:
            log.warning(f"Redis push failed for {job_id}: {exc}")
            return False

    def queue_depth(self) -> int:
        return self._r.llen(self.WAIT_KEY) or 0


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _fetch(url: str, headers: dict | None = None, timeout: int = 15) -> str | None:
    """Fetch URL, return text or None on error."""
    req = urllib.request.Request(url, headers=headers or {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            # Handle gzip
            encoding = resp.info().get("Content-Encoding", "")
            if "gzip" in encoding:
                import gzip as _gz
                raw = _gz.decompress(raw)
            enc = resp.headers.get_content_charset() or "utf-8"
            return raw.decode(enc, errors="replace")
    except Exception as exc:
        log.debug(f"Fetch failed {url[:80]}: {exc}")
        return None


def _parse_dt(raw: str) -> str | None:
    """Parse various date formats → ISO-8601 UTC string."""
    if not raw:
        return None
    raw = raw.strip()
    # RFC 2822 (RSS pubDate: "Wed, 30 Sep 2026 05:40:23 +0530")
    try:
        dt = parsedate_to_datetime(raw)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    except Exception:
        pass
    # ISO 8601 variants
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f+00:00",
        "%Y-%m-%dT%H:%M:%S+00:00",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            # Strip microseconds if format doesn't have them
            s = raw
            if "%f" not in fmt and "." in raw:
                s = raw.split(".")[0]
            if fmt.endswith("Z") and not raw.endswith("Z"):
                continue
            if "+00:00" in fmt and "+00:00" not in raw:
                continue
            dt = datetime.strptime(s.rstrip("Z"), fmt.rstrip("Z"))
            return dt.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        except ValueError:
            continue
    # Locale date formats: "01-Jan-2021", "15/03/2022", "Mar 15, 2022", "20230415"
    locale_fmts = [
        ("%d-%b-%Y %H:%M:%S", None),  # 07-Jan-2023 21:49:37  ← NSE primary format
        ("%d-%b-%Y",          None),  # 01-Jan-2021
        ("%d %b %Y",          None),  # 01 Jan 2021
        ("%b %d, %Y",         None),  # Jan 01, 2021
        ("%d/%m/%Y",          None),  # 01/03/2021
        ("%Y%m%d",            8),     # 20210301 — fixed 8 chars
    ]
    for fmt, max_len in locale_fmts:
        try:
            s = raw if max_len is None else raw[:max_len]
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        except ValueError:
            continue
    return None


def _strip_html(text: str) -> str:
    """Very simple HTML stripper."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;",  "&", text)
    text = re.sub(r"&lt;",   "<", text)
    text = re.sub(r"&gt;",   ">", text)
    text = re.sub(r"\s+",    " ", text).strip()
    return text


# ── Source: Economic Times Monthly Sitemaps ───────────────────────────────────

def _et_sitemap_urls(year: int, month: int) -> list[dict]:
    """Fetch ET Markets monthly sitemap → list of {url, title, published_at}."""
    # Try multiple ET sitemap URL patterns
    urls_to_try = [
        f"https://economictimes.indiatimes.com/sitemap/etmarkets/month-{year}-{month:02d}.xml",
        f"https://economictimes.indiatimes.com/sitemap/et-markets/month-{year}-{month:02d}.xml",
    ]
    xml = None
    for url in urls_to_try:
        xml = _fetch(url)
        if xml and "<urlset" in xml:
            break

    if not xml or "<urlset" not in xml:
        return []

    articles = []
    try:
        # Try multiple namespace declarations
        root = ET.fromstring(xml)
        ns = {
            "sm":   "http://www.sitemaps.org/schemas/sitemap/0.9",
            "news": "http://www.google.com/schemas/sitemap-news/0.9",
        }
        for url_el in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
            loc_el  = url_el.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
            loc     = (loc_el.text or "").strip() if loc_el is not None else ""
            pub_el  = url_el.find(".//{http://www.google.com/schemas/sitemap-news/0.9}publication_date")
            pub_raw = (pub_el.text or "").strip() if pub_el is not None else ""
            ttl_el  = url_el.find(".//{http://www.google.com/schemas/sitemap-news/0.9}title")
            title   = (ttl_el.text or "").strip() if ttl_el is not None else ""

            if not loc:
                continue
            pub = _parse_dt(pub_raw) or f"{year}-{month:02d}-15T00:00:00.000Z"
            articles.append({"url": loc, "title": title, "published_at": pub})
    except ET.ParseError as exc:
        log.debug(f"ET sitemap parse error: {exc}")
    return articles


def yield_et_sitemap(from_year: int, to_year: int) -> Iterator[dict]:
    """Yield ET Markets articles from monthly sitemaps."""
    today = datetime.now()
    for year in range(from_year, to_year + 1):
        max_month = 12 if year < today.year else today.month
        for month in range(1, max_month + 1):
            articles = _et_sitemap_urls(year, month)
            log.info(f"  ET sitemap {year}-{month:02d}: {len(articles)} articles")
            for art in articles:
                yield {
                    "source_id":    "economic-times",
                    "source_name":  "Economic Times",
                    "external_id":  art["url"],
                    "url":          art["url"],
                    "title":        art["title"],
                    "summary":      "",
                    "published_at": art["published_at"],
                }
            time.sleep(RATE_SLEEP)


# ── Source: RSS Feed parser ────────────────────────────────────────────────────

def yield_rss(feed_url: str, source_id: str, source_name: str,
              category: str = "") -> Iterator[dict]:
    """Parse a standard RSS 2.0 or Atom feed."""
    xml = _fetch(feed_url)
    if not xml:
        return
    try:
        root = ET.fromstring(xml)
        # RSS 2.0
        items = root.findall(".//item")
        # Atom
        if not items:
            ns = {"a": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//a:entry", ns)

        for item in items:
            def _t(tag: str) -> str:
                el = item.find(tag)
                return (el.text or "").strip() if el is not None else ""

            title    = _t("title") or _t("{http://www.w3.org/2005/Atom}title")
            link     = _t("link")  or _t("{http://www.w3.org/2005/Atom}link")
            summary  = _strip_html(_t("description") or _t("{http://www.w3.org/2005/Atom}summary"))
            pub_raw  = (_t("pubDate") or _t("published") or
                        _t("{http://www.w3.org/2005/Atom}published"))
            pub      = _parse_dt(pub_raw) or datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

            if not link or not title:
                continue
            yield {
                "source_id":    source_id,
                "source_name":  source_name,
                "external_id":  link,
                "url":          link,
                "title":        title,
                "summary":      summary[:500],
                "published_at": pub,
                "category":     category,
            }
    except ET.ParseError as exc:
        log.debug(f"RSS parse error {feed_url}: {exc}")


def yield_moneycontrol_rss() -> Iterator[dict]:
    """Yield Moneycontrol RSS articles from multiple categories."""
    feeds = [
        ("https://www.moneycontrol.com/rss/marketsnews.xml",     "markets"),
        ("https://www.moneycontrol.com/rss/businesesnews.xml",    "business"),
        ("https://www.moneycontrol.com/rss/economy.xml",          "economy"),
        ("https://www.moneycontrol.com/rss/latestnews.xml",       "latest"),
        ("https://www.moneycontrol.com/rss/technical.xml",        "technical"),
    ]
    for feed_url, cat in feeds:
        items = list(yield_rss(feed_url, "moneycontrol", "Moneycontrol", category=cat))
        log.info(f"  MC RSS {cat}: {len(items)} articles")
        yield from items
        time.sleep(RATE_SLEEP)


def yield_et_rss() -> Iterator[dict]:
    """Yield ET Markets RSS articles from key categories."""
    feeds = [
        # Current ET RSS feed URLs (verified working)
        ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",   "markets"),
        ("https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms","stocks"),
        ("https://economictimes.indiatimes.com/markets/ipo/rssfeeds/9779268.cms",  "ipo"),
        ("https://economictimes.indiatimes.com/economy/rssfeeds/1373380680.cms",   "economy"),
        ("https://economictimes.indiatimes.com/nri/rssfeeds/7716423.cms",          "nri"),
        # Fallback: Google News RSS for ET
        ("https://news.google.com/rss/search?q=site:economictimes.indiatimes.com+stocks+NSE&hl=en-IN&gl=IN&ceid=IN:en", "gnews-et"),
    ]
    for feed_url, cat in feeds:
        items = list(yield_rss(feed_url, "economic-times", "Economic Times", category=cat))
        log.info(f"  ET RSS {cat}: {len(items)} articles")
        yield from items
        time.sleep(RATE_SLEEP)


def yield_business_standard_rss() -> Iterator[dict]:
    """Yield Business Standard RSS (mapped to economic-times source)."""
    feeds = [
        ("https://www.business-standard.com/rss/markets-109.rss",       "markets"),
        ("https://www.business-standard.com/rss/economy-policy-101.rss","economy"),
        ("https://www.business-standard.com/rss/finance-103.rss",       "finance"),
        ("https://www.business-standard.com/rss/companies-101.rss",     "companies"),
    ]
    for feed_url, cat in feeds:
        items = list(yield_rss(feed_url, BS_SOURCE_ID, "Business Standard", category=cat))
        log.info(f"  BS RSS {cat}: {len(items)} articles")
        yield from items
        time.sleep(RATE_SLEEP)


def yield_google_news_rss(query: str, source_id: str, source_name: str) -> Iterator[dict]:
    """Yield India market news from Google News RSS search."""
    enc_q = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={enc_q}&hl=en-IN&gl=IN&ceid=IN:en"
    items = list(yield_rss(url, source_id, source_name, category="markets"))
    log.info(f"  Google News '{query}': {len(items)} articles")
    yield from items


# ── Source: NSE Corporate Announcements ───────────────────────────────────────

def yield_nse_announcements(from_date: str, to_date: str) -> Iterator[dict]:
    """
    Yield NSE corporate announcements as news articles.

    from_date / to_date: "DD-MM-YYYY"
    Endpoint: https://www.nseindia.com/api/corporate-announcements
    """
    url = (
        "https://www.nseindia.com/api/corporate-announcements"
        f"?index=equities&from_date={from_date}&to_date={to_date}"
    )
    text = _fetch(url, headers=NSE_HEADERS)
    if not text:
        log.debug(f"NSE announcements: no data for {from_date}→{to_date}")
        return

    try:
        data = json.loads(text)
        announcements = data if isinstance(data, list) else data.get("data", [])
        for ann in announcements:
            symbol  = ann.get("symbol", "")
            subject = ann.get("desc", "") or ann.get("subject", "")
            company = ann.get("comp", "") or ann.get("companyName", symbol)
            an_dt   = ann.get("an_dt", "") or ann.get("date", "")
            pdf_url = ann.get("attchmntFile", "") or ann.get("attachment", "")
            ext_id  = pdf_url or f"nse-{symbol}-{an_dt}-{hashlib.md5(subject.encode()).hexdigest()[:8]}"

            if not subject:
                continue

            title    = f"{company}: {subject}"[:250]
            pub      = _parse_dt(an_dt) or datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            art_url  = pdf_url or f"https://www.nseindia.com/companies-listing/corporate-filings-announcements"

            yield {
                "source_id":    NSE_SOURCE_ID,
                "source_name":  "Economic Times",  # maps to registered source
                "external_id":  ext_id,
                "url":          art_url,
                "title":        title,
                "summary":      f"NSE Filing: {subject}. Symbol: {symbol}. Company: {company}.",
                "published_at": pub,
                "category":     "corporate-announcement",
            }
    except (json.JSONDecodeError, KeyError) as exc:
        log.debug(f"NSE parse error: {exc}")


def yield_nse_date_range(from_year: int, to_year: int) -> Iterator[dict]:
    """Yield NSE announcements month by month."""
    today = datetime.now()
    for year in range(from_year, to_year + 1):
        max_month = 12 if year < today.year else today.month
        for month in range(1, max_month + 1):
            last_day = monthrange(year, month)[1]
            from_d   = f"01-{month:02d}-{year}"
            to_d     = f"{last_day}-{month:02d}-{year}"
            items = list(yield_nse_announcements(from_d, to_d))
            log.info(f"  NSE {year}-{month:02d}: {len(items)} announcements")
            yield from items
            time.sleep(RATE_SLEEP)


# ── Source: BSE Corporate Announcements ───────────────────────────────────────

def yield_bse_announcements(from_date: str, to_date: str) -> Iterator[dict]:
    """
    Yield BSE corporate announcements.
    from_date / to_date: "YYYYMMDD" format.
    """
    url = (
        "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
        f"?strCat=-1&strPrevDate={from_date}&strScrip=&strSearch=P"
        f"&strToDate={to_date}&strType=C&subcategory=-1"
    )
    text = _fetch(url, headers=BSE_HEADERS)
    if not text:
        return

    try:
        data = json.loads(text)
        tables = data.get("Table", data.get("table", []))
        if not isinstance(tables, list):
            return

        for row in tables:
            company = row.get("SLONGNAME", "") or row.get("SCOMPANYNAME", "")
            subject = row.get("NEWSSUB",   "") or row.get("HEADLINE", "")
            scrip   = row.get("SCRIP_CD",  "") or row.get("SCRIPCD", "")
            dt_raw  = row.get("NEWS_DT",   "") or row.get("NEWSDATE", "")
            pdf_url = row.get("ATTACHMENTNAME", "")

            if not subject:
                continue

            ext_id   = pdf_url or f"bse-{scrip}-{hashlib.md5((subject+dt_raw).encode()).hexdigest()[:10]}"
            title    = f"{company}: {subject}"[:250]
            pub      = _parse_dt(dt_raw) or datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            art_url  = pdf_url or "https://www.bseindia.com/corporates/ann.html"

            yield {
                "source_id":    BSE_SOURCE_ID,
                "source_name":  "Moneycontrol",
                "external_id":  ext_id,
                "url":          art_url,
                "title":        title,
                "summary":      f"BSE Filing: {subject}. Company: {company}. Scrip: {scrip}.",
                "published_at": pub,
                "category":     "corporate-announcement",
            }
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        log.debug(f"BSE parse error: {exc}")


def yield_bse_date_range(from_year: int, to_year: int) -> Iterator[dict]:
    """Yield BSE announcements month by month."""
    today = datetime.now()
    for year in range(from_year, to_year + 1):
        max_month = 12 if year < today.year else today.month
        for month in range(1, max_month + 1):
            last_day = monthrange(year, month)[1]
            from_d   = f"{year}{month:02d}01"
            to_d     = f"{year}{month:02d}{last_day:02d}"
            items = list(yield_bse_announcements(from_d, to_d))
            log.info(f"  BSE {year}-{month:02d}: {len(items)} announcements")
            yield from items
            time.sleep(RATE_SLEEP)


# ── Progress tracking ─────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {"pushed": 0, "skipped": 0, "sources_done": []}


def save_progress(p: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(p, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def run_backfill(
    from_year: int = 2021,
    to_year: int   = 2026,
    sources: list[str] | None = None,
    dry_run: bool  = False,
    resume: bool   = False,
) -> None:
    injector = None if dry_run else BullMQInjector()
    progress = load_progress() if resume else {"pushed": 0, "skipped": 0, "sources_done": []}

    all_sources = sources or ["et-sitemap", "et-rss", "mc-rss", "bs-rss",
                               "gnews", "nse", "bse"]

    log.info("=" * 65)
    log.info("  HISTORICAL NEWS BACKFILL — SentinelPulse BullMQ injection")
    log.info(f"  Date range:  {from_year}–{to_year}")
    log.info(f"  Sources:     {', '.join(all_sources)}")
    log.info(f"  Dry run:     {dry_run}")
    log.info("=" * 65)

    def _push(article: dict) -> None:
        if dry_run:
            progress["pushed"] += 1
            return
        ok = injector.push(
            source_id   = article["source_id"],
            source_name = article["source_name"],
            external_id = article["external_id"],
            url         = article["url"],
            title       = article["title"],
            summary     = article.get("summary", ""),
            content     = article.get("content", ""),
            published_at = article["published_at"],
            category    = article.get("category", ""),
        )
        if ok:
            progress["pushed"] += 1
        else:
            progress["skipped"] += 1

        if (progress["pushed"] + progress["skipped"]) % 500 == 0:
            save_progress(progress)
            depth = injector.queue_depth() if injector else 0
            log.info(
                f"  Progress: pushed={progress['pushed']} "
                f"skipped={progress['skipped']} "
                f"queue_depth={depth}"
            )

    t0 = time.monotonic()

    # ── ET Monthly Sitemaps ───────────────────────────────────────────────────
    if "et-sitemap" in all_sources and "et-sitemap" not in progress["sources_done"]:
        log.info("\n[1] Economic Times monthly sitemaps ...")
        n = 0
        for art in yield_et_sitemap(from_year, to_year):
            _push(art); n += 1
        log.info(f"    ET sitemaps total: {n}")
        progress["sources_done"].append("et-sitemap")
        save_progress(progress)
        time.sleep(BATCH_SLEEP)

    # ── ET RSS Feeds ──────────────────────────────────────────────────────────
    if "et-rss" in all_sources and "et-rss" not in progress["sources_done"]:
        log.info("\n[2] Economic Times RSS feeds ...")
        n = 0
        for art in yield_et_rss():
            _push(art); n += 1
        log.info(f"    ET RSS total: {n}")
        progress["sources_done"].append("et-rss")
        save_progress(progress)
        time.sleep(BATCH_SLEEP)

    # ── Moneycontrol RSS ──────────────────────────────────────────────────────
    if "mc-rss" in all_sources and "mc-rss" not in progress["sources_done"]:
        log.info("\n[3] Moneycontrol RSS feeds ...")
        n = 0
        for art in yield_moneycontrol_rss():
            _push(art); n += 1
        log.info(f"    Moneycontrol RSS total: {n}")
        progress["sources_done"].append("mc-rss")
        save_progress(progress)
        time.sleep(BATCH_SLEEP)

    # ── Business Standard RSS ─────────────────────────────────────────────────
    if "bs-rss" in all_sources and "bs-rss" not in progress["sources_done"]:
        log.info("\n[4] Business Standard RSS feeds ...")
        n = 0
        for art in yield_business_standard_rss():
            _push(art); n += 1
        log.info(f"    Business Standard RSS total: {n}")
        progress["sources_done"].append("bs-rss")
        save_progress(progress)
        time.sleep(BATCH_SLEEP)

    # ── Google News RSS ───────────────────────────────────────────────────────
    if "gnews" in all_sources and "gnews" not in progress["sources_done"]:
        log.info("\n[5] Google News RSS (India markets) ...")
        queries = [
            ("NSE BSE India stocks market",     "economic-times"),
            ("Nifty Sensex India equity",        "economic-times"),
            ("SEBI India regulations",           "economic-times"),
            ("RBI India monetary policy",        "economic-times"),
            ("India FII DII flows NSE",          "moneycontrol"),
            ("India IPO listing NSE BSE",        "moneycontrol"),
        ]
        n = 0
        for query, src in queries:
            for art in yield_google_news_rss(query, src, SP_SOURCES[src]):
                _push(art); n += 1
            time.sleep(RATE_SLEEP)
        log.info(f"    Google News total: {n}")
        progress["sources_done"].append("gnews")
        save_progress(progress)
        time.sleep(BATCH_SLEEP)

    # ── NSE Corporate Announcements ───────────────────────────────────────────
    if "nse" in all_sources and "nse" not in progress["sources_done"]:
        log.info("\n[6] NSE corporate announcements ...")
        n = 0
        for art in yield_nse_date_range(from_year, to_year):
            _push(art); n += 1
        log.info(f"    NSE total: {n}")
        progress["sources_done"].append("nse")
        save_progress(progress)
        time.sleep(BATCH_SLEEP)

    # ── BSE Corporate Announcements ───────────────────────────────────────────
    if "bse" in all_sources and "bse" not in progress["sources_done"]:
        log.info("\n[7] BSE corporate announcements ...")
        n = 0
        for art in yield_bse_date_range(from_year, to_year):
            _push(art); n += 1
        log.info(f"    BSE total: {n}")
        progress["sources_done"].append("bse")
        save_progress(progress)

    elapsed = time.monotonic() - t0
    depth   = injector.queue_depth() if injector else 0
    save_progress(progress)

    log.info("\n" + "=" * 65)
    log.info("  BACKFILL COMPLETE")
    log.info(f"  Articles pushed:    {progress['pushed']}")
    log.info(f"  Already existed:    {progress['skipped']}")
    log.info(f"  Queue depth now:    {depth}")
    log.info(f"  Elapsed:            {elapsed/60:.1f} min")
    log.info(f"  SentinelPulse will process queued articles automatically.")
    log.info("=" * 65)


def main() -> None:
    p = argparse.ArgumentParser(description="Historical news backfill → SentinelPulse")
    p.add_argument("--from-year",  type=int, default=2021)
    p.add_argument("--to-year",    type=int, default=2026)
    p.add_argument("--source",     nargs="+",
                   choices=["et-sitemap","et-rss","mc-rss","bs-rss","gnews","nse","bse"],
                   help="Run specific sources only")
    p.add_argument("--dry-run",    action="store_true",
                   help="Count articles without pushing to Redis")
    p.add_argument("--resume",     action="store_true",
                   help="Resume from last checkpoint (skip completed sources)")
    args = p.parse_args()

    run_backfill(
        from_year = args.from_year,
        to_year   = args.to_year,
        sources   = args.source,
        dry_run   = args.dry_run,
        resume    = args.resume,
    )


if __name__ == "__main__":
    main()
