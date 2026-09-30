"""
SignalBrief MCP server
======================
An MCP server (FastMCP, stdio transport) that gives the newsletter agent
real, dated, linkable sources. Every tool returns structured records that
include a URL and a publish date, so the agent can cite what it says.

Tools
-----
- search_news(query, days, max_results)          -> Google News RSS (no API key)
- search_research_papers(query, days, max_results) -> PubMed / NCBI E-utilities (no API key)
- get_artist_releases(artist, days)               -> MusicBrainz release groups (no API key)

All three data sources are free and keyless on purpose: anyone who clones
the repo can run the agent with only the course NRP key.

Run standalone (for debugging):  python news_mcp_server.py
The agent (signalbrief_agent.py) launches this file itself over stdio.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SignalBrief Sources")

USER_AGENT = "SignalBrief/0.1 (FGCU CEN4930 student project; contact via GitHub repo)"
TIMEOUT_SECONDS = 20
LOG_FILE = Path(__file__).with_name("output") / "mcp_server.log"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def log(message: str) -> None:
    """Append to a log file. Never print(): stdout is the MCP stdio channel."""
    LOG_FILE.parent.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        return resp.read()


def make_ref(prefix: str, url: str) -> str:
    """Short, stable citation id for a source, e.g. 'n-3f2a9'.

    The agent cites sources by this id instead of re-typing long URLs,
    and signalbrief_agent.py swaps each id back to the real link. A model
    can't cite a source it was never given, because unknown ids are rejected.
    """
    return f"{prefix}-{hashlib.sha1(url.encode('utf-8')).hexdigest()[:5]}"


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


# --------------------------------------------------------------------------
# parsers (kept separate from network code so they can be unit-tested offline)
# --------------------------------------------------------------------------
def parse_google_news_rss(xml_bytes: bytes, days: int, max_results: int,
                          now: datetime | None = None) -> list[dict]:
    """Turn a Google News RSS feed into a list of {title, source, url, published}."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    root = ET.fromstring(xml_bytes)
    items: list[dict] = []
    seen_titles: set[str] = set()
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else "Unknown"
        pub_raw = item.findtext("pubDate")
        try:
            published = parsedate_to_datetime(pub_raw) if pub_raw else None
        except (TypeError, ValueError):
            published = None
        if published is None or published < cutoff:
            continue
        # Google News titles end in " - Source"; strip it for readability.
        if source != "Unknown" and title.endswith(f" - {source}"):
            title = title[: -len(f" - {source}")]
        key = title.lower()
        if not title or not link or key in seen_titles:
            continue
        seen_titles.add(key)
        items.append({
            "ref": make_ref("n", link),
            "title": title,
            "source": source,
            "url": link,
            "published": published.strftime("%Y-%m-%d"),
        })
    items.sort(key=lambda x: x["published"], reverse=True)
    return items[:max_results]


def parse_pubmed_summary(summary_json: dict, max_results: int) -> list[dict]:
    """Turn an NCBI esummary JSON payload into {title, journal, url, published}."""
    result = summary_json.get("result", {})
    papers = []
    for pmid in result.get("uids", [])[:max_results]:
        doc = result.get(pmid, {})
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        papers.append({
            "ref": make_ref("p", url),
            "title": doc.get("title", "").strip(),
            "journal": doc.get("fulljournalname") or doc.get("source", ""),
            "url": url,
            "published": doc.get("sortpubdate", "")[:10].replace("/", "-"),
        })
    return papers


def parse_musicbrainz_releases(rg_json: dict, artist_name: str, days: int,
                               now: datetime | None = None) -> list[dict]:
    """Turn MusicBrainz release-group JSON into recent {title, type, url, published}."""
    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    releases = []
    for rg in rg_json.get("release-groups", []):
        date = rg.get("first-release-date") or ""
        if len(date) == 4:          # year-only dates can't be placed in a window
            continue
        if len(date) == 7:
            date += "-01"
        if not date or date < cutoff:
            continue
        kinds = [rg.get("primary-type") or "Release"] + (rg.get("secondary-types") or [])
        url = f"https://musicbrainz.org/release-group/{rg.get('id')}"
        releases.append({
            "ref": make_ref("m", url),
            "title": rg.get("title", ""),
            "artist": artist_name,
            "type": " / ".join(kinds),
            "url": url,
            "published": date,
        })
    releases.sort(key=lambda x: x["published"], reverse=True)
    return releases


# --------------------------------------------------------------------------
# MCP tools
# --------------------------------------------------------------------------
@mcp.tool()
def search_news(query: str, days: int = 3, max_results: int = 6) -> str:
    """Search recent news articles (Google News) for a topic.

    Use this for general topics: product launches (e.g. new VST plugins),
    industry news, company news, sports, tech, etc.
    Returns JSON: a list of articles, each with title, source, url and
    published date (YYYY-MM-DD). Only articles from the last `days` days are
    returned. Cite an article by its `ref` value, e.g. [n-3f2a9].
    An empty list means nothing new was found - do NOT invent news.
    """
    days = clamp(days, 1, 30)
    max_results = clamp(max_results, 1, 10)
    q = urllib.parse.quote_plus(f"{query} when:{days}d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    log(f"search_news query={query!r} days={days}")
    try:
        articles = parse_google_news_rss(http_get(url), days, max_results)
    except Exception as exc:  # network errors are reported, not raised
        log(f"search_news ERROR {exc}")
        return json.dumps({"error": f"news search failed: {exc}", "articles": []})
    log(f"search_news -> {len(articles)} articles")
    return json.dumps({"query": query, "days": days, "articles": articles}, ensure_ascii=False)


@mcp.tool()
def search_research_papers(query: str, days: int = 14, max_results: int = 5) -> str:
    """Search newly published biomedical research papers on PubMed.

    Use this for science / medical topics such as cancer research,
    where peer-reviewed papers are better sources than news articles.
    Returns JSON: a list of papers with title, journal, url (PubMed link)
    and published date. Cite a paper by its `ref` value, e.g. [p-81c0d].
    An empty list means nothing new was indexed.
    """
    days = clamp(days, 1, 60)
    max_results = clamp(max_results, 1, 10)
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    log(f"search_research_papers query={query!r} days={days}")
    try:
        search_url = (f"{base}/esearch.fcgi?db=pubmed&retmode=json&sort=pub_date"
                      f"&datetype=edat&reldate={days}&retmax={max_results}"
                      f"&term={urllib.parse.quote_plus(query)}")
        ids = json.loads(http_get(search_url))["esearchresult"].get("idlist", [])
        if not ids:
            return json.dumps({"query": query, "days": days, "papers": []})
        summary_url = f"{base}/esummary.fcgi?db=pubmed&retmode=json&id={','.join(ids)}"
        papers = parse_pubmed_summary(json.loads(http_get(summary_url)), max_results)
    except Exception as exc:
        log(f"search_research_papers ERROR {exc}")
        return json.dumps({"error": f"PubMed search failed: {exc}", "papers": []})
    log(f"search_research_papers -> {len(papers)} papers")
    return json.dumps({"query": query, "days": days, "papers": papers}, ensure_ascii=False)


@mcp.tool()
def get_artist_releases(artist: str, days: int = 30) -> str:
    """Look up an artist's new albums, singles and EPs on MusicBrainz.

    Use this when a topic is a music artist the user follows. Returns JSON
    with the matched artist name and a list of releases from the last
    `days` days (title, type, url, published date). Cite a release by its
    `ref` value, e.g. [m-0b7e2]. Features on other
    artists' songs are usually NOT listed here - use search_news for those.
    """
    days = clamp(days, 1, 120)
    log(f"get_artist_releases artist={artist!r} days={days}")
    try:
        a_url = ("https://musicbrainz.org/ws/2/artist?fmt=json&limit=1&query="
                 + urllib.parse.quote_plus(f'artist:"{artist}"'))
        matches = json.loads(http_get(a_url)).get("artists", [])
        if not matches:
            return json.dumps({"artist": artist, "matched": None, "releases": []})
        best = matches[0]
        time.sleep(1.1)  # MusicBrainz asks clients to stay under 1 request/second
        rg_url = (f"https://musicbrainz.org/ws/2/release-group?fmt=json&limit=100"
                  f"&type=album|single|ep&artist={best['id']}")
        releases = parse_musicbrainz_releases(json.loads(http_get(rg_url)), best["name"], days)
    except Exception as exc:
        log(f"get_artist_releases ERROR {exc}")
        return json.dumps({"error": f"MusicBrainz lookup failed: {exc}", "releases": []})
    log(f"get_artist_releases -> {len(releases)} releases for {best['name']}")
    return json.dumps({"artist": artist, "matched": best["name"], "days": days,
                       "releases": releases}, ensure_ascii=False)


if __name__ == "__main__":
    # When launched by the agent, stdin/stdout carry the MCP protocol,
    # which is why this file logs to output/mcp_server.log instead of printing.
    mcp.run(transport="stdio")
