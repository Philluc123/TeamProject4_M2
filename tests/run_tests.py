"""
Run SignalBrief's test cases (see tests/test_cases.json for inputs + expected outputs).

  python tests/run_tests.py          # offline cases O1-O4 (no network, no API key)
  python tests/run_tests.py --live   # also live cases L1-L4 (needs NRP_API_KEY + internet)

Live cases check properties (tools used, citations valid, no invented sources)
because the actual news changes every day.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import news_mcp_server as srv  # noqa: E402
from citations import sources_from_tool_outputs, verify_and_link  # noqa: E402

TODAY = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
results: list[tuple[str, bool, str]] = []


def check(case_id: str, condition: bool, detail: str) -> None:
    results.append((case_id, bool(condition), detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {case_id}: {detail}")


# ---------------------------------------------------------------- offline
def test_o1_news_parser():
    items = srv.parse_google_news_rss((FIX / "google_news_sample.xml").read_bytes(), 7, 10, now=TODAY)
    print("  O1 actual:", json.dumps(items, indent=2))
    ok = (len(items) == 2
          and items[0]["title"].startswith("Arturia") and items[0]["published"] == "2026-09-29"
          and items[1]["title"] == "Xfer Records releases Serum 2 update with new wavetables"
          and all(i["ref"].startswith("n-") for i in items))
    check("O1", ok, "2 recent, de-duplicated articles with refs, newest first")


def test_o2_music_parser():
    data = json.loads((FIX / "musicbrainz_rg_sample.json").read_text())
    rel = srv.parse_musicbrainz_releases(data, "Test Artist", 30, now=TODAY)
    print("  O2 actual:", [(r["title"], r["type"], r["published"]) for r in rel])
    ok = ([r["title"] for r in rel] == ["Brand New Single", "Live Set"]
          and rel[1]["type"] == "Album / Live"
          and rel[0]["url"].startswith("https://musicbrainz.org/release-group/"))
    check("O2", ok, "only exactly-dated releases in the last 30 days")


def test_o3_pubmed_parser():
    data = json.loads((FIX / "pubmed_summary_sample.json").read_text())
    papers = srv.parse_pubmed_summary(data, 5)
    print("  O3 actual:", [(p["url"], p["journal"], p["published"]) for p in papers])
    ok = (len(papers) == 2
          and papers[0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/40000001/"
          and papers[0]["journal"] == "The Lancet Oncology"
          and papers[0]["published"] == "2026-09-27"
          and papers[1]["journal"] == "Nat Commun")
    check("O3", ok, "PubMed links, journal names, dates")


def test_o4_guardrail():
    items = srv.parse_google_news_rss((FIX / "google_news_sample.xml").read_bytes(), 7, 10, now=TODAY)
    tool_output = json.dumps({"articles": items})
    # simulate the MCP wrapper the Agents SDK can put around tool text
    wrapped = json.dumps({"type": "text", "text": tool_output})
    sources = sources_from_tool_outputs([wrapped])
    serum_ref = next(i["ref"] for i in items if "Serum" in i["title"])
    agent_text = (
        f"- **Serum 2 update.** Xfer shipped new wavetables. [{serum_ref}]\n"
        "- **Fake news.** A plugin that does not exist launched. [n-fffff]\n"
        "- **Uncited claim.** Something with no source at all.\n"
    )
    md, stats = verify_and_link(agent_text, sources)
    print("  O4 output:\n    " + md.replace("\n", "\n    "), "\n  O4 stats:", stats)
    ok = (stats == {"kept": 1, "dropped": 2, "bad_refs": 1}
          and "[MusicRadar, 2026-09-28](https://news.google.com/rss/articles/SAMPLE_A)" in md
          and "Fake news" not in md and "Uncited" not in md)
    check("O4", ok, "real citation linked, invented + uncited bullets dropped")


# ---------------------------------------------------------------- live
async def run_live():
    import signalbrief_agent as sb
    from agents import Agent
    from agents.mcp import MCPServerStdio, MCPServerStdioParams

    cases = json.loads((ROOT / "tests" / "test_cases.json").read_text())["live"]
    async with MCPServerStdio(
        name="SignalBrief Sources",
        params=MCPServerStdioParams(command=sys.executable, args=[str(sb.SERVER_SCRIPT)]),
        client_session_timeout_seconds=90,
        cache_tools_list=True,
    ) as server:
        agent = Agent(name="SignalBrief", instructions=sb.INSTRUCTIONS,
                      mcp_servers=[server], model=sb.MODEL)
        for case in cases:
            topic = dict(case["input"])
            print(f"\n--- {case['id']}: {case['name']} ---")
            out = await sb.brief_topic(agent, topic)
            print("  section:\n    " + out["markdown"].replace("\n", "\n    "))
            s, tools = out["stats"], out["tools"]
            no_updates = "No notable updates" in out["markdown"]
            base_ok = "error" not in out and s["bad_refs"] == 0 and len(tools) <= 4
            if case["id"] == "L1":
                ok = base_ok and "search_news" in tools and (s["kept"] >= 1 or no_updates)
            elif case["id"] == "L2":
                ok = base_ok and "get_artist_releases" in tools and (s["kept"] >= 1 or no_updates)
            elif case["id"] == "L3":
                ok = (base_ok and tools[:1] == ["search_research_papers"]
                      and (s["kept"] >= 1 or no_updates))
            else:  # L4 nonsense topic
                ok = "error" not in out and s["kept"] == 0 and no_updates
            check(case["id"], ok, f"tools={tools} kept={s['kept']} dropped={s['dropped']} "
                                  f"bad_refs={s['bad_refs']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also run live agent cases (needs NRP key)")
    args = ap.parse_args()

    print("Offline test cases")
    test_o1_news_parser()
    test_o2_music_parser()
    test_o3_pubmed_parser()
    test_o4_guardrail()
    if args.live:
        print("\nLive test cases (real tools + NRP model)")
        asyncio.run(run_live())

    passed = sum(ok for _, ok, _ in results)
    print(f"\n{passed}/{len(results)} test cases passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
