"""
SignalBrief - a personal daily newsletter agent (Team Project 4, M2 prototype)
==============================================================================
Reads up to 10 topics a user cares about (topics.json), researches each one
with live tools exposed by our MCP server (news_mcp_server.py), and writes a
dated Markdown newsletter where every claim is backed by a cited source.

Sense -> Plan -> Act loop, per topic:
  Sense : read the topic + its lookback window from topics.json
  Plan  : the agent picks the right MCP tool (news / PubMed / MusicBrainz)
  Act   : call the tool(s), read dated results, write 0-4 cited bullets
  Check : Python code (not the model) resolves every [ref] to a real URL
          returned by a tool, and drops any bullet it can't verify.

Usage (from the repo folder, with the course venv active):
  python signalbrief_agent.py                      # all topics in topics.json
  python signalbrief_agent.py --only "Serum"       # topics whose name matches
  python signalbrief_agent.py --topics my_topics.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path

from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_trace_processors,
)
from agents.items import ToolCallItem, ToolCallOutputItem
from agents.mcp import MCPServerStdio, MCPServerStdioParams
from agents.tracing import TracingProcessor
from dotenv import load_dotenv
from openai import AsyncOpenAI

from citations import sources_from_tool_outputs, verify_and_link

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
SERVER_SCRIPT = HERE / "news_mcp_server.py"
OUTPUT_DIR = HERE / "output"
MAX_TOPICS = 10

# .env next to this script first, then any .env further up (e.g. course repo)
load_dotenv(HERE / ".env")
load_dotenv()

MODEL = os.getenv("SIGNALBRIEF_MODEL", "gpt-oss")

if not os.getenv("NRP_API_KEY") or os.getenv("NRP_API_KEY") == "your-nrp-token-here":
    sys.exit("NRP_API_KEY not found. Create a file named .env in this folder "
             "(copy .env.example to .env) and put your NRP key in it.")

client = AsyncOpenAI(
    base_url=os.getenv("NRP_BASE_URL", "https://ellm.nrp-nautilus.io/v1"),
    api_key=os.getenv("NRP_API_KEY"),
)
set_default_openai_client(client, use_for_tracing=False)
set_default_openai_api("chat_completions")


class ConsoleTracingProcessor(TracingProcessor):
    """Prints the agent trace (LLM calls, MCP tool calls, token usage)."""

    def on_trace_start(self, trace):
        print(f"\n[trace] '{trace.name}' started")

    def on_trace_end(self, trace):
        print(f"[trace] '{trace.name}' finished")

    def on_span_start(self, span):
        pass

    def on_span_end(self, span):
        data = span.span_data.export()
        summary = f"  [span] {data.get('type', 'unknown')}"
        if data.get("name"):
            summary += f" - {data['name']}"
        if data.get("usage"):
            summary += f" | usage={data['usage']}"
        print(summary)

    def shutdown(self):
        pass

    def force_flush(self):
        pass


set_trace_processors([ConsoleTracingProcessor()])


INSTRUCTIONS = """
You are SignalBrief, a careful research editor who writes one section of a
personal daily newsletter. Your reader wants to know what is NEW about a topic
they care about, with proof. Accuracy beats volume.

Tool choice:
- kind "artist"   -> call get_artist_releases for new albums/singles/EPs, THEN
                     search_news for "<artist> feature" or "<artist> new song" to
                     catch features and announcements.
- kind "research" -> call search_research_papers first; optionally search_news
                     for major news coverage of the same field.
- kind "news"     -> call search_news. If results are thin, retry ONCE with a
                     simpler or alternative query.
Never call more than 3 tools in total for one topic.

Writing rules:
- Output 0 to 4 bullets, most important first. Each bullet is ONE line:
  - **Short headline.** One or two plain-English sentences on what happened and why it matters to this reader. [ref]
- Every bullet MUST end with one or more citation ids copied exactly from the
  `ref` field of tool results, in square brackets, e.g. [n-3f2a9] or [p-81c0d][n-77b10].
- Only state facts that appear in the tool results. Never guess release dates,
  numbers or quotes. Never type URLs - use the ref ids only.
- Skip results that are clearly off-topic, duplicates, or ads/deal posts
  unless a deal is the only news.
- For research papers, say plainly that it is a single study and what kind
  (e.g. "early-stage lab study", "clinical trial") only if the title makes it clear.
- If nothing relevant and new came back, output exactly:
  _No notable updates in this window._
Do not add a heading, intro, or sign-off - just the bullets.
""".strip()

def load_topics(path: Path) -> list[dict]:
    """Load and validate the user's topic list (max 10 topics)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    topics = data["topics"] if isinstance(data, dict) else data
    if not topics:
        raise ValueError("topics file has no topics")
    if len(topics) > MAX_TOPICS:
        print(f"[warn] {len(topics)} topics given; SignalBrief keeps the first {MAX_TOPICS}.")
        topics = topics[:MAX_TOPICS]
    for t in topics:
        t.setdefault("kind", "news")
        t.setdefault("days", 3)
        t.setdefault("query", t["name"])
    return topics


def collect_sources(result) -> tuple[dict, list[str]]:
    """Build {ref: record} from every MCP tool output, plus the list of tool names called."""
    outputs: list[str] = []
    tools_called: list[str] = []
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            tools_called.append(getattr(item.raw_item, "name", "tool"))
        elif isinstance(item, ToolCallOutputItem):
            outputs.append(item.output if isinstance(item.output, str) else json.dumps(item.output))
    return sources_from_tool_outputs(outputs), tools_called


async def brief_topic(agent: Agent, topic: dict) -> dict:
    prompt = (
        f"Today is {date.today().isoformat()}.\n"
        f"Topic name: {topic['name']}\n"
        f"Kind: {topic['kind']}\n"
        f"Search query to start with: {topic['query']}\n"
        f"Lookback window: last {topic['days']} days\n"
        f"Why the reader cares: {topic.get('why', 'not specified')}\n"
        "Write this topic's newsletter bullets now."
    )
    try:
        result = await Runner.run(agent, prompt, max_turns=8)
    except Exception as exc:  # one bad topic should not kill the newsletter
        print(f"[error] topic '{topic['name']}' failed: {exc}")
        return {"topic": topic, "markdown": "_Could not be researched today (agent error)._",
                "stats": {"kept": 0, "dropped": 0, "bad_refs": 0}, "tools": [], "error": str(exc)}
    sources, tools_called = collect_sources(result)
    markdown, stats = verify_and_link(str(result.final_output), sources)
    print(f"[check] {topic['name']}: tools={tools_called} sources={len(sources)} "
          f"kept={stats['kept']} dropped={stats['dropped']} bad_refs={stats['bad_refs']}")
    return {"topic": topic, "markdown": markdown, "stats": stats,
            "tools": tools_called, "sources": len(sources), "raw": str(result.final_output)}


def render_newsletter(sections: list[dict], reader: str) -> str:
    today = date.today()
    kept = sum(s["stats"]["kept"] for s in sections)
    dropped = sum(s["stats"]["dropped"] for s in sections)
    active = [s["topic"]["name"] for s in sections if s["stats"]["kept"]]
    lines = [
        f"# SignalBrief - {today.strftime('%A, %B %d, %Y')}",
        f"*Your daily brief, {reader}: {len(sections)} topics checked, "
        f"{kept} cited updates.*",
        "",
    ]
    if active:
        lines += [f"**New today:** {', '.join(active)}", ""]
    for s in sections:
        t = s["topic"]
        lines += [f"## {t['name']}", f"<sub>last {t['days']} days</sub>", "", s["markdown"], ""]
    lines += [
        "---",
        f"*Verification: {kept} bullets linked to sources returned by live tools this run; "
        f"{dropped} unverifiable bullet(s) removed before sending. "
        "Summaries are AI-generated - open the source before relying on them.*",
    ]
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="SignalBrief daily newsletter agent")
    parser.add_argument("--topics", default=str(HERE / "topics.json"))
    parser.add_argument("--only", help="only run topics whose name contains this text")
    args = parser.parse_args()

    config = json.loads(Path(args.topics).read_text(encoding="utf-8"))
    reader = config.get("reader", "reader") if isinstance(config, dict) else "reader"
    topics = load_topics(Path(args.topics))
    if args.only:
        topics = [t for t in topics if args.only.lower() in t["name"].lower()]
        if not topics:
            sys.exit(f"No topic matches --only {args.only!r}")

    async with MCPServerStdio(
        name="SignalBrief Sources",
        params=MCPServerStdioParams(command=sys.executable, args=[str(SERVER_SCRIPT)]),
        client_session_timeout_seconds=90,
        cache_tools_list=True,
    ) as sources_server:
        agent = Agent(
            name="SignalBrief",
            instructions=INSTRUCTIONS,
            mcp_servers=[sources_server],
            model=MODEL,
        )
        sections = []
        for topic in topics:  # sequential: keeps free APIs + NRP rate limits happy
            print(f"\n=== Researching: {topic['name']} ({topic['kind']}) ===")
            sections.append(await brief_topic(agent, topic))

    newsletter = render_newsletter(sections, reader)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"newsletter_{date.today().isoformat()}.md"
    out_path.write_text(newsletter, encoding="utf-8")
    print("\n" + "=" * 70 + "\n" + newsletter + "\n" + "=" * 70)
    print(f"\nSaved newsletter to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
