# SignalBrief — Team Project 4, M2 Agent Prototype v1

**SignalBrief** is a personal daily newsletter agent. You give it up to 10 topics you care about (new VST plugins, an artist's new music, cancer research, …). Every run it researches each topic with live tools, then writes a short, dated Markdown newsletter in which **every bullet links to the source it came from**.

> CEN 4930 AI Agent Studio · Team Project 4 · Milestone 2 prototype. The architecture diagram, market research, competitive landscape, TAM/SAM and Business Model Canvas are in the PDF report (`TeamProject4_M2.pdf`), not this repo.

## What's in the repo

| File | What it is |
|---|---|
| `signalbrief_agent.py` | The agent. Loads topics, runs one agent pass per topic through the Sense → Plan → Act loop, verifies citations, writes `output/newsletter_YYYY-MM-DD.md`. |
| `news_mcp_server.py` | Our **MCP server** (FastMCP, stdio). Exposes 3 tools: `search_news` (Google News RSS), `search_research_papers` (PubMed), `get_artist_releases` (MusicBrainz). All free, no extra API keys. |
| `citations.py` | Citation guardrail: turns the model's `[ref]` ids into real links and drops any bullet that cites nothing a tool returned. |
| `topics.json` | The reader's topic list (max 10). Edit this to personalize. |
| `tests/test_cases.json` | Test case inputs + expected outputs (4 offline, 4 live). |
| `tests/run_tests.py` | Runs the test cases. |
| `requirements.txt`, `.env.example` | Setup. |

## Setup

Python 3.11+.

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env           # macOS/Linux: cp .env.example .env
```

Put your course NRP token in `.env` (`NRP_API_KEY=...`). The model defaults to NRP's `gpt-oss`; set `SIGNALBRIEF_MODEL` in `.env` to try another.

If you already have the course `AI-Agent-Workflows` venv, it already has every package this needs. You can activate it and skip `pip install`.

## Run it

```bash
python signalbrief_agent.py                 # all topics in topics.json
python signalbrief_agent.py --only VST      # just one topic
python signalbrief_agent.py --topics other_topics.json
```

The console shows the agent trace (every LLM call and MCP tool call, with token usage) and a `[check]` line per topic. The newsletter prints at the end and is saved to `output/`.

### Topic format (`topics.json`)

```json
{"reader": "Paul",
 "topics": [
   {"name": "New VST plugins", "kind": "news", "query": "new VST plugin release", "days": 3,
    "why": "Music producer; wants to know when notable synths/effects launch."}
 ]}
```

`kind` is a hint for which tool fits best: `news`, `artist`, or `research`. `days` is how far back to look.

## How it works

```
topics.json ─► signalbrief_agent.py ── for each topic ──► Agent (NRP gpt-oss)
                                                             │  picks tools
                                                             ▼
                                              MCP client (stdio) ─► news_mcp_server.py
                                                                     ├ search_news ─► Google News RSS
                                                                     ├ search_research_papers ─► PubMed
                                                                     └ get_artist_releases ─► MusicBrainz
                        citations.py ◄── agent bullets with [ref] ids + raw tool results
                              │  keep bullets whose refs are real, swap ids for links
                              ▼
                   output/newsletter_YYYY-MM-DD.md
```

**Design choice: the model never types a URL.** Every tool result carries a short `ref` id (for example `n-3f2a9`). The agent cites those ids, and plain Python code swaps them for the real links afterward. If the model invents a source or leaves a claim uncited, that bullet is removed before the newsletter is written, and the footer reports how many were removed.

## Test cases

```bash
python tests/run_tests.py          # O1–O4: offline, no key or internet needed
python tests/run_tests.py --live   # + L1–L4: real tools + NRP model
```

| ID | Input | Expected output |
|---|---|---|
| O1 | Saved Google News feed (4 items: 1 duplicate, 1 older than window) | 2 articles, newest first, source suffix stripped, `n-` refs |
| O2 | Saved MusicBrainz release list | Only the 2 releases dated within 30 days; 2024 album + year-only EP excluded |
| O3 | Saved PubMed summary | PubMed URLs, journal names, `YYYY-MM-DD` dates |
| O4 | Agent text with a real citation, a fake ref, and an uncited claim | Only the real bullet survives, as a markdown link; stats `kept 1 / dropped 2 / bad_refs 1` |
| L1 | News topic "New VST plugins", 7 days | Calls `search_news`; 1–4 cited bullets dated in the window, or the "No notable updates" line; 0 bad refs |
| L2 | Artist topic "Kendrick Lamar", 30 days | Calls `get_artist_releases` (+ `search_news` for features); release links go to MusicBrainz; 0 bad refs |
| L3 | Research topic "cancer immunotherapy clinical trial", 7 days | Calls `search_research_papers` first; PubMed links; framed as single studies, not cures |
| L4 | Nonsense topic "Zorblaxian quantum kazoo" | "_No notable updates in this window._", 0 bullets (no hallucinated news) |

Full inputs and expected outputs: `tests/test_cases.json`.

## Known limitations (v1)

- **Delivery isn't built yet.** The newsletter is a Markdown file plus console output. Email delivery and a daily schedule are planned for M3.
- **Summaries come from headlines and metadata, not full articles.** The agent reads titles, sources and dates, not article bodies, so a bullet can only be as specific as the headline. It can't fact-check the article itself.
- **Google News links are redirect links.** They open the publisher's page, but the URL isn't the publisher's own.
- **Artist features are hit or miss.** MusicBrainz lists an artist's own releases reliably, but features on other people's songs mostly come from news search.
- **Relevance is imperfect.** Broad queries like "new VST plugin" can pull in deal and sale posts. The prompt tells the agent to skip them, but it doesn't always.
- **Model behavior varies.** `gpt-oss` sometimes over-calls tools or formats citations oddly. The guardrail catches bad citations, but a run can still come back thinner than it should. Retry, or set `SIGNALBRIEF_MODEL=qwen3`.
- **No memory between runs yet.** Each day is researched from scratch, so a story can appear on two consecutive days.
- **Rate limits.** Topics run one after another to respect MusicBrainz (1 request/second) and NCBI. Ten topics take a few minutes.

## Team

Team Project 4, CEN 4930 (Fall 2026): Philippe Lucien, Paul Perez, Andrew Shinnick. See the PDF report for roles and contributions.

AI disclosure: code scaffolding and documentation were drafted with help from Claude (Anthropic) and reviewed, run and tested by the team.
