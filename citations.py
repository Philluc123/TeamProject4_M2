"""
Citation guardrail for SignalBrief.

The model is told to cite sources ONLY by the short `ref` ids that the MCP
tools return (e.g. [n-3f2a9]). This module turns those ids back into real
links and throws away any bullet that cites nothing a tool actually returned.
Kept free of the Agents SDK so it can be unit-tested on its own.
"""

from __future__ import annotations

import json
import re

# a citation id looks like n-3f2a9 (news), p-81c0d (paper), m-0b7e2 (music)
REF_PATTERN = re.compile(r"\b([nmp]-[0-9a-f]{5})\b")
# a bracket group holding one or more ids: [n-3f2a9] / [n-3f2a9, p-81c0d] / (n-3f2a9) / \u3010n-3f2a9\u3011
REF_GROUP_PATTERN = re.compile(
    r"[\[(\u3010]\s*[nmp]-[0-9a-f]{5}(?:[\s,;]+[nmp]-[0-9a-f]{5})*\s*[\])\u3011]")
BULLET_PATTERN = re.compile(r"^(?:[-*\u2022]\s+|\d+[.)]\s+)")


def sources_from_tool_outputs(outputs: list[str]) -> dict:
    """Build {ref: record} from raw MCP tool output strings."""
    sources: dict[str, dict] = {}
    for text in outputs:
        # MCP results may arrive wrapped, e.g. {"type":"text","text":"{...}"}
        for blob in _json_candidates(text):
            for key in ("articles", "papers", "releases"):
                for rec in blob.get(key, []) or []:
                    if isinstance(rec, dict) and rec.get("ref") and rec.get("url"):
                        sources[rec["ref"]] = rec
    return sources


def _json_candidates(text: str):
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return
    if isinstance(obj, dict):
        if "text" in obj and isinstance(obj["text"], str):
            yield from _json_candidates(obj["text"])
        yield obj
    elif isinstance(obj, list):
        for part in obj:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                yield from _json_candidates(part["text"])


def verify_and_link(section_text: str, sources: dict) -> tuple[str, dict]:
    """Replace [ref] ids with real links; drop bullets that cite nothing real.

    This is the anti-hallucination guardrail: the check is done in code,
    so a claim only reaches the reader if it points at a source a tool
    actually returned during this run.
    """
    stats = {"kept": 0, "dropped": 0, "bad_refs": 0}
    out_lines = []
    for line in section_text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        bullet = BULLET_PATTERN.match(stripped)
        if not bullet:
            # non-bullet text: keep only the explicit 'no updates' line
            if "no notable updates" in stripped.lower():
                out_lines.append("_No notable updates in this window._")
            continue
        stripped = "- " + stripped[bullet.end():].strip()
        refs = REF_PATTERN.findall(stripped)
        good = [r for r in refs if r in sources]
        stats["bad_refs"] += len(refs) - len(good)
        if not good:
            stats["dropped"] += 1
            continue
        body = REF_GROUP_PATTERN.sub("", stripped).rstrip()
        links = []
        for r in dict.fromkeys(good):  # de-duplicate, keep order
            s = sources[r]
            label = s.get("source") or s.get("journal") or "MusicBrainz"
            links.append(f"[{label}, {s.get('published', 'n.d.')}]({s['url']})")
        out_lines.append(f"{body} ({'; '.join(links)})")
        stats["kept"] += 1
    if not out_lines:
        out_lines.append("_No notable updates in this window._")
    return "\n".join(out_lines), stats
