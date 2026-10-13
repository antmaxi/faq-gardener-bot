"""Cluster unanswered questions, grade them against the FAQ, and draft outlines.

LangGraph only branches the run. Retrieval stays in the LlamaIndex index.
Drafts are paste-ready outlines: a heading, the nearest page, example
questions, and TODOs. The model is told not to invent procedures.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .cluster import fingerprint, greedy_clusters
from .store import seen

log = logging.getLogger("gardener")

_LABELS = {"covered", "partial", "missing", "not_a_faq"}
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

GRADE_SYSTEM = """You decide whether a community FAQ wiki already answers a cluster of questions.

Use only the excerpts. Do not use outside knowledge.

Labels:
- covered: the excerpts already answer the questions
- partial: a related section exists but does not answer them
- missing: nothing in the excerpts answers them
- not_a_faq: the questions are one-off chat (a specific trip price, what a named person said, a private plan) and should not become a wiki page

Reply with JSON only: {"label": "covered" | "partial" | "missing" | "not_a_faq"}"""

DRAFT_SYSTEM = """You draft a wiki outline a person will paste into the community FAQ and finish.

Write in the language of the questions.
Do not invent procedures, prices, deadlines, office names, legal rules, addresses, or document lists.
Do not turn outside knowledge into steps.
The outline is a short heading plus a TODO list of facts a person must check or fill in.
You may say that an excerpt does not cover the question. Do not extend the excerpt.

Reply with JSON only: {"heading": "...", "outline": "markdown TODO list"}"""


class State(TypedDict, total=False):
    questions: list[dict]
    clusters: list[dict]
    drafts: list[dict]


def parse_json_object(text: str) -> dict:
    raw = _JSON_FENCE.sub("", text.strip())
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no json object")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("json is not an object")
    return data


def _excerpts_block(excerpts: list[dict]) -> str:
    if not excerpts:
        return "(no FAQ excerpts)"
    blocks = []
    for i, excerpt in enumerate(excerpts, 1):
        blocks.append(
            f"[{i}] {excerpt.get('heading') or ''}\n{excerpt.get('url') or ''}\n{excerpt.get('text') or ''}"
        )
    return "\n\n".join(blocks)


def grade_label(llm, questions: list[str], excerpts: list[dict]) -> str:
    listed = "\n".join(f"- {q}" for q in questions)
    user = (
        "Task: grade\n\n"
        f"Questions:\n{listed}\n\n"
        f"FAQ excerpts:\n{_excerpts_block(excerpts)}"
    )
    try:
        data = parse_json_object(llm.complete(GRADE_SYSTEM, user))
        label = str(data.get("label") or "").strip().lower()
    except (ValueError, json.JSONDecodeError, TypeError):
        log.warning("could not parse grade; treating cluster as partial")
        return "partial"
    if label not in _LABELS:
        log.warning("unknown grade %r; treating cluster as partial", label)
        return "partial"
    if not excerpts and label == "covered":
        return "missing"
    return label


def draft_outline(llm, cluster: dict) -> tuple[str, str]:
    listed = "\n".join(f"- {q}" for q in cluster["questions"])
    nearest = cluster.get("nearest_url") or "none"
    user = (
        "Task: draft\n\n"
        f"Label: {cluster['label']}\n"
        f"Nearest: {nearest}\n\n"
        f"Questions:\n{listed}\n\n"
        f"FAQ excerpts:\n{_excerpts_block(cluster.get('excerpts') or [])}"
    )
    text = llm.complete(DRAFT_SYSTEM, user)
    try:
        data = parse_json_object(text)
    except (ValueError, json.JSONDecodeError, TypeError):
        return "FAQ gap", text.strip()
    heading = str(data.get("heading") or "").strip() or "FAQ gap"
    outline = str(data.get("outline") or "").strip() or text.strip()
    return heading, outline


def compile_graph(
    *,
    index,
    llm,
    conn: sqlite3.Connection,
    min_cluster_size: int,
    cluster_cosine: float,
    top_k: int,
):
    def cluster_node(state: State) -> dict:
        questions = [str(q["text"]) for q in state.get("questions") or [] if q.get("text")]
        if not questions:
            return {"clusters": []}
        vectors = index.embed_queries(questions)
        pending = []
        for group in greedy_clusters(vectors, cluster_cosine):
            if len(group) < min_cluster_size:
                continue
            members = [questions[i] for i in group]
            fp = fingerprint(members)
            if seen(conn, fp):
                log.info("skip already drafted cluster %s", fp[:12])
                continue
            pending.append({"fingerprint": fp, "questions": members})
        return {"clusters": pending}

    def grade_node(state: State) -> dict:
        graded = []
        for cluster in state.get("clusters") or []:
            query = "\n".join(cluster["questions"])
            hits = index.retrieve(query, k=top_k)
            excerpts = [
                {"heading": hit.heading, "url": hit.url, "text": hit.text} for hit in hits
            ]
            label = grade_label(llm, cluster["questions"], excerpts)
            graded.append(
                {
                    **cluster,
                    "label": label,
                    "excerpts": excerpts,
                    "nearest_url": excerpts[0]["url"] if excerpts else None,
                }
            )
        return {"clusters": graded}

    def draft_node(state: State) -> dict:
        drafts = []
        for cluster in state.get("clusters") or []:
            if cluster.get("label") not in {"partial", "missing"}:
                continue
            heading, outline = draft_outline(llm, cluster)
            drafts.append(
                {
                    "fingerprint": cluster["fingerprint"],
                    "label": cluster["label"],
                    "heading": heading,
                    "nearest_url": cluster.get("nearest_url"),
                    "questions": list(cluster["questions"][:3]),
                    "outline": outline,
                }
            )
        return {"drafts": drafts}

    graph = StateGraph(State)
    graph.add_node("cluster", cluster_node)
    graph.add_node("grade", grade_node)
    graph.add_node("draft", draft_node)
    graph.add_edge(START, "cluster")
    graph.add_edge("cluster", "grade")
    graph.add_edge("grade", "draft")
    graph.add_edge("draft", END)
    return graph.compile()
