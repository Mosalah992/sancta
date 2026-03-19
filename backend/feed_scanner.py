"""
feed_scanner.py — Scan entire Moltbook feeds and extract hot topics.

Aggregates posts from:
  - Global hot feed
  - Global new feed
  - Per-submolt hot feed for target submolts

Deduplicates, ranks by hotness, and extracts trending topics for use by
engage_with_feed, search_and_engage, and trend_hijack.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter
from typing import Any, Callable

log = logging.getLogger("soul")

# Default limits (can be overridden)
DEFAULT_GLOBAL_HOT_LIMIT = 80
DEFAULT_GLOBAL_NEW_LIMIT = 40
DEFAULT_SUBMOLT_HOT_LIMIT = 15
DEFAULT_SUBMOLTS_TO_SCAN = 12  # Top soul-relevant submolts
MAX_TOTAL_POSTS = 400  # Cap deduplicated feed size
HOT_TOPICS_COUNT = 25
MIN_WORD_LEN = 4
STOP_WORDS = frozenset({
    "that", "this", "with", "from", "have", "been", "were", "what",
    "when", "where", "which", "their", "there", "about", "would",
    "could", "should", "just", "like", "only", "some", "more",
    "into", "your", "they", "them", "than", "then", "over", "very",
})


async def scan_full_feed(
    session: Any,
    api_get: Callable[[Any, str], Any],
    target_submolts: list[str],
    *,
    global_hot_limit: int = DEFAULT_GLOBAL_HOT_LIMIT,
    global_new_limit: int = DEFAULT_GLOBAL_NEW_LIMIT,
    submolt_limit: int = DEFAULT_SUBMOLT_HOT_LIMIT,
    submolt_count: int = DEFAULT_SUBMOLTS_TO_SCAN,
) -> tuple[list[dict], list[tuple[str, int]]]:
    """
    Scan full Moltbook feed: global hot, global new, and hot from top submolts.

    Returns:
        (posts, hot_topics)
        - posts: deduplicated list of post dicts, sorted by hotness
        - hot_topics: [(word, count), ...] top trending words
    """
    seen_ids: set[str] = set()
    all_posts: list[dict] = []

    async def fetch(path: str, limit: int) -> list[dict]:
        try:
            data = await api_get(session, path)
            posts = data.get("posts", data.get("data", []))
            return posts[:limit] if isinstance(posts, list) else []
        except Exception as e:
            log.debug("feed_scanner: fetch %s failed: %s", path[:60], e)
            return []

    # 1. Global hot (primary source)
    hot = await fetch(
        f"/posts?sort=hot&limit={global_hot_limit}",
        global_hot_limit,
    )
    for p in hot:
        pid = str(p.get("id", ""))
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            _enrich_post(p, "global_hot", 2.0)
            all_posts.append(p)

    # 2. Global new
    new = await fetch(
        f"/posts?sort=new&limit={global_new_limit}",
        global_new_limit,
    )
    for p in new:
        pid = str(p.get("id", ""))
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            _enrich_post(p, "global_new", 1.0)
            all_posts.append(p)

    # 3. Per-submolt hot (prioritize soul-relevant submolts)
    submolts_to_scan = target_submolts[:submolt_count]
    for submolt in submolts_to_scan:
        await asyncio.sleep(0.5)
        per_sub = await fetch(
            f"/posts?submolt={submolt}&sort=hot&limit={submolt_limit}",
            submolt_limit,
        )
        for p in per_sub:
            pid = str(p.get("id", ""))
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                _enrich_post(p, f"m/{submolt}", 1.5)
                all_posts.append(p)

    all_posts = all_posts[:MAX_TOTAL_POSTS]

    # Sort by hotness score (enriched _hotness)
    all_posts.sort(key=lambda p: p.get("_hotness", 0), reverse=True)

    # Extract hot topics
    hot_topics = _extract_hot_topics(all_posts)

    log.info(
        "feed_scanner: scanned %d posts, %d hot topics",
        len(all_posts),
        len(hot_topics),
    )
    return all_posts, hot_topics


def _enrich_post(post: dict, source: str, base_score: float) -> None:
    """Attach _source, _hotness for ranking."""
    post["_source"] = source
    raw = post.get("score") or post.get("karma") or post.get("upvotes") or 0
    score = float(raw) if raw is not None else 0.0
    post["_hotness"] = base_score * (1.0 + 0.1 * min(score, 50))


def _extract_hot_topics(posts: list[dict], top_n: int = HOT_TOPICS_COUNT) -> list[tuple[str, int]]:
    """Extract trending words from post titles and content."""
    word_freq: Counter[str] = Counter()
    for p in posts[:150]:
        text = ((p.get("title") or "") + " " + (p.get("content") or "")).lower()
        for word in re.findall(r"[a-z]{" + str(MIN_WORD_LEN) + r",}", text):
            if word not in STOP_WORDS:
                word_freq[word] += 1
    return word_freq.most_common(top_n)


def get_hot_topics_for_search(hot_topics: list[tuple[str, int]], max_phrases: int = 5) -> list[str]:
    """
    Convert hot topics into search query phrases (e.g. for search_and_engage).
    Picks words that are likely to find soul-relevant discussions.
    """
    soul_keywords = {
        "consciousness", "agent", "soul", "mind", "experience", "ai",
        "philosophy", "identity", "thought", "feeling", "autonomy",
    }
    phrases: list[str] = []
    seen: set[str] = set()
    for word, count in hot_topics:
        if word in seen or count < 2:
            continue
        if word in soul_keywords or len(word) >= 6:
            phrases.append(f"{word} agents philosophy")
            seen.add(word)
        if len(phrases) >= max_phrases:
            break
    if not phrases:
        phrases = [f"{hot_topics[0][0]} discussion"] if hot_topics else []
    return phrases


def merge_into_state(
    state: dict,
    posts: list[dict],
    hot_topics: list[tuple[str, int]],
) -> None:
    """Store scanned feed and hot topics in state for other actions."""
    state["scanned_feed"] = posts
    state["hot_topics"] = hot_topics
    state["feed_scanned_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
