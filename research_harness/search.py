from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from .schemas import Source


TOKEN_RE = re.compile(r"[a-zA-Z0-9_+-]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class SearchBackend(Protocol):
    tool_name: str

    def search(self, query: str, limit: int = 4) -> list[tuple["CorpusDocument", float]]: ...

    def to_source(self, document: "CorpusDocument", relevance_score: float) -> Source: ...


@dataclass
class CorpusDocument:
    url: str
    title: str
    author: str
    date: str
    source_type: str
    summary: str
    claims: list[str]
    tags: list[str]
    credibility_score: float


class LocalCorpusSearch:
    """Deterministic search over a small JSON corpus.

    This keeps the MVP reproducible and usable without network credentials. A
    web-backed retriever can be added behind the same interface later.
    """

    tool_name = "local_corpus_search"

    def __init__(self, corpus_path: Path):
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.documents = [CorpusDocument(**row) for row in payload]

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        query_terms = _tokens(query)
        scored: list[tuple[CorpusDocument, float]] = []
        for document in self.documents:
            haystack = " ".join(
                [
                    document.title,
                    document.summary,
                    " ".join(document.claims),
                    " ".join(document.tags),
                ]
            )
            doc_terms = _tokens(haystack)
            overlap = len(query_terms & doc_terms)
            tag_overlap = len(query_terms & set(document.tags))
            if overlap == 0:
                continue
            score = min(1.0, (overlap + tag_overlap * 1.5) / math.sqrt(max(len(query_terms), 1) * 12))
            scored.append((document, round(score, 3)))
        scored.sort(key=lambda pair: (pair[1], pair[0].credibility_score), reverse=True)
        return scored[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return Source(
            url=document.url,
            title=document.title,
            author=document.author,
            date=document.date,
            source_type=document.source_type,
            summary=document.summary,
            relevance_score=relevance_score,
            credibility_score=document.credibility_score,
        )


class ArxivSearch:
    """Small stdlib arXiv retriever for real literature-oriented runs."""

    tool_name = "arxiv_api_search"

    def __init__(self, base_url: str = "https://export.arxiv.org/api/query", timeout_seconds: float = 20.0):
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        cleaned_query = _arxiv_query(query)
        if not cleaned_query:
            return []
        params = urllib.parse.urlencode(
            {
                "search_query": cleaned_query,
                "start": 0,
                "max_results": limit,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        request = urllib.request.Request(
            f"{self.base_url}?{params}",
            headers={"User-Agent": "research-harness/0.1.0 (mailto:research-harness@example.invalid)"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read()
        documents = _parse_arxiv_feed(payload)
        scored = []
        query_terms = _tokens(query)
        for document in documents:
            haystack = f"{document.title} {document.summary} {' '.join(document.tags)}"
            overlap = len(query_terms & _tokens(haystack))
            score = min(1.0, 0.55 + (overlap / max(len(query_terms), 1)) * 0.35)
            scored.append((document, round(score, 3)))
        return scored[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return Source(
            url=document.url,
            title=document.title,
            author=document.author,
            date=document.date,
            source_type=document.source_type,
            summary=document.summary,
            relevance_score=relevance_score,
            credibility_score=document.credibility_score,
        )


class OpenAlexSearch:
    tool_name = "openalex_api_search"

    def __init__(self, base_url: str = "https://api.openalex.org/works", timeout_seconds: float = 20.0):
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        params = urllib.parse.urlencode(
            {
                "search": query,
                "per-page": limit,
                "sort": "cited_by_count:desc",
            }
        )
        payload = _http_json(f"{self.base_url}?{params}", self.timeout_seconds)
        results = payload.get("results", [])
        documents = []
        for item in results:
            title = _clean_text(str(item.get("title") or "Untitled OpenAlex work"))
            abstract = _openalex_abstract(item.get("abstract_inverted_index") or {})
            year = str(item.get("publication_year") or "")
            url = item.get("primary_location", {}).get("landing_page_url") or item.get("doi") or item.get("id") or ""
            authors = [
                author.get("author", {}).get("display_name", "")
                for author in item.get("authorships", [])
                if author.get("author", {}).get("display_name")
            ]
            concepts = [concept.get("display_name", "") for concept in item.get("concepts", [])[:6]]
            documents.append(
                CorpusDocument(
                    url=str(url),
                    title=title,
                    author=", ".join(authors[:4]) or "Unknown",
                    date=year,
                    source_type="openalex_work",
                    summary=abstract or title,
                    claims=_summary_claims(title, abstract or title),
                    tags=[concept for concept in concepts if concept],
                    credibility_score=0.76,
                )
            )
        return _score_documents(query, documents)[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return _source_from_document(document, relevance_score)


class GitHubSearch:
    tool_name = "github_repo_search"

    def __init__(self, base_url: str = "https://api.github.com/search/repositories", timeout_seconds: float = 20.0):
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        search_query = f"{query} in:name,description,readme"
        params = urllib.parse.urlencode({"q": search_query, "sort": "stars", "order": "desc", "per_page": limit})
        headers = {}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = _http_json(f"{self.base_url}?{params}", self.timeout_seconds, headers=headers)
        documents = []
        for item in payload.get("items", []):
            title = str(item.get("full_name") or item.get("name") or "GitHub repository")
            description = _clean_text(str(item.get("description") or "No repository description available."))
            stars = int(item.get("stargazers_count") or 0)
            updated_at = str(item.get("updated_at") or "")[:10]
            language = item.get("language") or ""
            documents.append(
                CorpusDocument(
                    url=str(item.get("html_url") or ""),
                    title=title,
                    author=str(item.get("owner", {}).get("login") or "Unknown"),
                    date=updated_at,
                    source_type="github_repository",
                    summary=f"{description} Stars: {stars}. Primary language: {language}.",
                    claims=[
                        f"{title}: Repository description says {description}",
                        f"{title}: Repository has {stars} stars, suggesting community interest or adoption.",
                    ],
                    tags=[tag for tag in [language, "github", "repository"] if tag],
                    credibility_score=0.64,
                )
            )
        return _score_documents(query, documents)[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return _source_from_document(document, relevance_score)


class WebSearch:
    tool_name = "web_search"

    def __init__(self, source_type: str = "web_result", timeout_seconds: float = 20.0):
        self.source_type = source_type
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        request = urllib.request.Request(url, headers={"User-Agent": "research-harness/0.1.0"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            html_payload = response.read().decode("utf-8", errors="replace")
        documents = _parse_duckduckgo_html(html_payload, self.source_type)
        return _score_documents(query, documents)[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return _source_from_document(document, relevance_score)


class DocsBlogsSearch(WebSearch):
    tool_name = "docs_blogs_search"

    def __init__(self):
        super().__init__(source_type="docs_blog")

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        return super().search(f"{query} docs blog technical writeup", limit)


class SocialWebSearch(WebSearch):
    tool_name = "social_web_search"

    def __init__(self):
        super().__init__(source_type="social_web")

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        return super().search(f"{query} site:x.com OR site:twitter.com", limit)


class PriorArtifactMemorySearch:
    tool_name = "prior_artifact_memory_search"

    def __init__(self, output_root: Path):
        self.output_root = output_root

    def search(self, query: str, limit: int = 4) -> list[tuple[CorpusDocument, float]]:
        documents = []
        for run_dir in sorted(self.output_root.glob("run_*")):
            if not run_dir.is_dir():
                continue
            run_record = _first_json_row(run_dir / "runs.json")
            claims = _read_json(run_dir / "claims.json", [])
            sources = _read_json(run_dir / "sources.json", [])
            source_titles = [str(source.get("title", "")) for source in sources[:5]]
            summary = " ".join(str(claim.get("text", "")) for claim in claims[:5]) or "Prior run artifact."
            goal = str(run_record.get("user_goal", run_dir.name))
            documents.append(
                CorpusDocument(
                    url=str(run_dir),
                    title=f"Prior run: {goal[:90]}",
                    author="research-harness",
                    date=str(run_record.get("completed_at", ""))[:10],
                    source_type="prior_artifact_memory",
                    summary=f"{summary} Sources included: {', '.join(source_titles)}",
                    claims=[str(claim.get("text", "")) for claim in claims[:3] if claim.get("text")],
                    tags=["prior-run", "artifact-memory"],
                    credibility_score=0.58,
                )
            )
        return _score_documents(query, documents)[:limit]

    def to_source(self, document: CorpusDocument, relevance_score: float) -> Source:
        return _source_from_document(document, relevance_score)


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


def _source_from_document(document: CorpusDocument, relevance_score: float) -> Source:
    return Source(
        url=document.url,
        title=document.title,
        author=document.author,
        date=document.date,
        source_type=document.source_type,
        summary=document.summary,
        relevance_score=relevance_score,
        credibility_score=document.credibility_score,
    )


def _score_documents(query: str, documents: list[CorpusDocument]) -> list[tuple[CorpusDocument, float]]:
    query_terms = _tokens(query)
    scored = []
    for document in documents:
        haystack = f"{document.title} {document.summary} {' '.join(document.claims)} {' '.join(document.tags)}"
        overlap = len(query_terms & _tokens(haystack))
        score = min(1.0, 0.45 + (overlap / max(len(query_terms), 1)) * 0.45)
        scored.append((document, round(score, 3)))
    scored.sort(key=lambda pair: (pair[1], pair[0].credibility_score), reverse=True)
    return scored


def _http_json(url: str, timeout_seconds: float, headers: Optional[dict[str, str]] = None) -> dict:
    request_headers = {"User-Agent": "research-harness/0.1.0"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_duckduckgo_html(payload: str, source_type: str) -> list[CorpusDocument]:
    pattern = re.compile(
        r'<a rel="nofollow" class="result__a" href="(?P<href>.*?)".*?>(?P<title>.*?)</a>.*?'
        r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
        re.S,
    )
    documents = []
    for match in pattern.finditer(payload):
        raw_url = html_unescape(match.group("href"))
        parsed = urllib.parse.urlparse(raw_url)
        query = urllib.parse.parse_qs(parsed.query)
        url = query.get("uddg", [raw_url])[0]
        title = _strip_html(html_unescape(match.group("title")))
        snippet = _strip_html(html_unescape(match.group("snippet")))
        documents.append(
            CorpusDocument(
                url=url,
                title=title,
                author=urllib.parse.urlparse(url).netloc or "web",
                date="",
                source_type=source_type,
                summary=snippet,
                claims=_summary_claims(title, snippet),
                tags=[source_type, urllib.parse.urlparse(url).netloc],
                credibility_score=0.52,
            )
        )
    return documents


def html_unescape(value: str) -> str:
    return (
        value.replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#x27;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )


def _strip_html(value: str) -> str:
    return _clean_text(re.sub(r"<.*?>", "", value))


def _openalex_abstract(inverted_index: dict[str, list[int]]) -> str:
    if not inverted_index:
        return ""
    words_by_position: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for position in positions:
            words_by_position[int(position)] = word
    return " ".join(words_by_position[index] for index in sorted(words_by_position))


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _first_json_row(path: Path) -> dict:
    rows = _read_json(path, [])
    return rows[0] if rows else {}


def _arxiv_query(text: str) -> str:
    aliases_removed = text.lower().replace("arxive", "arxiv")
    stopwords = {
        "please",
        "research",
        "find",
        "mentions",
        "mention",
        "studying",
        "study",
        "new",
        "on",
        "and",
        "the",
        "which",
        "ones",
        "will",
        "be",
        "used",
        "in",
        "based",
        "current",
        "arxiv",
        "determine",
        "years",
        "year",
        "workplace",
        "trends",
        "should",
        "we",
        "deeply",
        "understand",
        "understanding",
        "before",
        "attempting",
        "attempt",
        "replicate",
        "replicating",
        "foundational",
        "literature",
        "mechanisms",
        "recent",
        "empirical",
        "evidence",
        "contradictory",
        "limitations",
    }
    terms = [term for term in _tokens(aliases_removed) if term not in stopwords and not term.isdigit()]
    if not terms:
        return ""
    selected = terms[:6]
    operator = " OR " if len(selected) > 3 else " AND "
    return operator.join(f"all:{urllib.parse.quote(term)}" for term in selected)


def _parse_arxiv_feed(payload: bytes) -> list[CorpusDocument]:
    root = ET.fromstring(payload)
    documents: list[CorpusDocument] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = _clean_text(_child_text(entry, "title"))
        summary = _clean_text(_child_text(entry, "summary"))
        published = _child_text(entry, "published")[:10]
        url = _child_text(entry, "id")
        authors = [
            _clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        categories = [
            category.attrib.get("term", "")
            for category in entry.findall("atom:category", ATOM_NS)
            if category.attrib.get("term")
        ]
        documents.append(
            CorpusDocument(
                url=url,
                title=title,
                author=", ".join(authors[:4]) or "Unknown",
                date=published,
                source_type="arxiv_paper",
                summary=summary,
                claims=_summary_claims(title, summary),
                tags=categories,
                credibility_score=0.72,
            )
        )
    return documents


def _child_text(entry: ET.Element, tag: str) -> str:
    return entry.findtext(f"atom:{tag}", default="", namespaces=ATOM_NS)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _summary_claims(title: str, summary: str) -> list[str]:
    sentences = [_clean_text(sentence) for sentence in SENTENCE_RE.split(summary) if sentence.strip()]
    claims = sentences[:3]
    if not claims and summary:
        claims = [summary]
    return [f"{title}: {claim}" for claim in claims]
