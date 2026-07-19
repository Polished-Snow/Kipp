#!/usr/bin/env python3
"""Render Kipp's canonical Markdown into the static GitHub Pages site."""

from __future__ import annotations

import argparse
import html
import os
import pathlib
import re
import string
import sys
from dataclasses import dataclass

from markdown_it import MarkdownIt


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
TEMPLATE = ROOT / "tools" / "docs" / "page.html"
GITHUB = "https://github.com/Polished-Snow/Kipp"


@dataclass(frozen=True)
class Page:
    source: pathlib.Path
    output: pathlib.Path
    title: str
    description: str


PAGES = (
    Page(
        ROOT / "README.md",
        DOCS / "index.html",
        "Kipp — hand-written inference engine",
        "A small, native inference engine for pinned Qwen3 dense checkpoints.",
    ),
    Page(
        DOCS / "ARCHITECTURE.md",
        DOCS / "architecture.html",
        "Architecture — Kipp",
        "Model, memory, backend, serving, and correctness contracts.",
    ),
    Page(
        DOCS / "ROADMAP.md",
        DOCS / "roadmap.html",
        "Roadmap — Kipp",
        "Correctness-gated implementation phases and delivered extensions.",
    ),
    Page(
        DOCS / "MODEL-SUPPORT.md",
        DOCS / "model-support.html",
        "Model support — Kipp",
        "Pinned checkpoint, backend, and quantization support status.",
    ),
    Page(
        DOCS / "BENCHMARKS.md",
        DOCS / "benchmarks.html",
        "Benchmarks — Kipp",
        "Reproducible benchmark policy and measured results.",
    ),
    Page(
        ROOT / "AGENT.md",
        DOCS / "contributing.html",
        "Contributing — Kipp",
        "Engineering constraints for human and AI contributors.",
    ),
    Page(
        DOCS / "research" / "inspiration-notes.md",
        DOCS / "research" / "inspiration-notes.html",
        "Inspiration notes — Kipp",
        "Reference repositories and provenance rules.",
    ),
    Page(
        DOCS / "research" / "phase5-notes.md",
        DOCS / "research" / "phase5-notes.html",
        "Batching and KV research — Kipp",
        "Delivered batching groundwork and future KV block design notes.",
    ),
)

OUTPUT_BY_SOURCE = {page.source.resolve(): page.output.resolve() for page in PAGES}
MARKDOWN_LINK = re.compile(r"(\[[^\]]+\]\()([^) \t]+)([^)]*\))")
HTML_LINK = re.compile(r'(?P<prefix>\b(?:href|src)=")(?P<target>[^"]+)(?P<suffix>")')


def relative_output(target: pathlib.Path, output: pathlib.Path) -> str:
    return pathlib.PurePosixPath(
        os.path.relpath(target, start=output.parent)
    ).as_posix()


def rewrite_target(target: str, source: pathlib.Path, output: pathlib.Path) -> str:
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return target
    path_text, separator, fragment = target.partition("#")
    candidate = (source.parent / path_text).resolve()
    rendered = OUTPUT_BY_SOURCE.get(candidate)
    if rendered is not None:
        destination = relative_output(rendered, output)
    elif candidate == DOCS.resolve() or DOCS.resolve() in candidate.parents:
        destination = relative_output(candidate, output)
    elif candidate == ROOT.resolve() or ROOT.resolve() in candidate.parents:
        try:
            repository_path = candidate.relative_to(ROOT).as_posix()
        except ValueError:
            return target
        kind = "tree" if candidate.is_dir() else "blob"
        destination = f"{GITHUB}/{kind}/main/{repository_path}"
    else:
        return target
    return destination + (separator + fragment if separator else "")


def rewrite_links(markdown: str, source: pathlib.Path, output: pathlib.Path) -> str:
    def markdown_replacement(match: re.Match[str]) -> str:
        target = rewrite_target(match.group(2), source, output)
        return f"{match.group(1)}{target}{match.group(3)}"

    def html_replacement(match: re.Match[str]) -> str:
        target = rewrite_target(match.group("target"), source, output)
        return f'{match.group("prefix")}{target}{match.group("suffix")}'

    markdown = MARKDOWN_LINK.sub(markdown_replacement, markdown)
    return HTML_LINK.sub(html_replacement, markdown)


def navigation(page: Page) -> str:
    prefix = relative_output(DOCS / "index.html", page.output)
    prefix = prefix[: -len("index.html")]
    links = (
        ("index.html", "Home"),
        ("architecture.html", "Architecture"),
        ("roadmap.html", "Roadmap"),
        ("model-support.html", "Models"),
        ("benchmarks.html", "Benchmarks"),
        ("contributing.html", "Contributing"),
    )
    items = []
    for href, label in links:
        current = ' aria-current="page"' if page.output.name == href else ""
        items.append(f'<a href="{prefix}{href}"{current}>{label}</a>')
    return "\n        ".join(items)


def render(page: Page, template: string.Template, markdown: MarkdownIt) -> str:
    source = page.source.read_text(encoding="utf-8")
    source = rewrite_links(source, page.source, page.output)
    body = markdown.render(source).rstrip()
    nav_prefix = relative_output(DOCS / "index.html", page.output)
    nav_prefix = nav_prefix[: -len("index.html")]
    rendered = template.substitute(
        title=html.escape(page.title, quote=True),
        description=html.escape(page.description, quote=True),
        asset_prefix=nav_prefix,
        nav_prefix=nav_prefix,
        navigation=navigation(page),
        body=body,
    )
    return rendered.rstrip() + "\n"


def stale_pages() -> list[tuple[Page, str]]:
    template = string.Template(TEMPLATE.read_text(encoding="utf-8"))
    markdown = MarkdownIt("commonmark", {"html": True}).enable(
        ("table", "strikethrough")
    )
    stale = []
    for page in PAGES:
        generated = render(page, template, markdown)
        current = (
            page.output.read_text(encoding="utf-8")
            if page.output.exists()
            else None
        )
        if current != generated:
            stale.append((page, generated))
    return stale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed HTML differs from the Markdown sources",
    )
    args = parser.parse_args()
    stale = stale_pages()
    if args.check:
        for page, _generated in stale:
            print(f"stale generated page: {page.output.relative_to(ROOT)}")
        return 1 if stale else 0
    for page, generated in stale:
        page.output.parent.mkdir(parents=True, exist_ok=True)
        page.output.write_text(generated, encoding="utf-8")
        print(f"rendered {page.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        print(f"render_docs.py: {error}", file=sys.stderr)
        raise SystemExit(1) from error
