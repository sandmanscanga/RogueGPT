#!/usr/bin/env python3
"""gptcli: A compact ChatGPT command-line client for Linux/WSL.

Features
--------
- Centralized, per-directory threads stored under ``~/.config/gptcli/threads``.
- Streaming output with colored headers via ``rich``.
- Titles and turn indices (web-app-style feel without printing the whole convo).
- Thread management: list, show, retitle, start new, continue existing.
- Search across all saved conversations (substring or regex) with context,
  plus **JSON output** for piping.
- Export a full thread to ``md`` | ``txt`` | ``json`` (stdout or file), with
  **role filter** and optional **content filter** (substring or regex).
- When ``--max-tokens`` is set, adds a **conciseness hint** so the model
  compresses instead of getting truncated; also warns if a reply was
  **truncated by the cap**.

Environment Variables
---------------------
- ``OPENAI_API_KEY``: Required. Your OpenAI API key.
- ``GPT_MODEL``: Optional. Default model name (default: ``gpt-5``).
- ``GPT_HOME``: Optional. Root config directory (default: ``~/.config/gptcli``).

Dependencies
------------
- ``openai`` SDK: ``python3 -m pip install --upgrade openai``
- ``rich`` for colored terminal output: ``python3 -m pip install rich``

Example Usage
-------------
    # Ask a question (auto-continues this directory's thread)
    gpt "List biggest files in /var and explain flags"

    # Start a fresh thread for this CWD
    gpt -n "Plan a tmux layout"

    # Browse history
    gpt threads
    gpt show --thread <THREADID> -n 10

    # Search your entire archive (JSON for piping)
    gpt search -q "tmux"
    gpt search -q "^(ffmpeg|yt)" -r --json | jq '.'

    # Export a thread (only assistant replies whose content matches 'docker')
    gpt export --thread <THREADID> --to md --role assistant --filter docker
    gpt export --thread <THREADID> --to md --filter "^error:" --regex --out ~/errs.md

This file is intentionally single-module for easy drop-in under ``~/.local/bin/gpt``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ---------- Config & Constants ----------
DEFAULT_MODEL: str = os.getenv("GPT_MODEL", "gpt-5")
GPT_HOME: Path = Path(os.getenv("GPT_HOME", str(Path.home() / ".config/gptcli")))
THREADS_DIR: Path = GPT_HOME / "threads"
REGISTRY: Path = GPT_HOME / "registry.json"  # maps cwd-hash -> thread_id
DATE_FMT: str = "%Y-%m-%d %H:%M:%S %Z"

THEME: Theme = Theme(
    {
        "title": "bold cyan",
        "meta": "dim",
        "role.user": "bold yellow",
        "role.assistant": "bold green",
        "role.system": "bold magenta",
        "hint": "italic dim",
        "error": "bold red",
        "search.hit": "bold white on dark_green",
    }
)

console: Console = Console(theme=THEME)


# ---------- Utility ----------
def ensure_dirs() -> None:
    """Ensure configuration directories and files exist."""
    THREADS_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY.exists():
        REGISTRY.write_text("{}", encoding="utf-8")


def now_local_str() -> str:
    """Return the current local time formatted for display."""
    return datetime.now(timezone.utc).astimezone().strftime(DATE_FMT)


def now_iso() -> str:
    """Return the current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def sha1(s: str) -> str:
    """Return a short (16-char) SHA1 digest for a string."""
    import hashlib as _hashlib

    return _hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def load_registry() -> Dict[str, Any]:
    """Load the CWD→thread registry."""
    ensure_dirs()
    try:
        return json.loads(REGISTRY.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}


def save_registry(reg: Dict[str, Any]) -> None:
    """Persist the CWD→thread registry."""
    ensure_dirs()
    REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def thread_path(thread_id: str) -> Path:
    """Compute the path to a thread's JSONL message log."""
    return THREADS_DIR / f"{thread_id}.jsonl"


def meta_path(thread_id: str) -> Path:
    """Compute the path to a thread's metadata file."""
    return THREADS_DIR / f"{thread_id}.meta.json"


def read_thread(thread_id: str) -> List[Dict[str, Any]]:
    """Read the full message list for a thread."""
    p = thread_path(thread_id)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def append_msg(thread_id: str, rec: Dict[str, Any]) -> None:
    """Append a single record to a thread's JSONL log."""
    p = thread_path(thread_id)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "")


def load_meta(thread_id: str) -> Dict[str, Any]:
    """Load metadata for a thread."""
    mp = meta_path(thread_id)
    if mp.exists():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_meta(thread_id: str, meta: Dict[str, Any]) -> None:
    """Persist metadata for a thread."""
    mp = meta_path(thread_id)
    mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def count_turns(msgs: List[Dict[str, Any]]) -> int:
    """Count assistant replies in a message list."""
    return sum(1 for m in msgs if m.get("role") == "assistant")


def ensure_thread_for_cwd(cwd: Optional[str], title_hint: Optional[str] = None, force_new: bool = False) -> str:
    """Ensure a thread exists (and is registered) for the given working directory."""
    ensure_dirs()
    reg = load_registry()
    key = sha1(cwd or os.getcwd())
    entry = reg.get(key)
    if entry and not force_new:
        return entry["thread_id"]

    # create new thread
    thread_id = sha1(f"{cwd}-{time.time()}")
    created = now_iso()
    reg[key] = {"thread_id": thread_id, "cwd": cwd or os.getcwd(), "created": created}
    save_registry(reg)

    meta = {"title": title_hint or "Untitled", "created": created, "cwd": reg[key]["cwd"], "turns": 0, "model": DEFAULT_MODEL}
    save_meta(thread_id, meta)
    return thread_id


# ---------- Formatting ----------
def short_title_from(prompt: str) -> str:
    """Produce a short human title from a user prompt."""
    s = prompt.strip().splitlines()[0] if prompt.strip() else "Untitled"
    s = s.rstrip(".!?")
    return (s[:57] + "...") if len(s) > 60 else s


def print_reply_header(meta: Dict[str, Any], turn_index: int) -> None:
    """Render a compact header above the answer."""
    title = meta.get("title") or "Untitled"
    hdr = Text.assemble(("[ ", "meta"), (f"{title}", "title"), (" ]  ", "meta"), (f"#{turn_index}", "meta"))
    console.print(hdr)


def print_message(role: str, content: str) -> None:
    """Pretty-print a chat message with role panel."""
    role_style = f"role.{role}" if role in ("user", "assistant", "system") else "meta"
    console.print(Panel.fit(Text(content), title=role.upper(), title_align="left", border_style=role_style))


def print_conversation(thread_id: str, max_messages: Optional[int] = None) -> None:
    """Render a thread similar to a web chat view."""
    msgs = read_thread(thread_id)
    if max_messages is not None:
        msgs = msgs[-max_messages:]
    meta = load_meta(thread_id)
    console.print(Text(meta.get("title", "Untitled"), style="title"))
    t = Table(show_header=True, header_style="meta")
    t.add_column("#", style="meta", width=4)
    t.add_column("Role", style="meta", width=10)
    t.add_column("Content")
    t.add_column("Time", style="meta", width=20)
    for idx, m in enumerate(msgs, start=1):
        t.add_row(str(idx), m.get("role", "?"), m.get("content", ""), m.get("time", ""))
    console.print(t)


# ---------- OpenAI Call ----------
def openai_chat(messages: List[Dict[str, str]], model: str, temperature: float, stream: bool, max_tokens: Optional[int]) -> Tuple[str, Optional[str]]:
    """Call OpenAI Chat Completions and return (text, finish_reason)."""
    client = OpenAI()
    if stream:
        out_chunks: List[str] = []
        finish_reason: Optional[str] = None
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        for chunk in resp:
            choice = chunk.choices[0]
            delta = choice.delta
            if delta and (txt := delta.content):
                out_chunks.append(txt)
                console.print(txt, end="", soft_wrap=True, highlight=False)
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
        print()
        return "".join(out_chunks).strip(), finish_reason
    else:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        choice = resp.choices[0]
        return choice.message.content.strip(), getattr(choice, "finish_reason", None)


# ---------- Search & Export ----------

def iter_threads() -> Iterable[str]:
    """Iterate over all thread IDs found on disk."""
    for p in THREADS_DIR.glob("*.jsonl"):
        yield p.stem


def search_archive(query: str, regex: bool = False, role: Optional[str] = None, context: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
    """Search across all threads."""
    hits: List[Dict[str, Any]] = []
    pattern: Optional[re.Pattern[str]] = re.compile(query, re.IGNORECASE) if regex else None

    for tid in iter_threads():
        meta = load_meta(tid)
        msgs = read_thread(tid)
        for idx, m in enumerate(msgs):
            if role and m.get("role") != role:
                continue
            content = m.get("content", "")
            matched = bool(pattern.search(content)) if pattern else (query.lower() in content.lower())
            if not matched:
                continue
            start = max(0, idx - context)
            end = min(len(msgs), idx + context + 1)
            window = msgs[start:end]
            hits.append({"thread_id": tid, "title": meta.get("title", "Untitled"), "index": idx + 1, "match": m, "window": window})
            if len(hits) >= limit:
                return hits
    return hits


def print_search_results(results: List[Dict[str, Any]], show_window: bool = True, as_json: bool = False) -> None:
    """Pretty-print (or JSON-print) search results."""
    if not results:
        if as_json:
            print(json.dumps([], ensure_ascii=False))
        else:
            console.print("No matches.", style="hint")
        return

    for hit in results:
        if as_json:
            out = {
                "thread_id": hit["thread_id"],
                "title": hit["title"],
                "index": hit["index"],
                "match": hit["match"],
                "window": hit["window"] if show_window else None,
            }
            print(json.dumps(out, ensure_ascii=False))
            continue
        hdr = Text.assemble((hit["title"], "title"), ("  ", "meta"), (f"({hit['thread_id']})", "meta"), ("  idx ", "meta"), (str(hit["index"]), "meta"))
        console.print(hdr)
        if show_window:
            t = Table(show_header=True, header_style="meta")
            t.add_column("#", style="meta", width=4)
            t.add_column("Role", style="meta", width=10)
            t.add_column("Content")
            t.add_column("Time", style="meta", width=20)
            for i, m in enumerate(hit["window"], start=1):
                row_style = "search.hit" if m is hit["match"] else None
                t.add_row(str(i), m.get("role", "?"), m.get("content", ""), m.get("time", ""), style=row_style)
            console.print(t)
        else:
            m = hit["match"]
            print_message(m.get("role", "?"), m.get("content", ""))


def export_thread(thread_id: str, to: str = "md", role: Optional[str] = None, content_filter: Optional[str] = None, regex: bool = False) -> str:
    """Export a thread to Markdown, plain text, or JSON (with optional filters)."""
    to = to.lower()
    meta = load_meta(thread_id)
    msgs = read_thread(thread_id)
    if role:
        msgs = [m for m in msgs if m.get("role") == role]
    if content_filter:
        if regex:
            pat = re.compile(content_filter, re.IGNORECASE)
            msgs = [m for m in msgs if pat.search(m.get("content", "") or "")]
        else:
            low = content_filter.lower()
            msgs = [m for m in msgs if low in (m.get("content", "") or "").lower()]

    if to == "json":
        payload = {"meta": meta, "messages": msgs}
        return json.dumps(payload, indent=2, ensure_ascii=False)

    title = meta.get("title", "Untitled")
    lines: List[str] = []

    if to == "md":
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"- Thread ID: `{thread_id}`")
        lines.append(f"- Created: {meta.get('created','')}")
        lines.append(f"- Linked CWD: `{meta.get('cwd','')}`")
        lines.append(f"- Model: `{meta.get('model','')}`")
        lines.append("")
        for i, m in enumerate(msgs, start=1):
            role_nm = m.get("role", "?")
            time_str = m.get("time", "")
            lines.append(f"## {i}. {role_nm.title()}  ·  {time_str}")
            lines.append("")
            lines.append(m.get("content", ""))
            lines.append("")
        return "".join(lines)

    if to == "txt":
        lines.append(f"TITLE: {title}")
        lines.append(f"THREAD: {thread_id}")
        lines.append(f"CREATED: {meta.get('created','')}")
        lines.append(f"CWD: {meta.get('cwd','')}")
        lines.append(f"MODEL: {meta.get('model','')}")
        lines.append("""
------------------------------------------------------------
""".strip(""))
        for i, m in enumerate(msgs, start=1):
            lines.append(f"[{i}] {m.get('role','?').upper()} @ {m.get('time','')}")
            lines.append(m.get("content", ""))
            lines.append("""
------------------------------------------------------------
""".strip(""))
        return "".join(lines)

    raise ValueError("export format must be one of: md, txt, json")


def write_output(data: str, out_path: Optional[str]) -> None:
    """Write exported data to a file or stdout."""
    if not out_path or out_path == "-":
        print(data)
        return
    p = Path(out_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data, encoding="utf-8")
    console.print(f"Wrote {p}", style="meta")


# ---------- CLI ----------

def read_prompt(words: Sequence[str]) -> Optional[str]:
    """Read the prompt either from arguments or stdin."""
    if words:
        return " ".join(words).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return None


def cmd_threads(_: argparse.Namespace) -> None:
    """Handle ``threads`` subcommand (list all threads)."""
    ensure_dirs()
    rows: List[Tuple[str, str, str, str, int]] = []
    for p in THREADS_DIR.glob("*.meta.json"):
        tid = p.stem.replace(".meta", "").replace(".json", "").replace(".meta", "")
        meta = load_meta(tid)
        rows.append((tid, meta.get("title", "Untitled"), meta.get("created", ""), meta.get("cwd", ""), count_turns(read_thread(tid))))
    rows.sort(key=lambda r: r[2])
    t = Table(show_header=True, header_style="meta")
    t.add_column("Thread ID", style="title", width=18)
    t.add_column("Title")
    t.add_column("Created", style="meta", width=22)
    t.add_column("Linked CWD", style="meta")
    t.add_column("Turns", style="meta", width=7)
    for r in rows:
        t.add_row(*map(lambda x: str(x), r))
    console.print(t)


def cmd_show(args: argparse.Namespace) -> None:
    """Handle ``show`` subcommand."""
    print_conversation(args.thread, max_messages=args.last)


def cmd_retitle(args: argparse.Namespace) -> None:
    """Handle ``retitle`` subcommand."""
    meta = load_meta(args.thread)
    meta["title"] = " ".join(args.title)
    save_meta(args.thread, meta)
    console.print(f"Retitled thread {args.thread} → ", style="meta", end="")
    console.print(meta["title"], style="title")


def cmd_search(args: argparse.Namespace) -> None:
    """Handle ``search`` subcommand."""
    results = search_archive(query=args.query, regex=args.regex, role=args.role, context=args.context, limit=args.limit)
    print_search_results(results, show_window=not args.brief, as_json=args.json)


def cmd_export(args: argparse.Namespace) -> None:
    """Handle ``export`` subcommand."""
    try:
        data = export_thread(args.thread, to=args.to, role=args.role, content_filter=args.filter, regex=args.regex)
    except ValueError as exc:
        console.print(str(exc), style="error")
        sys.exit(2)
    write_output(data, args.out)


def cmd_chat(args: argparse.Namespace) -> None:
    """Handle primary chat behavior (send a message, stream reply, persist)."""
    prompt = read_prompt(getattr(args, "prompt", []))
    if not prompt:
        console.print('Usage: gpt "your question"  or  echo text | gpt  [-r|-n]', style="hint")
        sys.exit(1)

    # pick thread
    if args.thread:
        thread_id = args.thread
    else:
        thread_id = ensure_thread_for_cwd(os.getcwd(), title_hint=short_title_from(prompt), force_new=args.new)

    # load and hydrate
    msgs = read_thread(thread_id)
    meta = load_meta(thread_id)

    chat_messages: List[Dict[str, str]] = []
    if args.system:
        chat_messages.append({"role": "system", "content": args.system})
    else:
        first_sys = next((m for m in msgs if m.get("role") == "system"), None)
        if first_sys:
            chat_messages.append({"role": "system", "content": first_sys.get("content", "")})

    for m in msgs:
        if m.get("role") in ("user", "assistant"):
            chat_messages.append({"role": m["role"], "content": m["content"]})

    # persist user message
    user_rec: Dict[str, Any] = {"role": "user", "content": prompt, "time": now_local_str(), "time_iso": now_iso()}
    append_msg(thread_id, user_rec)

    # If user set a max token cap, nudge the model to be concise
    if args.max_tokens is not None:
        approx_words = max(50, int(args.max_tokens * 0.75))
        concise_note = f"Be concise. Aim to fit within ~{approx_words} words. Use bullet points when helpful."
        chat_messages.append({"role": "system", "content": concise_note})

    # model call
    try:
        answer, finish_reason = openai_chat(
            messages=chat_messages + [{"role": "user", "content": prompt}],
            model=args.model,
            temperature=args.temperature,
            stream=(not args.no_stream),
            max_tokens=args.max_tokens,
        )
    except Exception as exc:  # surfacing SDK/storage issues
        console.print(f"[error] {exc}", style="error")
        sys.exit(2)

    asst_rec: Dict[str, Any] = {"role": "assistant", "content": answer, "time": now_local_str(), "time_iso": now_iso()}
    append_msg(thread_id, asst_rec)

    # update meta
    if (meta.get("title") in (None, "", "Untitled")) and prompt:
        meta["title"] = short_title_from(prompt)
    meta["model"] = args.model
    all_msgs = read_thread(thread_id)
    turn_index = count_turns(all_msgs)
    meta["turns"] = turn_index
    save_meta(thread_id, meta)

    # print header & answer
    print_reply_header(meta, turn_index)
    if args.no_stream:
        print_message("assistant", answer)
    if finish_reason == "length":
        console.print("[warning] Reply hit --max-tokens limit and was truncated.", style="error")


# ---------- Entrypoint ----------

def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="gpt",
        description=(
            "ChatGPT CLI with centralized history, per-dir threads, titles, "
            "indices, colors, search, and export."
        ),
    )
    sub = parser.add_subparsers(dest="cmd")

    # chat (default)
    chat_p = sub.add_parser("chat", help="Send a message (default).")
    chat_p.add_argument("prompt", nargs="*", help="Message; if omitted, read from stdin.")
    chat_p.add_argument("-r", "--reply", action="store_true", help="Use existing thread for this CWD (default).")
    chat_p.add_argument("-n", "--new", action="store_true", help="Create a new thread for this CWD.")
    chat_p.add_argument("-s", "--system", default=None, help="System prompt.")
    chat_p.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    chat_p.add_argument("-t", "--temperature", type=float, default=0.2)
    chat_p.add_argument("--no-stream", action="store_true", help="Disable streaming output.")
    chat_p.add_argument("--max-tokens", type=int, default=None)
    chat_p.add_argument("--thread", default=None, help="Explicit thread id (overrides CWD binding).")

    # threads: list
    sub.add_parser("threads", help="List all threads.")

    # show: view a conversation
    show_p = sub.add_parser("show", help="Show a thread (like the web app history).")
    show_p.add_argument("--thread", required=True, help="Thread id to show.")
    show_p.add_argument("-n", "--last", type=int, default=None, help="Show only the last N messages.")

    # retitle
    tit_p = sub.add_parser("retitle", help="Set a thread title.")
    tit_p.add_argument("--thread", required=True, help="Thread id.")
    tit_p.add_argument("title", nargs="+", help="New title.")

    # search
    sch = sub.add_parser("search", help="Search your entire archive.")
    sch.add_argument("-q", "--query", required=True, help="Substring or regex pattern to find.")
    sch.add_argument("-r", "--regex", action="store_true", help="Treat query as a regex.")
    sch.add_argument("--role", choices=["user", "assistant", "system"], default=None, help="Filter by role.")
    sch.add_argument("-c", "--context", type=int, default=1, help="Neighboring messages to include.")
    sch.add_argument("-l", "--limit", type=int, default=50, help="Max number of hits to show.")
    sch.add_argument("-b", "--brief", action="store_true", help="Only print the matched message.")
    sch.add_argument("--json", action="store_true", help="Emit machine-readable JSON hits (one object per line).")

    # export
    exp = sub.add_parser("export", help="Export a thread to md|txt|json (stdout or file).")
    exp.add_argument("--thread", required=True, help="Thread id to export.")
    exp.add_argument("--to", choices=["md", "txt", "json"], default="md", help="Export format.")
    exp.add_argument("--role", choices=["user", "assistant", "system"], default=None, help="Only include this role in export.")
    exp.add_argument("--filter", help="Only include messages whose content matches this pattern.")
    exp.add_argument("--regex", action="store_true", help="Treat --filter as regex (case-insensitive).")
    exp.add_argument("--out", default="-", help="Output path or '-' for stdout.")

    # allow `gpt "hello"` without the word 'chat'
    parser.add_argument("prompt_passthrough", nargs="*", help=argparse.SUPPRESS)

    return parser


def main() -> None:
    """Program entrypoint."""
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()

    # If called like: gpt "hello", treat as chat (reply mode)
    if args.cmd is None and args.prompt_passthrough:
        args.cmd = "chat"
        args.prompt = args.prompt_passthrough
        args.reply = True
        args.new = False
        args.system = None
        args.model = DEFAULT_MODEL
        args.temperature = 0.2
        args.no_stream = False
        args.max_tokens = None
        args.thread = None

    if args.cmd is None:
        parser.print_help()
        sys.exit(0)

    if args.cmd == "threads":
        cmd_threads(args)
        return
    if args.cmd == "show":
        cmd_show(args)
        return
    if args.cmd == "retitle":
        cmd_retitle(args)
        return
    if args.cmd == "search":
        cmd_search(args)
        return
    if args.cmd == "export":
        cmd_export(args)
        return
    if args.cmd == "chat":
        cmd_chat(args)
        return

    console.print("Unknown command.", style="error")
    sys.exit(2)


if __name__ == "__main__":
    main()
