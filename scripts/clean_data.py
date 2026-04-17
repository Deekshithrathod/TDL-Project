"""
clean_data.py
=============
WhatsApp chat export cleaning pipeline.

Parses raw exports, filters noise, anonymises senders, normalises text, and
writes one clean sentence per line to the output file.

Supports two export formats:
  [YYYY-MM-DD HH:MM] Sender: message          (primary, ~99% of corpus)
  DD/MM/YY, HH:MM AM/PM - Sender: message     (legacy WhatsApp format)

Multi-line messages (continuation lines without a timestamp) are joined to
their parent message before processing.

Usage:
    python scripts/clean_data.py --input data/whatsapp_chats/WhatsApp_Chat_1.txt \\
                                  --output data/processed/cleaned_data.txt
    # or process the whole corpus directory:
    python scripts/clean_data.py --input data/whatsapp_chats/ \\
                                  --output data/processed/cleaned_data.txt
"""

import argparse
import os
import re
import sys
from pathlib import Path

import emoji

# ---------------------------------------------------------------------------
# Timestamp regexes — both formats
# ---------------------------------------------------------------------------

# [YYYY-MM-DD HH:MM] Sender: body
_FMT_ISO = re.compile(
    r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] ([^:]+): (.*)"
)

# DD/MM/YY, HH:MM AM/PM - Sender: body  (and minor variants)
_FMT_LEGACY = re.compile(
    r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?"
    r"\s*(?:AM|PM)?\s*[-\u2013]\s*([^:]+): (.*)",
    re.IGNORECASE,
)

# Phone number — Indian mobile (with or without +91 prefix)
_PHONE_RE = re.compile(r"(\+91[\s\-]?)?(?<!\d)[6-9]\d{9}(?!\d)")

# URL-only message
_URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)

# Punctuation removal baseline — keep apostrophes and hyphens
_PUNCT_RE = re.compile(r"[^\w\s'\-]")
_STANDALONE_HYPHEN_RE = re.compile(r"(?<!\w)-(?!\w)")
_MULTI_SPACE_RE = re.compile(r" {2,}")

# System / noise messages to drop (matched against lowercased body)
_SYSTEM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"<media omitted>",
        r"messages and calls are end-to-end encrypted",
        r"this message was deleted",
        r"you deleted this message",
        r"missed voice call",
        r"missed video call",
    ]
]

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _open_file(path: str):
    """Try utf-8-sig first; fall back to latin-1 on decode error."""
    try:
        f = open(path, encoding="utf-8-sig")
        f.read()
        f.seek(0)
        return f, "utf-8-sig"
    except (UnicodeDecodeError, ValueError):
        pass
    try:
        f = open(path, encoding="latin-1")
        print(f"  WARNING: {path} — utf-8 failed, using latin-1 fallback", file=sys.stderr)
        return f, "latin-1"
    except Exception as e:
        raise RuntimeError(f"Cannot open {path}: {e}")


def collect_files(input_path: str) -> list[str]:
    p = Path(input_path)
    if p.is_dir():
        return sorted(str(f) for f in p.iterdir() if f.suffix == ".txt")
    return [str(p)]


# ---------------------------------------------------------------------------
# Parsing — returns list of (sender, body) for the whole file
# ---------------------------------------------------------------------------

def parse_file(path: str) -> list[tuple[str, str]]:
    """
    Parse one file into (sender, full_body) pairs.
    Multi-line continuation lines are appended to the preceding message.
    Lines that match neither timestamp format are attached to the current
    message if one exists, otherwise discarded.
    """
    with _open_file(path)[0] as f:
        raw_lines = [line.rstrip("\n") for line in f]

    messages: list[tuple[str, str]] = []
    current_sender: str | None = None
    current_body_parts: list[str] = []

    def flush():
        if current_sender is not None and current_body_parts:
            messages.append((current_sender, " ".join(current_body_parts)))

    for line in raw_lines:
        if not line.strip():
            continue

        m = _FMT_ISO.match(line) or _FMT_LEGACY.match(line)
        if m:
            flush()
            current_sender = m.group(1).strip()
            current_body_parts = [m.group(2).strip()]
        else:
            # Continuation of the previous message
            if current_sender is not None:
                current_body_parts.append(line.strip())
            # else: unparseable line before any message — discard

    flush()
    return messages


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _is_emoji_only(text: str) -> bool:
    """Return True if stripping all emoji leaves nothing meaningful."""
    stripped = emoji.replace_emoji(text, replace="").strip()
    return stripped == ""


def _is_url_only(text: str) -> bool:
    return bool(_URL_RE.match(text.strip()))


def _is_system_message(text: str) -> bool:
    return any(p.search(text) for p in _SYSTEM_PATTERNS)


def filter_message(body: str) -> tuple[bool, str]:
    """
    Returns (keep, reason).  reason is non-empty when keep=False.
    """
    if _is_system_message(body):
        return False, "system"
    if _is_url_only(body):
        return False, "url"
    if _is_emoji_only(body):
        return False, "emoji_only"
    # Word count check happens after normalisation in the main loop
    return True, ""


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise(body: str) -> str:
    # Redact phone numbers
    body = _PHONE_RE.sub("[PHONE]", body)
    # Lowercase
    body = body.lower()
    # Remove punctuation — keep apostrophes and intra-word hyphens
    body = _PUNCT_RE.sub(" ", body)
    body = _STANDALONE_HYPHEN_RE.sub(" ", body)
    # Collapse spaces
    body = _MULTI_SPACE_RE.sub(" ", body)
    return body.strip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_path: str, output_path: str, report_path: str):
    files = collect_files(input_path)
    if not files:
        sys.exit(f"No .txt files found at {input_path}")

    print(f"Processing {len(files)} file(s)...")

    # --- Pass 1: parse all files, collect sender names ---
    all_messages: list[tuple[str, str]] = []
    for fp in files:
        all_messages.extend(parse_file(fp))

    total_parsed = len(all_messages)

    # Build sender → USER_X mapping in order of first appearance
    seen_senders: dict[str, str] = {}
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    def sender_label(name: str) -> str:
        if name not in seen_senders:
            idx = len(seen_senders)
            if idx < 26:
                tag = f"USER_{labels[idx]}"
            else:
                tag = f"USER_{idx}"
            seen_senders[name] = tag
        return seen_senders[name]

    # Touch senders in order so mapping is stable
    for sender, _ in all_messages:
        sender_label(sender)

    # --- Pass 2: filter + normalise ---
    kept: list[str] = []

    counts = {
        "system":     0,
        "url":        0,
        "emoji_only": 0,
        "too_short":  0,
    }

    example_before: str | None = None
    example_after:  str | None = None

    for sender, body in all_messages:
        keep, reason = filter_message(body)
        if not keep:
            if reason == "system":
                counts["system"] += 1
            elif reason == "url":
                counts["url"] += 1
            elif reason == "emoji_only":
                counts["emoji_only"] += 1
            continue

        normalised = normalise(body)
        if len(normalised.split()) < 3:
            counts["too_short"] += 1
            continue

        kept.append(normalised)
        if example_before is None:
            example_before = body
            example_after = normalised

    # --- Write output ---
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")

    # --- Stats ---
    total_filtered = total_parsed - len(kept)
    pct_kept = 100 * len(kept) / total_parsed if total_parsed else 0
    total_noise = counts["system"] + counts["url"] + counts["emoji_only"]

    sender_list = " ... ".join(list(seen_senders.values())[:5])
    if len(seen_senders) > 5:
        sender_list += f" ... (+ {len(seen_senders) - 5} more)"

    report_lines = [
        f"Total messages parsed:     {total_parsed}",
        f"After filtering:           {len(kept)}  ({pct_kept:.1f}% kept)",
        f"Breakdown of filtered:",
        f"  - Media/system:          {counts['system']}",
        f"  - Too short (<3 words):  {counts['too_short']}",
        f"  - URL-only:              {counts['url']}",
        f"  - Emoji-only:            {counts['emoji_only']}",
        f"Unique senders found:      {len(seen_senders)}  → {sender_list}",
        f"",
        f"Example (before): {example_before}",
        f"Example (after):  {example_after}",
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    rpt_path = Path(report_path)
    rpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(f"\nOutput  → {out_path}")
    print(f"Report  → {rpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clean a WhatsApp export or directory of exports."
    )
    parser.add_argument(
        "--input",  required=True,
        help="Path to a single .txt export or a directory of exports",
    )
    parser.add_argument(
        "--output", default="data/processed/cleaned_data.txt",
        help="Output path for cleaned text (default: data/processed/cleaned_data.txt)",
    )
    parser.add_argument(
        "--report", default="report/cleaning_report.txt",
        help="Path to write the stats report (default: report/cleaning_report.txt)",
    )
    args = parser.parse_args()
    run(args.input, args.output, args.report)


if __name__ == "__main__":
    main()
