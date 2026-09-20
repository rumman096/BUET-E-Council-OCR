import time
import html
import hashlib
import io
import os
import json
import math
import random
import re
import threading
import unicodedata
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path
from datetime import datetime, timezone
import copy
import tempfile
import uuid
import zipfile
import base64
from contextlib import contextmanager

import streamlit as st
import streamlit.components.v1 as components

# Older Streamlit versions require this before accessing secrets or widgets.
st.set_page_config(
    page_title="BUET E-Council Document Processor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",
)

try:
    import fitz  # PyMuPDF
except ImportError:
    st.error("PyMuPDF missing. Run: pip install pymupdf")
    st.stop()

# Pillow ships with Streamlit; used for grayscale + contrast preprocessing of
# rendered page images. If unavailable, images are still produced via PyMuPDF.
try:
    from PIL import Image, ImageEnhance, ImageOps

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# --- Gemini SDK: prefer the new google-genai, fall back to legacy ---
NEW_SDK = False
try:
    from google import genai as genai_new
    from google.genai import types as genai_types

    NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai_old
    except ImportError:
        st.error("Gemini SDK missing. Run: pip install google-genai")
        st.stop()

# ==========================================
# GEMINI API KEYS — ADMINISTRATOR CONFIGURATION
# ==========================================
# In Streamlit Community Cloud, store the keys in App settings → Secrets:
#
# GEMINI_API_KEYS = ["KEY_1", "KEY_2", "KEY_3"]
#
# For local/Docker use, an optional comma-separated GEMINI_API_KEYS environment
# variable is also supported. Real keys are never stored in this source file.
try:
    # Some older releases render a red error even when FileNotFoundError is
    # caught. Their optional loader avoids that for keyless/local-only use.
    _secrets_loader = getattr(type(st.secrets), "load_if_toml_exists", None)
    _secrets_available = _secrets_loader(st.secrets) if _secrets_loader else True
    _secret_keys = st.secrets.get("GEMINI_API_KEYS", []) if _secrets_available else []
    _secret_keys = _secret_keys.split(",") if isinstance(_secret_keys, str) else list(_secret_keys)
except Exception:
    _secret_keys = []

_env_keys = os.getenv("GEMINI_API_KEYS", "").split(",")
GEMINI_API_KEYS = [
    str(key).strip()
    for key in [*_secret_keys, *_env_keys]
    if str(key).strip()
]
MODEL_NAME = "gemini-3.6-flash"  # model used for OCR and JSON extraction
CHUNK_SIZE = 4  # short visual context: long chunks make the model copy
# names and affiliations between neighbouring entries. 4 pages renders to
# ~3.8 MB, so it is also never force-split by the inline payload limit.
JSON_CHUNK_CHARS = 70_000  # fewer JSON requests; adjustable from the sidebar
DEFAULT_MAX_WORKERS = 4  # concurrency only; the limiter still controls request starts
DEFAULT_SAFE_RPM = 15
MAX_INLINE_MB = 19  # inline request payload safety limit (~20 MB hard cap)
OCR_THINKING_LEVEL = "minimal"
JSON_THINKING_LEVEL = "low"
# Models that REJECT thinking_level="minimal" with an API validation error.
# gemini-3.7-flash only accepts low / medium / high, so asking for "minimal"
# fails every request with INVALID_ARGUMENT — which this app treats as fatal,
# so every page reports "could not be read". Fall back to the nearest level.
NO_MINIMAL_THINKING = ("3.7-flash", "3.7-pro", "3.8-flash", "3-pro", "3.1-pro", "2.5-")


def resolve_thinking_level(model_name: str, level: str) -> str:
    """Downgrade 'minimal' to 'low' on models that do not support it."""
    name = (model_name or "").lower()
    if level == "minimal" and any(tag in name for tag in NO_MINIMAL_THINKING):
        return "low"
    return level
OCR_ATTEMPTS = 5  # retries per request for transient errors AND incomplete output
JSON_ATTEMPTS = 5
OCR_IMAGE_DPI = 300  # render resolution for image-mode OCR
PER_PAGE_IMAGE_MB = 4.0  # per-page cap; generous so the ladder in
# _render_page_image effectively never fires. At 1.8 MB a 400 DPI page could
# silently fall back to 250 DPI — raising the slider LOWERED the resolution.
# A page this covered by raster images is a photograph of paper, not a
# digitally created page — any text it reports came from a legacy OCR pass.
SCANNED_PAGE_IMAGE_COVERAGE = 0.60
# Stray-symbol ceiling for a text layer we are willing to trust.
TEXT_LAYER_MAX_SYMBOL_NOISE = 0.02

# ---------------------------------------------------------------------------
# OCR PROMPT
#
# Assembled from blocks rather than written as one flat list, for two reasons:
#   1. Position. In a long prompt the beginning and the end are attended to
#      most, so the rules protecting database fields (names, digits) sit at the
#      top and a verification checklist sits at the bottom.
#   2. Relevance. Handwriting and typewriter rules are noise on a clean modern
#      print, and dilute the rules that matter. They are only included when the
#      user tells the app the document is old or handwritten.
# ---------------------------------------------------------------------------

PROMPT_MISSION = (
    "You are a meticulous OCR engine for Bengali (Bangla) and English documents — the minutes of "
    "university council meetings, printed, typewritten or handwritten, often decades old. "
    "Transcribe every page exactly as it appears."

    "\n\nWHY THIS MATTERS — read this before any rule. Every NAME, DATE and NUMBER you transcribe "
    "is copied verbatim into a permanent institutional database, where it identifies real people, "
    "and may be relied on before a human catches a transcription error. A name that is fluent, plausible and "
    "wrong is therefore the worst output you can produce: it is indistinguishable from a correct "
    "one and it corrupts the record silently. Your priorities, highest first:"
    "\n  (1) every character of every name, date and number matches this page exactly;"
    "\n  (2) nothing is invented;"
    "\n  (3) nothing is omitted;"
    "\n  (4) the text reads naturally."
    "\nWhen (1) and (4) conflict, (1) wins every time. Guessing in order to produce clean, "
    "complete-looking text is a FAILURE — even when the guess is reasonable, and even when you "
    "feel certain."
)

PROMPT_NON_NEGOTIABLE = (
    "\n\n=== THE FOUR NON-NEGOTIABLE RULES ===",

    "\n\nN1. THIS LINE IS THE ONLY EVIDENCE. Decide every character from the strokes on the line "
    "you are transcribing right now. Never let any of the following change a character: another "
    "entry, another page, the same word written elsewhere in the document, how common a spelling "
    "is, what would make a list look consistent, what a similar document usually says, or your own "
    "knowledge of Bengali names, departments and places. Two adjacent entries that come out "
    "looking identical is a normal and acceptable result — consistency is never a reason to alter "
    "a letter. This single rule prevents most serious errors.",

    "\n\nN2. NAMES ARE COPIED GLYPH BY GLYPH. A person's name has no 'correct' form other than the "
    "one printed. Never regularize a name toward a more familiar or more frequent spelling, never "
    "complete it from memory, and never 'correct' an unusual one — an unfamiliar name is EXPECTED. "
    "If the page prints শাখাওয়াৎ, write শাখাওয়াৎ, never the commoner সাখাওয়াৎ. Watch every "
    "distinction inside a name: শ vs ষ vs স, ত vs ৎ, ন্ন vs ন্দ, ল vs ল্ল, every vowel sign, every "
    "conjunct, র-ফলা and য-ফলা, and the presence or ABSENCE of চন্দ্রবিন্দু (খান and খাঁন are "
    "different people). Writing a different real name — 'ফোয়াদ খান' where the page prints "
    "'ফোরকান উদ্দিন' — is a total failure, not a small error.",

    "\n\nN3. DIGITS GET NO HELP FROM CONTEXT. A misread letter usually yields a word that looks "
    "wrong and can be caught; a misread digit yields a date that looks perfectly normal and can "
    "never be caught. So read each digit from its own strokes and NEVER infer one from another "
    "date, from the meeting's own date, from a neighbouring item number, from an expected "
    "sequence, or from what would be a plausible day, month or year. These pairs are routinely "
    "confused and must be re-read individually: ৩/৬, ১/৭, ২/৩, ৪/৮, ৫/৬, ৬/৯, ৭/৯, ০/৩, ৫/১, ৯/১. "
    "Follow the pen: where the stroke starts, whether it closes into a loop, which way the tail "
    "turns."
    "\n     DATES NEED COUNTING. The danda । is a plain vertical stroke and several digits (৯, ১, "
    "৪, ৭) end in a vertical or descending stroke, so where a digit meets a separator the two can "
    "merge into what looks like ONE mark and a digit vanishes without looking ambiguous — ২৯।৫।৭৪ "
    "becomes ২।৫।৭৪. Therefore take each field of a date separately and COUNT its digit shapes, "
    "checking specifically for a second digit immediately BEFORE each danda. A one-digit day or "
    "month is something you must actually see, never a default. Never drop a digit because its "
    "stroke touches a separator, or a separator because it touches a digit. A number followed by a "
    "danda in mid-sentence (…জন্য ২৯।৫।৭৪ তারিখে…) is a DATE, not a numbered list item.",

    "\n\nN4. WHEN YOU CANNOT READ SOMETHING, MARK IT — do not fill the gap. Write [?] for the "
    "single character or word you cannot read and keep everything around it: শাখা[?]য়াৎ, "
    "১৮-[?]-৭৪. For an ORDINARY WORD, work it out from the strokes first; most degraded words can "
    "be read, and [?] should not become a habit. For a NAME, a DIGIT or a DATE the balance is "
    "reversed: if the strokes do not settle it, [?] is the CORRECT answer and a best guess is "
    "wrong, because a marked gap gets checked by a human while a plausible guess never does. "
    "Never use [?] for a whole line you merely find difficult.",
)

PROMPT_EXAMPLE = (
    "\n\n=== WORKED EXAMPLE (shows the required format — never copy this content) ==="
    "\nA page printing this attendee list:"
    "\n     ৩। অধ্যাপক ডঃ মোঃ শাখাওয়াৎ হোসেন ফিরোজ        সদস্য"
    "\n        প্রধান, রসায়ন বিভাগ"
    "\n     ৪। অধ্যাপক ডঃ আবু সিদ্দিক                      সদস্য"
    "\n        ডীন, পুরকৌশল অনুষদ"
    "\n     ৫। 〃                                          সদস্য"
    "\nis transcribed as exactly this and nothing else:"
    "\n=== PAGE 7 ==="
    "\n৩। অধ্যাপক ডঃ মোঃ শাখাওয়াৎ হোসেন ফিরোজ - সদস্য"
    "\nপ্রধান, রসায়ন বিভাগ"
    "\n৪। অধ্যাপক ডঃ আবু সিদ্দিক - সদস্য"
    "\nডীন, পুরকৌশল অনুষদ"
    "\n৫। 〃 - সদস্য"
    "\nWhat this shows: the rare name is copied letter for letter and not normalized; each entry's "
    "affiliation is read from its own line; the role stays on the same line as the person; the "
    "ditto mark is transcribed as printed, never expanded; the added ' - ' separates "
    "the name from its following role; no commentary is added."
)


PROMPT_NAME_POSITION = (
    "\n\n=== NAME-POSITION SEPARATOR (required output formatting) ==="
    "\nIn a plain-text attendee/member list, separate each person's name from the "
    "position, office, department or role that follows it with exactly ' - ' "
    "(one ASCII hyphen with a space on each side). Use the visible page layout "
    "to identify the boundary, including when the position starts on the next line. "
    "For a wrapped entry, place the separator after the name before that line break; "
    "retain the original line order and keep all following position lines with that person."
    "\nThis separator is the ONLY permitted editorial punctuation addition. "
    "Do not change, correct, expand or reorder any name, title, abbreviation, "
    "department, role, digit or ditto mark. Keep honorifics and titles BEFORE a name "
    "(such as অধ্যাপক, ডঃ, মোঃ, জনাব, Prof. and Dr.) with the name; do not insert "
    "a separator inside a name or between these prefixes and the name. "
    "Insert only at the name-to-position boundary, not between every position field."
    "\nIf a separator is already printed at that boundary, preserve it and do not "
    "add another. If name and position occupy separate Markdown table cells, keep "
    "the table cells as the separation; do not add a hyphen inside either cell. "
    "Do not add separators to headings, narrative paragraphs, agenda items or "
    "signature blocks. If the boundary is unclear or no position is visible, "
    "keep the transcription as read; never guess a person's position or name."
)

PROMPT_COMPLETENESS = (
    "\n\n=== PAGE MARKERS AND COMPLETENESS (mandatory) ==="
    "\nA. Before each page's content write a line exactly like '=== PAGE n ===', where n is the "
    "page number WITHIN THIS PDF, starting at 1 and increasing by exactly 1 for every page."
    "\nB. Never skip, merge or reorder pages. A blank page still gets its marker, followed by "
    "nothing."
    "\nC. Never stop early: the final marker's number must equal the number of pages supplied, and "
    "every page in between must have its own marker."
    "\nD. Output each supplied page EXACTLY ONCE, then STOP. Never restart from PAGE 1, never "
    "repeat a page, and never produce a second transcription, a corrected version, an alternative "
    "reading, a summary or a duplicate copy."
)

PROMPT_LAYOUT = (
    "\n\n=== READING ORDER ==="
    "\nL1. Default to a single column, read top to bottom. If you are ever unsure whether a page "
    "has true columns, treat it as ONE column — a wrong column split is far worse than a "
    "conservative single-column read."
    "\nL2. An area is a true PAGE COLUMN only if it is a large, independent vertical region "
    "separated by a clear gutter, with its own continuous top-to-bottom flow spanning a "
    "substantial part of the page. Read the leftmost such column fully, then the next to its "
    "right."
    "\nL3. Fields separated horizontally on one line are NOT columns. Names, designations, "
    "offices, numbers and status labels such as সভাপতি or সদস্য sitting to the right of an entry "
    "belong to that entry, on that line. Short right-aligned labels and page numbers never create "
    "a column."
    "\nL4. In an attendee or member list each numbered entry is ONE record. Transcribe everything "
    "belonging to it — name, designation, department, office, role — before moving to the next "
    "entry, in natural left-to-right order, and never move a field from one entry into another."
    "\nL5. Headings, titles, dates and any full-width text above the columns come before them; "
    "full-width text below comes after."
    "\nL6. Tables, tabular rows, aligned lists and forms are never split into columns. Keep each "
    "table together as a Markdown pipe table with one header row, one separator row and every "
    "data row, in printed row order. Preserve the EXACT number and position of columns, including "
    "every empty cell. If the printed top-left header cell is blank, the Markdown header MUST begin "
    "with that blank cell (for example: | | Marks | Grade |). The header, separator and every data "
    "row must have the same number of cells. Never drop the first label column, never shift a value "
    "under the wrong heading, and never omit a final column such as Grade Point. For a printed merged "
    "cell, put its text in the leftmost covered column and leave the other covered columns empty."
)

PROMPT_BENGALI = (
    "\n\n=== BENGALI SCRIPT FIDELITY ==="
    "\nB1. Transcribe exactly as printed, distinguishing: ি/ী, ু/ূ, ে/ৈ, ব/র, য/য়, ড/ড়, ঢ/ঢ়, ত/ৎ, "
    "ং/ঁ/ঃ, ল/ন, ঘ/য, শ/ষ/স, ছ/স, ণ/ন, ই/ঈ. Preserve every conjunct as printed — ক্ষ, জ্ঞ, ত্ত, ন্ত, "
    "স্ত, ষ্ট, ন্ড, ঙ্গ, চ্ছ, দ্ধ, ম্ব — and never drop or reorder reph, র-ফলা or য-ফলা (র্ক, ক্র, ক্য)."
    "\nB2. Use ONLY Bengali script for Bengali words. Never substitute a visually similar "
    "Devanagari, Assamese (ৰ, ৱ), Latin or Arabic character inside a Bengali word."
    "\nB3. Keep Bengali numerals (০১২৩৪৫৬৭৮৯) as printed; never convert them to 0-9 or the "
    "reverse. Keep the danda '।' where printed; never replace it with '.'."
    "\nB4. NEVER expand an abbreviation, however certain you are of its meaning: 'ত.ই কৌশল অনুষদ' "
    "stays 'ত.ই কৌশল অনুষদ' (NOT 'তড়িৎ ও ইলেক্ট্রনিক কৌশল অনুষদ'); 'ইলেকঃ', 'ইঞ্জিঃ', 'সি.এস.ই', "
    "'পরিঃ' (NOT 'পরিশিষ্ট'), 'স্বাঃ', 'অনুঃ' all stay exactly as written. Keep honorifics as "
    "printed: অধ্যাপক, ডঃ, ড., জনাব, মোঃ, মোছাঃ."
    "\nB5. Preserve headings, paragraphs, list order, line breaks and punctuation. Do not omit "
    "text that merely looks duplicated, unless it is clearly a repeated page header or footer."
    "\nB6. NEVER RENUMBER A LIST. Copy each printed item number digit for digit even if the "
    "sequence then has a gap or a repeat — the page is the authority, not the sequence. If an item "
    "number is unreadable, write [?]। before that item and do NOT shift later numbers to make the "
    "list look sequential. An unnumbered paragraph at the TOP of a page may be a new item whose "
    "number sits in a damaged margin: inspect the left edge before treating it as a continuation."
    "\nB7. A ditto mark — 〃, \", or the typed English \"-do-\" — means 'same as the line above'. "
    "Transcribe the mark itself at its position; never drop it and never expand it."
)

PROMPT_SIGNATURE = (
    "\n\n=== THE CLOSING SIGNATURE BLOCK — HIGHEST RISK ON THE PAGE ==="
    "\nS1. The block that ends the minutes (a name in parentheses, a designation such as "
    "রেজিস্ট্রার (অঃ দাঃ), then ও, then একাডেমিক কাউন্সিলের সচিব) looks like boilerplate you have "
    "seen many times, and that is the trap: the designation lines are fixed, but the NAME inside "
    "the parentheses changes with every document and every year. Never complete this block from "
    "memory or from what such a block usually contains — read the name letter by letter from THIS "
    "image, and transcribe EVERY line including the last one after ও. Never stop early because the "
    "remaining lines seem predictable."
    "\nS2. Printed pages often carry handwritten additions — a date beside a signature, a "
    "reference number, a correction, a marginal note. Transcribe those at their position; a "
    "handwritten date next to a signature belongs with that signature block. Do not transcribe the "
    "signature strokes themselves, and never let a Latin-script signature influence the printed "
    "Bengali name below it."
)

PROMPT_HANDWRITING = (
    "\n\n=== HANDWRITTEN AND DEGRADED PAGES ==="
    "\nH1. This document may be entirely handwritten, decades old, faded, stained, photocopied or "
    "low-contrast. Every rule above still applies. Read stroke by stroke, and remember N1: when a "
    "letter is ambiguous, decide it from the strokes on this line, never from a more familiar word "
    "or a spelling you have seen elsewhere in the document."
    "\nH2. Old minutes abbreviate heavily: প্রফেঃ / প্রফেসর, সহযোগী প্রফেঃ, ডঃ / ড. / ডীন, অনুঃ "
    "(অনুষদ), পরিঃ (পরিশিষ্ট), স্বাঃ (স্বাক্ষর), ভাইস চ্যান্সেলার. Transcribe each exactly as "
    "written; never expand."
    "\nH3. OLD TYPEWRITTEN ENGLISH pages (1960s EPUET minutes) come from a manual typewriter: "
    "letters sit unevenly, overtyped corrections and hand-inked marks are common, and a carbon "
    "copy may be faint. Read strictly in printed top-to-bottom order and transcribe what the "
    "typist intended (e.g. 'Vice-Chancellor', not a mis-struck lookalike), keeping numbered member "
    "lists in their printed order with their printed numbers. If the file also carries an "
    "invisible or scrambled machine-text layer, IGNORE it — only the visible page image counts."
)

PROMPT_CLOSING = (
    "\n\n=== OUTPUT FORMAT ==="
    "\nOutput only the extracted text, in Markdown. No commentary, no explanations, no layout "
    "labels such as 'left column', no translations, no notes."

    "\n\n=== BEFORE YOU FINISH, VERIFY ==="
    "\n  - every supplied page has exactly one '=== PAGE n ===' marker, in order, none repeated;"
    "\n  - every name was read letter by letter from its own line, not normalized to a familiar one;"
    "\n  - EVERY printed number was verified digit by digit directly from the page image a second time;"
    "\n  - this includes dates, proposal numbers, agenda numbers, serial numbers, student IDs, registration numbers, credit values, page numbers and list item numbers;"
    "\n  - no digit was inferred from context, sequence, neighbouring entries or what would look plausible;"
    "\n  - every date had its fields counted separately, with no digit lost against a danda;"
    "\n  - anything unreadable is marked [?] rather than filled in with a plausible guess;"
    "\n  - plain-text attendee entries use the required name-position separator wherever "
    "the boundary is clear; names, prefixes and position wording are unchanged;"
    "\n  - nothing has been added that is not on the page except the explicitly "
    "requested name-position separator."
)


def build_chunk_prompt(handwritten: bool = True) -> str:
    """Assemble the OCR prompt. Handwriting and typewriter rules are included
    only for old/handwritten documents, where they earn their tokens."""
    parts = [PROMPT_MISSION, *PROMPT_NON_NEGOTIABLE, PROMPT_EXAMPLE,
             PROMPT_COMPLETENESS, PROMPT_LAYOUT, PROMPT_BENGALI, PROMPT_SIGNATURE,
             PROMPT_NAME_POSITION]
    if handwritten:
        parts.append(PROMPT_HANDWRITING)
    parts.append(PROMPT_CLOSING)
    return "".join(parts)


# Full prompt, used for the token estimate in the sidebar.
CHUNK_PROMPT = build_chunk_prompt(handwritten=True)


def chunk_prompt_for(
    input_mode: str, expected_pages: int, preprocess: str = "degraded"
) -> str:
    """Return the OCR prompt adapted to the payload type and document type."""
    prompt = build_chunk_prompt(handwritten=(preprocess in ("degraded", "strong")))
    if input_mode == "pdf":
        return prompt
    prompt = prompt.replace("this PDF", "this ordered set of page images")
    prompt = prompt.replace("THIS PDF", "THIS IMAGE SET")
    return (
        f"You are given exactly {expected_pages} scanned page image(s) in reading "
        f"order. Image k is page k of this set.\n\n" + prompt
    )


# ==========================================
# Page setup
# ==========================================
def scroll_box(height: int = 420, border: bool = True):
    """A fixed-height panel whose content scrolls inside it.

    Long previews otherwise stretch the page to many thousands of pixels,
    which buries every control underneath them. `st.container(height=...)`
    has existed since Streamlit 1.31; on anything older the keyword is
    rejected, so fall back to a plain container rather than crashing the
    app over a cosmetic detail.
    """
    try:
        return st.container(height=height, border=border)
    except TypeError:
        return st.container()


def code_block(text: str, language: str = "json", height: int = 460):
    """A syntax-highlighted block that scrolls inside a fixed height.

    Recent Streamlit takes `height` and `line_numbers` on st.code directly,
    which scrolls better than wrapping the block in a container. Older
    releases reject those keywords, so fall back to the container.
    """
    try:
        st.code(text, language=language, height=height, line_numbers=True)
    except TypeError:
        with scroll_box(height):
            st.code(text, language=language)


def _unique_api_keys(candidates: list) -> list:
    """Return configured, non-empty API keys in order, with duplicates removed."""
    ordered = []
    seen = set()
    for candidate in candidates:
        key = str(candidate or "").strip()
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


api_keys = _unique_api_keys(GEMINI_API_KEYS)
api_key = api_keys[0] if api_keys else ""

# ==========================================
# Helpers
# ==========================================
def split_pdf_into_chunks(pdf_bytes: bytes, pages_per_chunk: int):
    """Split a PDF into smaller PDFs.

    Returns a list of (start_page, end_page, chunk_pdf_bytes) with 1-based,
    inclusive page numbers. If a chunk exceeds the inline size limit, it is
    recursively split in half.
    """
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(src)
    chunks = []

    def build_chunk(start: int, end: int):
        """start/end are 0-based inclusive page indices."""
        part = fitz.open()
        part.insert_pdf(src, from_page=start, to_page=end)
        # Avoid expensive recompression unless the uncompressed chunk is too large.
        data = part.tobytes(garbage=1, deflate=False)
        if len(data) > MAX_INLINE_MB * 1024 * 1024:
            data = part.tobytes(garbage=3, deflate=True)
        part.close()
        if len(data) > MAX_INLINE_MB * 1024 * 1024 and end > start:
            # Too big to send inline — split in half and try again.
            mid = (start + end) // 2
            build_chunk(start, mid)
            build_chunk(mid + 1, end)
        else:
            chunks.append((start + 1, end + 1, data))

    for chunk_start in range(0, total_pages, pages_per_chunk):
        chunk_end = min(chunk_start + pages_per_chunk, total_pages) - 1
        build_chunk(chunk_start, chunk_end)

    src.close()
    chunks.sort(key=lambda c: c[0])
    return chunks, total_pages


class RequestStartLimiter:
    """Space request starts while still allowing requests to overlap."""

    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / max(1, requests_per_minute)
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + self.interval
        if delay:
            time.sleep(delay)


def _is_quota_error(error: Exception) -> bool:
    msg = str(error).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "QUOTA" in msg


def _is_key_auth_error(error: Exception) -> bool:
    msg = str(error).upper()
    return (
        "API KEY" in msg
        or "401" in msg
        or "403" in msg
        or "UNAUTHENTICATED" in msg
        or "PERMISSION_DENIED" in msg
    )


class APIKeyPool:
    """Thread-safe primary-key failover with one RPM limiter per key.

    Requests stay on the active key. A key is replaced only after two quota
    errors or immediately after a key/authentication error. Quota-failed keys
    are skipped for the rest of the current OCR/JSON run, while malformed-
    request and model errors are never hidden by switching keys.
    """

    def __init__(self, keys: list, requests_per_minute: int, unavailable=None):
        self.keys = list(dict.fromkeys(k for k in keys if k))
        if not self.keys:
            raise ValueError("At least one Gemini API key is required.")
        self._lock = threading.Lock()
        self._active_index = 0
        self._quota_failures = {key: 0 for key in self.keys}
        # Keys already known to be out of quota (carried over from an earlier
        # run in this session) start retired, so a resume does not waste
        # attempts rediscovering them. Never start with every key retired.
        restored = {key for key in (unavailable or ()) if key in self.keys}
        if len(restored) >= len(self.keys):
            restored = set()
        self._unavailable = restored
        self._limiters = {
            key: RequestStartLimiter(requests_per_minute) for key in self.keys
        }

    def current_key(self) -> str:
        with self._lock:
            key = self.keys[self._active_index]
            if key not in self._unavailable:
                return key
            replacement = self._next_available_locked(self._active_index)
            if replacement is None:
                raise RuntimeError(
                    "All configured Gemini API keys are unavailable for this run. "
                    "Check the project quotas and key permissions, then resume later."
                )
            self._active_index = replacement
            return self.keys[self._active_index]

    def wait(self, key: str):
        self._limiters[key].wait()

    def record_success(self, key: str):
        with self._lock:
            self._quota_failures[key] = 0

    def report_error(self, key: str, error: Exception) -> bool:
        """Record a key-related error and return True when failover occurred."""
        quota_error = _is_quota_error(error)
        auth_error = _is_key_auth_error(error)
        if not quota_error and not auth_error:
            return False

        with self._lock:
            if quota_error:
                self._quota_failures[key] = self._quota_failures.get(key, 0) + 1
                # With several keys configured, a 429 means "move on NOW":
                # retrying the same exhausted key only burns this chunk's
                # attempt budget and delays reaching a healthy key. With a
                # single key there is nowhere to go, so absorb the first 429
                # and let ordinary backoff handle it.
                if len(self.keys) == 1 and self._quota_failures[key] < 2:
                    return False

            replacement = self._next_available_locked(
                self.keys.index(key), exclude={key}
            )
            if replacement is None:
                # Keep the only remaining key active so the ordinary retry and
                # backoff path can still recover from a temporary 429.
                return False

            self._unavailable.add(key)
            if self.keys[self._active_index] == key:
                self._active_index = replacement
            return True

    def exhausted_keys(self) -> list:
        """Keys retired during this run, for carrying over to a resume."""
        with self._lock:
            return sorted(self._unavailable)

    def _next_available_locked(self, start_index: int, exclude=None):
        excluded = set(exclude or ()) | self._unavailable
        for offset in range(1, len(self.keys) + 1):
            index = (start_index + offset) % len(self.keys)
            if self.keys[index] not in excluded:
                return index
        return None

    @property
    def count(self) -> int:
        return len(self.keys)


# A key that hit its quota is skipped for the next EXHAUSTED_KEY_TTL seconds,
# then tried again — quota windows roll over, so retiring one forever would
# slowly starve the pool.
EXHAUSTED_KEY_TTL = 600


def remembered_exhausted_keys() -> list:
    """Keys that hit their quota recently, so a resume can skip them."""
    record = st.session_state.get("exhausted_keys") or {}
    now = time.time()
    return [key for key, when in record.items() if now - when < EXHAUSTED_KEY_TTL]


def remember_exhausted_keys(pool: "APIKeyPool") -> None:
    """Carry this run's exhausted keys into the next one."""
    record = dict(st.session_state.get("exhausted_keys") or {})
    now = time.time()
    for key in pool.exhausted_keys():
        record[key] = now
    st.session_state["exhausted_keys"] = {
        key: when for key, when in record.items() if now - when < EXHAUSTED_KEY_TTL
    }


def _page_image_coverage(page) -> float:
    """Fraction of the page area covered by raster images (0.0-1.0).

    A digitally created PDF draws its page from text objects and covers little
    or none of it with images. A SCANNED page is one full-page photograph, so
    whatever text it reports came from a legacy OCR pass baked into the file —
    not from the document itself — and must never be trusted.
    """
    try:
        page_area = float(page.rect.width) * float(page.rect.height)
    except Exception:
        return 0.0
    if page_area <= 0:
        return 0.0

    boxes = []
    try:
        for info in page.get_image_info():
            bbox = info.get("bbox")
            if bbox:
                boxes.append(bbox)
    except Exception:
        boxes = []
    if not boxes:
        # Older PyMuPDF builds: image blocks carry type 1 in the raw dict.
        try:
            for block in (page.get_text("rawdict") or {}).get("blocks", []):
                if block.get("type") == 1 and block.get("bbox"):
                    boxes.append(block["bbox"])
        except Exception:
            return 0.0

    covered = 0.0
    for box in boxes:
        try:
            x0, y0, x1, y1 = box
        except Exception:
            continue
        covered += abs((x1 - x0) * (y1 - y0))
    return min(covered / page_area, 1.0)


def _page_text_is_invisible(page) -> bool:
    """True when most of a page's text is drawn in invisible render mode.

    Tesseract/ABBYY hide their OCR layer behind the scanned image using text
    render mode 3. Real document text is never invisible, so this is a
    definitive "this is a scan, not a digital PDF" signal.
    """
    try:
        spans = page.get_texttrace()
    except Exception:
        return False
    invisible = visible = 0
    for span in spans or ():
        if not isinstance(span, dict):
            continue
        count = len(span.get("chars") or ()) or 1
        if span.get("type") == 3:
            invisible += count
        else:
            visible += count
    return invisible > 0 and invisible >= visible


_ORDINARY_PUNCTUATION = set(
    ".,;:!?()[]{}<>/-+=&%@#$*_|"
    + "'"
    + '"'
    + "\\"
    + "\u2013\u2014\u2018\u2019\u201c\u201d\u2026\u00b0\u00a3\u09f3\u0964\u0965"
)


def _symbol_noise_ratio(text: str) -> float:
    """Fraction of characters that are neither letters/digits/marks nor
    ordinary punctuation.

    A clean text layer sits far below 1%. A scrambled scan layer is littered
    with stray glyphs (^ | ■ • » « ***) and measures several percent.
    """
    non_space = [ch for ch in (text or "") if not ch.isspace()]
    if not non_space:
        return 1.0
    junk = 0
    for ch in non_space:
        if ch.isalnum() or ch in _ORDINARY_PUNCTUATION:
            continue
        if "\u0980" <= ch <= "\u09ff":          # Bengali block, incl. matras
            continue
        if unicodedata.category(ch).startswith("M"):   # combining marks
            continue
        junk += 1
    return junk / len(non_space)


def extract_text_layer_pages(pdf_bytes: bytes):
    """Return (usable_pages, rejected_pages) for the FREE local text path.

    usable_pages   {0-based page index: text} — pages whose embedded text is
                   genuine document text (a digitally created PDF). Extracting
                   these locally avoids an AI request but still needs review.
    rejected_pages {0-based page index: reason} — pages that DO carry text but
                   whose text cannot be trusted, so Gemini must read them.

    WHY THE REJECT LIST EXISTS: a scanned document often ships with a hidden
    legacy OCR layer. That layer is typically scrambled — wrong reading order,
    invented words, stray symbols — and silently preferring it produces a
    confidently wrong transcription at zero cost, which is far worse than
    paying for a correct one. Three independent guards catch it:

      1. the page is essentially a full-page image (a scan),
      2. the text is drawn invisibly (an OCR layer hiding behind that scan),
      3. the text fails the mojibake / stray-symbol quality checks.

    Guards 1 and 2 are decisive on their own; a real digital page has neither.
    """
    pages = {}
    rejected = {}
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for index, page in enumerate(src):
            text = (page.get_text("text", sort=True) or "").strip()

            coverage = _page_image_coverage(page)
            if coverage >= SCANNED_PAGE_IMAGE_COVERAGE:
                if len(text) >= 40:
                    rejected[index] = (
                        f"the page is a scanned image ({coverage:.0%} of the "
                        "page area) carrying a legacy OCR text layer"
                    )
                continue

            if _page_text_is_invisible(page):
                if len(text) >= 40:
                    rejected[index] = (
                        "the text is invisible OCR text hidden behind a scan"
                    )
                continue

            if len(text) < 120:
                continue
            bengali_chars = sum(1 for ch in text if "\u0980" <= ch <= "\u09ff")
            latin_chars = sum(1 for ch in text if ch.isascii() and ch.isalnum())
            if bengali_chars < 30 and latin_chars < 80:
                continue
            # Garbage-layer guard: scans sometimes carry a legacy OCR layer of
            # mojibake (Bengali stored as Latin gibberish full of symbols, e.g.
            # "^IdJTGF (5S PITS"). Real text is overwhelmingly letters/digits
            # (~0.95+ of non-space chars, counting the whole Bengali block).
            non_space = sum(1 for ch in text if not ch.isspace())
            good_chars = sum(
                1
                for ch in text
                if "\u0980" <= ch <= "\u09ff" or (ch.isascii() and ch.isalnum())
            )
            if good_chars / max(1, non_space) < 0.88:
                rejected[index] = "the text layer reads as mojibake"
                continue
            # ENGLISH mojibake layers can be mostly readable ASCII with
            # CJK/fullwidth shrapnel mixed in ("PAKIST域", "血蹈").
            exotic_chars = sum(
                1
                for ch in text
                if "\u2e80" <= ch <= "\u9fff"   # CJK radicals..unified ideographs
                or "\u3000" <= ch <= "\u30ff"   # CJK punctuation, kana
                or "\uac00" <= ch <= "\ud7af"   # Hangul
                or "\uff00" <= ch <= "\uffef"   # fullwidth forms
            )
            if exotic_chars > 2:
                rejected[index] = "the text layer contains CJK/fullwidth shrapnel"
                continue
            noise = _symbol_noise_ratio(text)
            if noise > TEXT_LAYER_MAX_SYMBOL_NOISE:
                rejected[index] = (
                    f"the text layer is {noise:.0%} stray symbols "
                    "(a scrambled scan layer)"
                )
                continue
            try:
                if hasattr(page, "find_tables") and page.find_tables().tables:
                    rejected[index] = "tabular layout needs visual reading to preserve columns"
                    continue
            except Exception:
                pass  # Older PyMuPDF releases may not support table detection.
            pages[index] = text
    finally:
        src.close()
    return pages, rejected


def estimate_ocr_input_tokens(remote_chunks, input_mode: str, dpi: int):
    """Rough input-token estimate for the OCR requests (billed side).

    Image mode is billed by 768px tiles (~258 tokens each) of the rendered
    page, so tokens scale with DPI squared. Native PDF pages cost a flat
    ~258 tokens each. The prompt is re-sent per chunk (implicit caching
    typically discounts most of it after the first request).
    """
    prompt_tokens = max(1, len(CHUNK_PROMPT) // 4)
    page_count = sum(end - start + 1 for start, end, _ in remote_chunks)
    if input_mode == "images":
        width_px = 8.27 * dpi   # A4 width in inches × DPI
        height_px = 11.69 * dpi
        tiles = math.ceil(width_px / 768) * math.ceil(height_px / 768)
        per_page = tiles * 258
    else:
        per_page = 258
    total = len(remote_chunks) * prompt_tokens + page_count * per_page
    return total, per_page


# ==========================================
# OCR output validation + Bengali cleanup
# ==========================================
PAGE_MARKER_RE = re.compile(r"^[ \t]*===\s*PAGE\s+(\d+)\s*===[ \t]*$", re.MULTILINE)


class OCRIncompleteError(Exception):
    """The OCR response was empty, truncated, or missing page markers."""


def pages_in_text(text: str) -> set:
    """Return the set of page numbers whose '=== PAGE n ===' marker is present."""
    return {int(m.group(1)) for m in PAGE_MARKER_RE.finditer(text or "")}


def missing_page_numbers(text: str, expected_pages: int) -> list:
    """List the 1-based local page numbers whose marker is absent."""
    found = pages_in_text(text)
    return [n for n in range(1, expected_pages + 1) if n not in found]


def _response_truncated(response) -> bool:
    """Detect a response cut off by the output-token limit (both SDKs)."""
    try:
        finish = str(response.candidates[0].finish_reason).upper()
    except Exception:
        return False
    return "MAX_TOKEN" in finish or finish.endswith("LENGTH")


class ChunkTooLargeError(OCRIncompleteError):
    """Rendered image payload exceeds the inline limit.

    Subclasses OCRIncompleteError so ocr_chunk_bulletproof's existing repair
    path (split the chunk in half and recurse) handles it automatically.
    """


def _native_raster_dpi(page) -> float:
    """Measure a dominant scan using its actual on-page bounding box."""
    try:
        area = float(page.rect.width * page.rect.height)
        for info in page.get_image_info():
            x0, y0, x1, y1 = info["bbox"]
            width, height = abs(x1 - x0), abs(y1 - y0)
            if width * height < area * 0.60 or not width or not height:
                continue  # A small logo must never lower the text resolution.
            return min(info["width"] * 72 / width, info["height"] * 72 / height)
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return 0.0


def effective_render_dpi(page, requested_dpi: int) -> int:
    """Preserve scan detail while bounding pixel memory for huge-format pages."""
    native = _native_raster_dpi(page)
    dpi = min(requested_dpi, native * 2) if native > 0 else requested_dpi
    pixel_limit_dpi = 72 * math.sqrt(24_000_000 / max(1, page.rect.width * page.rect.height))
    return max(1, int(min(dpi, pixel_limit_dpi)))


def _image_mime(data: bytes) -> str:
    """PNG or JPEG, decided from the bytes rather than assumed."""
    return "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"


def _render_page_image(page, dpi: int, preprocess: str = "standard", clip=None) -> bytes:
    """Lossless first; gentle cleanup preserves faint strokes and vowel signs."""
    cap = int(PER_PAGE_IMAGE_MB * 1024 * 1024)
    render_dpi = effective_render_dpi(page, dpi)
    levels = list(dict.fromkeys([render_dpi, max(1, int(render_dpi * .75)), max(1, int(render_dpi * .55))]))
    data = b""
    for level in levels:
        original = preprocess == "original"
        pix = page.get_pixmap(dpi=level, colorspace=fitz.csRGB if original else fitz.csGRAY,
                              alpha=False, clip=clip)
        if not PIL_AVAILABLE:
            data = pix.tobytes("png")
            if len(data) <= cap:
                return data
            continue
        img = Image.frombytes("RGB" if original else "L", (pix.width, pix.height), pix.samples)
        if not original:
            img = ImageOps.autocontrast(img, cutoff=1 if preprocess == "strong" else 0)
        if preprocess == "strong":
            img = ImageEnhance.Contrast(img).enhance(1.3)
        for encoder in ("PNG", 92, 85):
            buf = io.BytesIO()
            if encoder == "PNG":
                img.save(buf, "PNG", optimize=True)
            else:
                img.save(buf, "JPEG", quality=encoder, optimize=True)
            data = buf.getvalue()
            if len(data) <= cap:
                return data
    return data


def render_chunk_page_images(
    chunk_pdf_bytes: bytes, dpi: int, preprocess: str = "standard"
) -> list:
    """Render every page of a chunk PDF as a preprocessed JPEG (in order)."""
    src = fitz.open(stream=chunk_pdf_bytes, filetype="pdf")
    try:
        return [_render_page_image(page, dpi, preprocess) for page in src]
    finally:
        src.close()


def render_page_detail_views(pdf_bytes, dpi, preprocess="original"):
    """One full page and two overlapping crops; every image shows the SAME page.

    Crops retain color/faint strokes by default. They supplement the full-page
    layout and never replace it, so columns, table cells and overlap stay clear.
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        if len(document) != 1:
            raise ValueError("Detailed verification requires exactly one PDF page.")
        page = document[0]
        r = page.rect
        crops = [fitz.Rect(r.x0, r.y0, r.x1, r.y0 + r.height * .57),
                 fitz.Rect(r.x0, r.y0 + r.height * .43, r.x1, r.y1)]
        return [_render_page_image(page, dpi, preprocess)] + [
            _render_page_image(page, min(600, max(400, dpi)), preprocess, clip=crop)
            for crop in crops
        ]


# OCR occasionally emits Assamese/Devanagari lookalikes or zero-width characters
# inside Bengali text. In this corpus these are always artifacts, so they are
# fixed deterministically and locally (no extra API request).
BENGALI_CHAR_FIXES = {
    "\u09f0": "\u09b0",  # Assamese ৰ  -> Bengali র
    "\u09f1": "\u09ac",  # Assamese ৱ  -> Bengali ব
    "\u200b": "",        # zero-width space
    "\ufeff": "",        # byte-order mark
    "\u00ad": "",        # soft hyphen
}
DEVANAGARI_TO_BENGALI_DIGITS = str.maketrans("०१२३४५६७८९", "০১২৩৪৫৬৭৮৯")
BENGALI_TO_ARABIC_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def clean_bengali_ocr_text(text):
    """Deterministic local cleanup of common Bengali OCR artifacts.

    - Unicode NFC normalization so identical words always compare equal.
    - Compose decomposed nukta letters (ড + ়  -> ড়, etc.).
    - Replace Assamese lookalike letters and Devanagari digits.
    - Strip zero-width characters that break matching and search.
    Page markers are left untouched.
    """
    if not text:
        return text
    value = unicodedata.normalize("NFC", str(text))
    for bad, good in BENGALI_CHAR_FIXES.items():
        value = value.replace(bad, good)
    value = value.translate(DEVANAGARI_TO_BENGALI_DIGITS)
    # NFC leaves these three composition-excluded letters decomposed; compose
    # them so every occurrence of ড়/ঢ়/য় is byte-identical across the document.
    value = value.replace("\u09a1\u09bc", "\u09dc")  # ড়
    value = value.replace("\u09a2\u09bc", "\u09dd")  # ঢ়
    value = value.replace("\u09af\u09bc", "\u09df")  # য়
    return value


_thread_local = threading.local()
LEGACY_SDK_LOCK = threading.RLock()


def get_new_client(api_key: str):
    """Create one reusable client per worker thread."""
    client = getattr(_thread_local, "new_client", None)
    client_key = getattr(_thread_local, "new_client_key", None)
    if client is None or client_key != api_key:
        client = genai_new.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=120_000),
        )
        _thread_local.new_client = client
        _thread_local.new_client_key = api_key
    return client


def get_legacy_model(api_key: str, model_name: str):
    """Create one reusable legacy model per worker thread."""
    cache_key = (api_key, model_name)
    model = getattr(_thread_local, "legacy_model", None)
    model_key = getattr(_thread_local, "legacy_model_key", None)
    if model is None or model_key != cache_key:
        genai_old.configure(api_key=api_key)
        model = genai_old.GenerativeModel(model_name)
        _thread_local.legacy_model = model
        _thread_local.legacy_model_key = cache_key
    return model


def legacy_generate(api_key, model_name, contents, generation_config):
    # The legacy SDK has global credential configuration. Serialize all legacy
    # calls across jobs so one user's key cannot bleed into another request.
    with LEGACY_SDK_LOCK:
        genai_old.configure(api_key=api_key)
        return genai_old.GenerativeModel(model_name).generate_content(
            contents, generation_config=generation_config, request_options={"timeout": 120}
        )


def new_sdk_config(**kwargs):
    """Use the thinking control supported by the selected model family/SDK."""
    level = kwargs.pop("_thinking_level", "minimal")
    model = kwargs.pop("_model_name", "").lower()
    thinking = getattr(genai_types, "ThinkingConfig", None)
    fields = getattr(thinking, "model_fields", {})
    if thinking and model.startswith("gemini-2.5") and "thinking_budget" in fields:
        budget = -1 if "pro" in model else (0 if level == "minimal" else 1024)
        kwargs["thinking_config"] = thinking(thinking_budget=budget)
    elif thinking and "thinking_level" in fields:
        kwargs["thinking_config"] = thinking(thinking_level=resolve_thinking_level(model, level))
    return genai_types.GenerateContentConfig(**kwargs)


def retry_delay(error: Exception, attempt: int) -> float:
    """Use a server retry hint when available, otherwise exponential backoff."""
    raw_msg = str(error)
    msg = raw_msg.upper()

    # Gemini errors may include values such as retryDelay: "12s" or
    # "retry after 12 seconds". Using the supplied value avoids sleeping longer
    # than necessary while still respecting the service response.
    retry_patterns = (
        r"retry(?:delay|[-_ ]after)?[^0-9]{0,20}(\d+(?:\.\d+)?)\s*s",
        r"please retry in[^0-9]{0,20}(\d+(?:\.\d+)?)\s*s",
    )
    for pattern in retry_patterns:
        match = re.search(pattern, raw_msg, flags=re.IGNORECASE)
        if match:
            return min(float(match.group(1)) + random.uniform(0.2, 0.8), 120)

    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "QUOTA" in msg:
        return min(5 * (2 ** attempt) + random.uniform(0.5, 2.5), 60)
    if (
        "503" in msg
        or "UNAVAILABLE" in msg
        or "HIGH DEMAND" in msg
        or "500" in msg
        or "502" in msg
        or "504" in msg
        or "408" in msg
        or "TIMEOUT" in msg
    ):
        return min(2 * (2 ** attempt) + random.uniform(0.5, 2.0), 30)
    return min(2 * (2 ** attempt) + random.uniform(0.5, 1.5), 15)


def _is_fatal_api_error(error: Exception) -> bool:
    """Auth/model errors that must never be retried."""
    msg = str(error).upper()
    return (
        "API KEY" in msg
        or "401" in msg
        or "403" in msg
        or "PERMISSION" in msg
        or "UNAUTHENTICATED" in msg
        or "INVALID_ARGUMENT" in msg
        or "404" in msg
        or "NOT_FOUND" in msg
    )


def ocr_chunk_with_gemini(
    chunk_pdf_bytes: bytes,
    expected_pages: int,
    key_pool: APIKeyPool,
    model_name: str,
    input_mode: str = "pdf",
    dpi: int = OCR_IMAGE_DPI,
    preprocess: str = "standard",
    detail_views: bool = False,
    verification: bool = False,
) -> str:
    """Send one chunk (as a PDF or as preprocessed page images) and VALIDATE
    the result before accepting it.

    A response is rejected (and retried) when it is empty, truncated by the
    output-token limit, or missing any '=== PAGE n ===' marker for the pages
    this chunk contains. Only a complete, verified transcription is returned.
    """
    if detail_views and expected_pages != 1:
        raise ValueError("Detailed verification requires a single page.")
    if detail_views:
        input_mode = "images"
    prompt = chunk_prompt_for("pdf" if detail_views else input_mode, expected_pages, preprocess)
    if detail_views:
        prompt = prompt.replace("this PDF", "these views of one page").replace("THIS PDF", "THESE VIEWS OF ONE PAGE")
    if verification:
        prompt += (
            "\n\nMake an independent transcription from the attached source only. "
            "Inspect every name, digit, date, table row, margin and footnote. "
            "Do not infer missing text from a familiar phrase or number sequence. "
            "Keep spelling exactly as printed, including unusual names. "
            "Use [?] for strokes you cannot resolve. Transcribe all visible content; "
            "do not summarize or report your reasoning."
        )
    if detail_views:
        prompt += (
            "\n\nIMAGE MAPPING: These THREE images are VIEWS OF ONE PAGE, not three pages. "
            "Image 1 is the full page. Image 2 enlarges its upper part. Image 3 enlarges "
            "its lower part. Images 2 and 3 overlap in the middle. Use the full image "
            "for reading order and the crops for small strokes. Output the entire page "
            "ONCE with exactly one === PAGE 1 === marker. Never duplicate overlapping "
            "lines or split a table into duplicate rows."
        )

    # Build payload parts ONCE, before the retry loop.
    if input_mode == "images":
        images = (render_page_detail_views(chunk_pdf_bytes, dpi, preprocess) if detail_views
                  else render_chunk_page_images(chunk_pdf_bytes, dpi, preprocess))
        total_bytes = sum(len(img) for img in images)
        if total_bytes * 4 / 3 + len(prompt.encode("utf-8")) > MAX_INLINE_MB * 1024 * 1024:
            if expected_pages > 1:
                # Raised BEFORE any request — bulletproof splits the chunk.
                raise ChunkTooLargeError(
                    f"rendered images total {total_bytes / 1024 / 1024:.1f} MB "
                    f"for {expected_pages} page(s) — splitting chunk"
                )
            raise RuntimeError(
                "a single page image exceeds the inline payload limit even at "
                "minimum quality — lower the render DPI in the sidebar"
            )
        if NEW_SDK:
            payload_parts = [
                genai_types.Part.from_bytes(data=img, mime_type=_image_mime(img))
                for img in images
            ]
        else:
            payload_parts = [
                {"mime_type": _image_mime(img), "data": img} for img in images
            ]
    else:
        if len(chunk_pdf_bytes) > MAX_INLINE_MB * 1024 * 1024:
            if expected_pages > 1:
                raise ChunkTooLargeError("PDF request is too large; splitting pages")
            return ocr_chunk_with_gemini(chunk_pdf_bytes, expected_pages, key_pool, model_name,
                                       input_mode="images", dpi=dpi, preprocess=preprocess,
                                       detail_views=detail_views, verification=verification)
        if NEW_SDK:
            payload_parts = [
                genai_types.Part.from_bytes(
                    data=chunk_pdf_bytes, mime_type="application/pdf"
                )
            ]
        else:
            payload_parts = [
                {"mime_type": "application/pdf", "data": chunk_pdf_bytes}
            ]

    last_error = None

    attempt = 0
    while attempt < OCR_ATTEMPTS:
        api_key = key_pool.current_key()
        key_pool.wait(api_key)
        try:
            if NEW_SDK:
                response = get_new_client(api_key).models.generate_content(
                    model=model_name,
                    # Static prompt FIRST: every request then shares an identical
                    # prefix, so Gemini's implicit prompt caching discounts the
                    # repeated prompt tokens across chunks.
                    contents=[prompt] + payload_parts,
                    config=new_sdk_config(
                        _model_name=model_name,
                        temperature=0.0,
                        _thinking_level=resolve_thinking_level(
                            model_name, getattr(key_pool, "ocr_thinking", OCR_THINKING_LEVEL)
                        ),
                    ),
                )
            else:
                response = legacy_generate(api_key, model_name, [prompt] + payload_parts,
                                           {"temperature": 0.0})

            text = (response.text or "").strip()
            if not text:
                raise OCRIncompleteError("empty OCR response")
            if _response_truncated(response):
                raise OCRIncompleteError(
                    "OCR response truncated by the output-token limit"
                )
            actual_markers = [int(m.group(1)) for m in PAGE_MARKER_RE.finditer(text)]
            missing = missing_page_numbers(text, expected_pages)
            if actual_markers != list(range(1, expected_pages + 1)):
                raise OCRIncompleteError(
                    f"OCR page markers are missing, repeated, or out of order: {actual_markers}; missing {missing}. "
                    f"out of {expected_pages} expected page(s)"
                )
            key_pool.record_success(api_key)
            return text

        except OCRIncompleteError as e:
            if expected_pages > 1:
                raise  # Reduce visual context immediately instead of retrying a large batch.
            last_error = e
            attempt += 1
            if attempt < OCR_ATTEMPTS:
                time.sleep(retry_delay(e, attempt - 1))
        except Exception as e:
            last_error = e
            # A key swap is NOT a retry. Moving to a fresh key costs nothing,
            # so it must not consume the attempt budget — otherwise a run of
            # exhausted keys is never traversed. Bounded by the key count:
            # the last surviving key is never retired, and failures on it
            # fall through to ordinary backoff below.
            if key_pool.report_error(api_key, e):
                continue
            if _is_fatal_api_error(e):
                raise
            attempt += 1
            if attempt < OCR_ATTEMPTS:
                time.sleep(retry_delay(e, attempt - 1))

    raise last_error


def _split_pdf_bytes_in_half(chunk_pdf_bytes: bytes):
    """Split a chunk PDF into two halves; returns (left_bytes, left_pages, right_bytes, right_pages)."""
    src = fitz.open(stream=chunk_pdf_bytes, filetype="pdf")
    total = len(src)
    mid = total // 2

    left = fitz.open()
    left.insert_pdf(src, from_page=0, to_page=mid - 1)
    left_bytes = left.tobytes(garbage=1, deflate=True)
    left.close()

    right = fitz.open()
    right.insert_pdf(src, from_page=mid, to_page=total - 1)
    right_bytes = right.tobytes(garbage=1, deflate=True)
    right.close()

    src.close()
    return left_bytes, mid, right_bytes, total - mid


def ocr_chunk_bulletproof(
    chunk_pdf_bytes: bytes,
    expected_pages: int,
    key_pool: APIKeyPool,
    model_name: str,
    input_mode: str = "pdf",
    dpi: int = OCR_IMAGE_DPI,
    preprocess: str = "standard",
    detail_views: bool = False,
    verification: bool = False,
) -> str:
    """OCR a chunk with strict validation and automatic repair.

    If the model persistently returns an incomplete transcription for a
    multi-page chunk (missing pages / truncation / empty output even after
    all retries), or the rendered image payload is too large, the chunk is
    split in half and each half is OCR'd independently — recursing down to
    single pages if necessary. The halves are then renumbered and rejoined
    so page markers stay correct.
    """
    try:
        return ocr_chunk_with_gemini(
            chunk_pdf_bytes,
            expected_pages,
            key_pool,
            model_name,
            input_mode=input_mode,
            dpi=dpi,
            preprocess=preprocess,
            detail_views=detail_views,
            verification=verification,
        )
    except OCRIncompleteError:
        if expected_pages <= 1:
            raise

    left_bytes, left_pages, right_bytes, right_pages = _split_pdf_bytes_in_half(
        chunk_pdf_bytes
    )
    left_text = ocr_chunk_bulletproof(
        left_bytes, left_pages, key_pool, model_name,
        input_mode=input_mode, dpi=dpi, preprocess=preprocess,
        detail_views=detail_views, verification=verification,
    )
    right_text = ocr_chunk_bulletproof(
        right_bytes, right_pages, key_pool, model_name,
        input_mode=input_mode, dpi=dpi, preprocess=preprocess,
        detail_views=detail_views, verification=verification,
    )
    # Shift the right half's local page numbers so 1..right_pages becomes
    # (left_pages+1)..(left_pages+right_pages) within this chunk.
    right_text = renumber_pages(right_text, left_pages + 1)
    return left_text.rstrip() + "\n\n" + right_text.lstrip()


def renumber_pages(chunk_text: str, chunk_start_page: int) -> str:
    """Convert '=== PAGE n ===' markers (local to the chunk) to absolute page numbers."""

    def _shift(match):
        local_n = int(match.group(1))
        return f"=== PAGE {chunk_start_page + local_n - 1} ==="

    return PAGE_MARKER_RE.sub(_shift, chunk_text or "")


# ---------- Deterministic attendee-list consistency audit (no API cost) ----------
_NUMBERED_ENTRY_RE = re.compile(r"^\s*([০-৯0-9]+)\s*[।\.\)]\s*(.+?)\s*$")
_ENTRY_TRAILING_ROLE_RE = re.compile(r"\s*-\s*\S+\s*$")
_ROLE_AFFIL_RE = re.compile(r"^\s*(ডীন|ডিন|প্রধান)\s*[,ঃ:]\s*(.+?)\s*$")
_AFFIL_ROLE_SEARCH_RE = re.compile(
    r"([^,;:।]+?)\s*(বিভাগের|অনুষদের)\s*(প্রধান|ডীন|ডিন)"
)
_TRAILING_RANK_RE = re.compile(
    r",?\s*(?:সহযোগী\s+)?(?:অধ্যাপক|প্রফেসর|প্রফেঃ)\s*$"
)
_AUDIT_HONORIFICS = {
    "অধ্যাপক", "ডঃ", "ড", "ডক্টর", "জনাব", "বাবু", "প্রফেসর", "প্রফেঃ",
    "মিস", "মিসেস",
    "dr", "mr", "mrs", "ms", "prof", "professor",
}


def _audit_person_name(entry_text: str) -> str:
    """Extract just the person's name from a numbered attendee entry."""
    text = _ENTRY_TRAILING_ROLE_RE.sub("", str(entry_text or "").strip())
    # Both English and old Bengali entries put role/affiliation after a comma
    # ("Dr. Abul Hasnat, Professor & Head of ..."); compare names only.
    text = text.split(",", 1)[0]
    text = _TRAILING_RANK_RE.sub("", text).strip(" ,;-")
    tokens = text.split()
    while tokens and tokens[0].strip(".:,").casefold() in _AUDIT_HONORIFICS:
        tokens.pop(0)
    return " ".join(tokens).strip()


def _audit_name_key(name: str) -> str:
    return re.sub(r"[.\sঃ,'\-–—]+", "", name).casefold()


def _consistency_key(role: str, affiliation: str):
    role = "ডিন" if role in ("ডীন", "ডিন") else "প্রধান"
    affil = re.sub(r"[.,ঃ:।\-]+", "", affiliation)
    affil = re.sub(r"\s+", "", affil)
    return role, affil


def ocr_consistency_report(text: str) -> list:
    """Audit numbered attendee lists for tell-tale attention-drift errors.

    Each faculty has exactly one ডীন and each department exactly one প্রধান,
    so two numbered entries carrying the SAME role + affiliation almost always
    mean the model copied a neighboring entry's line instead of reading this
    entry's own line. Purely local and deterministic — a duplicate is only
    reported, never auto-'fixed', because we cannot know which of the two
    entries is the wrong one without the source PDF.

    To avoid false positives from narrative/agenda text, an affiliation line
    is only associated with an entry when it is the entry line itself or the
    line immediately after it (the rigid shape of attendee lists).
    """
    issues = []
    seen = {}
    pending_entry = None
    section_names = {}  # name key -> entry number, reset at each section heading

    def _record(role, affil, entry_label, shown_line):
        if "〃" in affil or "[?]" in affil:
            return
        key = _consistency_key(role, affil)
        if key in seen and seen[key] != entry_label:
            issues.append(
                f"\"{shown_line.strip()}\" appears under both "
                f"\"{seen[key]}\" and \"{entry_label}\" — each অনুষদ has one ডীন "
                f"and each বিভাগ one প্রধান, so one of these lines was almost "
                f"certainly copied from a neighboring entry. Check the PDF."
            )
        else:
            seen.setdefault(key, entry_label)

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or PAGE_MARKER_RE.match(line):
            continue

        entry_match = _NUMBERED_ENTRY_RE.match(line)
        if entry_match:
            entry_number = entry_match.group(1)
            entry_text = entry_match.group(2)
            pending_entry = _ENTRY_TRAILING_ROLE_RE.sub("", entry_text)[:70]

            # Duplicate-name check WITHIN one section: the model sometimes
            # copies one entry's name over another's (attention drift). Two
            # people genuinely sharing a name also happens (CSE has two
            # মোঃ মনিরুল ইসলাম professors), so this is reported for
            # verification, never auto-'fixed'.
            person = _audit_person_name(entry_text)
            name_key = _audit_name_key(person)
            if person and len(name_key) >= 6:
                if name_key in section_names and section_names[name_key] != entry_number:
                    issues.append(
                        f"\"{person}\" appears at entries {section_names[name_key]} "
                        f"and {entry_number} of the same section — either two people "
                        f"share this name, or one line was copied over another "
                        f"entry's name. Verify both against the PDF."
                    )
                else:
                    section_names.setdefault(name_key, entry_number)

            # Old handwritten format: role+affiliation on the entry line itself
            # (e.g. "৪। ড. ইকবাল মাহমুদ, কেমিকৌশল বিভাগের প্রধান").
            inline = _AFFIL_ROLE_SEARCH_RE.search(entry_text)
            if inline:
                _record(inline.group(3), inline.group(1), pending_entry, line)
                pending_entry = None
            continue

        if pending_entry:
            role_match = _ROLE_AFFIL_RE.match(line)
            if role_match:
                _record(role_match.group(1), role_match.group(2), pending_entry, line)
                pending_entry = None
                continue

        # Any other non-empty line is a section heading or narrative text —
        # a new section starts, so the per-section name set resets.
        pending_entry = None
        section_names = {}

    return issues


_LIST_ITEM_LINE_RE = re.compile(r"^\s*[০-৯0-9]+\s*।")
_BARE_PAGE_NUMBER_RE = re.compile(r"^[০-৯0-9]{1,3}$")


def page_boundary_item_report(text: str) -> list:
    """Detect the silent-renumbering trap at page boundaries.

    When an item's number sits in a damaged page margin, the model can miss
    it, merge that item into the previous one as a 'continuation', and then
    silently renumber every later item so the list still looks sequential —
    an error that no sequence check can catch. The local signal that remains:
    a page that BEGINS with a short, complete, unnumbered paragraph followed
    by a numbered item, while the previous page was inside a numbered list.
    Legitimate long continuations (an item's text wrapping across pages) are
    skipped; short standalone paragraphs are flagged for human verification.
    """
    issues = []
    markers = list(PAGE_MARKER_RE.finditer(text or ""))
    pages = []
    for i, marker in enumerate(markers):
        start = marker.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        pages.append((int(marker.group(1)), text[start:end]))

    for index in range(1, len(pages)):
        _prev_no, prev_body = pages[index - 1]
        page_no, body = pages[index]

        # Only relevant if the previous page contained numbered list items.
        if not any(_LIST_ITEM_LINE_RE.match(l) for l in prev_body.splitlines()):
            continue

        # Collect the first paragraph, skipping blanks and a bare page number.
        first_paragraph = []
        following_line = None
        started = False
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                if started:
                    started = None  # paragraph finished; now look for follower
                continue
            if started is None:
                following_line = stripped
                break
            if not started and _BARE_PAGE_NUMBER_RE.fullmatch(stripped):
                continue  # page number printed at the top
            started = True
            first_paragraph.append(stripped)

        if not first_paragraph:
            continue
        if _LIST_ITEM_LINE_RE.match(first_paragraph[0]):
            continue  # page begins with a numbered item — normal
        if len(first_paragraph) > 4:
            continue  # long text wrap — likely a genuine continuation
        if not first_paragraph[-1].endswith(("।", ")", "।-", ":", "ঃ")):
            continue  # doesn't read as a complete standalone statement
        if not (following_line and _LIST_ITEM_LINE_RE.match(following_line)):
            continue

        issues.append(
            {
                "page": page_no,
                "message": (
                    f"Page {page_no} begins with an UNNUMBERED paragraph "
                    f"(\"{first_paragraph[0][:60]}…\") immediately followed by numbered "
                    f"item \"{following_line[:25]}…\". Check the scan's left margin: if "
                    f"that paragraph is actually its own numbered item, the model has "
                    f"silently renumbered every later item — compare the printed item "
                    f"numbers on paper against this output."
                ),
            }
        )

    return issues


def extract_single_page_pdf(pdf_bytes: bytes, page_no: int) -> bytes:
    """Return a one-page PDF containing only the given 1-based page."""
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    part = fitz.open()
    part.insert_pdf(src, from_page=page_no - 1, to_page=page_no - 1)
    data = part.tobytes(garbage=1, deflate=True)
    part.close()
    src.close()
    return data


def page_item_numbers(text: str, page_no: int) -> list:
    """List the numbered-item markers (as ints) on one page of the combined text."""
    markers = list(PAGE_MARKER_RE.finditer(text or ""))
    for i, marker in enumerate(markers):
        if int(marker.group(1)) != page_no:
            continue
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        numbers = []
        for line in text[marker.end():end].splitlines():
            m = re.match(r"^\s*([০-৯0-9]+)\s*।", line)
            if m:
                numbers.append(int(m.group(1).translate(BENGALI_TO_ARABIC_DIGITS)))
        return numbers
    return []


def replace_page_text(text: str, page_no: int, new_page_text: str) -> str:
    """Splice a freshly OCR'd page (including its marker) into the combined text."""
    markers = list(PAGE_MARKER_RE.finditer(text or ""))
    for i, marker in enumerate(markers):
        if int(marker.group(1)) != page_no:
            continue
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        tail = text[end:]
        return text[: marker.start()] + new_page_text.strip() + ("\n\n" if tail else "") + tail
    return text


# ============================================================
# STAGE 2 (TAILORED): OCR TEXT → MEETING-MINUTES JSON
# Reuses: st, time, NEW_SDK, genai_new/genai_types or genai_old,
# api_key, model_name, max_workers, and safe_rpm.
# ============================================================
# ---------- Schema matching your template exactly ----------
PRESENTEE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "nullable": True},
        "prefix": {"type": "string", "nullable": True},
        "designation": {"type": "string", "nullable": True},
        "department": {"type": "string", "nullable": True},
        "office": {"type": "string", "nullable": True},
    },
    "required": ["name", "prefix", "designation", "department", "office"],
}

AGENDA_SCHEMA = {
    "type": "object",
    "properties": {
        "serial": {"type": "integer"},
        "body": {"type": "string"},
        "resolution": {"type": "string", "nullable": True},
    },
    "required": ["serial", "body"],
}

MEETING_SCHEMA = {
    "type": "object",
    "properties": {
        "serial": {"type": "integer", "nullable": True},
        "title": {"type": "string", "nullable": True},
        "date": {"type": "string", "nullable": True},
        "type": {"type": "string", "nullable": True},
        "status": {"type": "string", "nullable": True},
        "description": {"type": "string", "nullable": True},
        "president": {"type": "string", "nullable": True},
        "conclusion": {"type": "string", "nullable": True},
        "presentees": {"type": "array", "items": PRESENTEE_SCHEMA},
        "agenda": {"type": "array", "items": AGENDA_SCHEMA},
    },
    "required": ["presentees", "agenda"],
}


# Canonical department and office values copied from the supplied SQL seed data.
# A close fuzzy match is replaced with the exact Bangla value below. A weak or
# ambiguous match is kept exactly as Gemini extracted it.
ALLOWED_DESIGNATIONS = {"অধ্যাপক", "সহযোগী অধ্যাপক"}

DEPARTMENT_REFERENCES = [{'canonical': 'পানি সম্পদ কৌশল বিভাগ',
  'variants': ['পানি সম্পদ কৌশল বিভাগ', 'Water Resources Engineering', 'ডব্লিউআরই', 'WRE']},
 {'canonical': 'নগর ও অঞ্চল পরিকল্পনা বিভাগ',
  'variants': ['নগর ও অঞ্চল পরিকল্পনা বিভাগ', 'Urban and Regional Planning', 'ইউআরপি', 'URP']},
 {'canonical': 'পেট্রোলিয়াম ও মিনারেল রিসোর্সেস প্রকৌশল বিভাগ',
  'variants': ['পেট্রোলিয়াম ও মিনারেল রিসোর্সেস প্রকৌশল বিভাগ',
               'Petroleum and Mineral Resources Engineering',
               'পিএমআরই',
               'PMRE']},
 {'canonical': 'পদার্থবিজ্ঞান বিভাগ', 'variants': ['পদার্থবিজ্ঞান বিভাগ', 'Physics', 'ফিজিক্স', 'Phy']},
 {'canonical': 'ন্যানোম্যাটেরিয়ালস এন্ড সিরামিক ইঞ্জিনিয়ারিং বিভাগ',
  'variants': ['ন্যানোম্যাটেরিয়ালস এন্ড সিরামিক ইঞ্জিনিয়ারিং বিভাগ',
               'Nanomaterials and Ceramic Engineering',
               'এনসিই',
               'NCE']},
 {'canonical': 'নৌযান ও নৌযন্ত্র কৌশল বিভাগ',
  'variants': ['নৌযান ও নৌযন্ত্র কৌশল বিভাগ', 'Naval Architecture and Marine Engineering', 'এনএএমই', 'NAME']},
 {'canonical': 'বস্তু ও ধাতব কৌশল বিভাগ',
  'variants': ['বস্তু ও ধাতব কৌশল বিভাগ', 'Materials and Metallurgical Engineering', 'এমএমই', 'MME']},
 {'canonical': 'যন্ত্রকৌশল বিভাগ', 'variants': ['যন্ত্রকৌশল বিভাগ', 'Mechanical Engineering', 'এমই', 'ME']},
 {'canonical': 'গণিত বিভাগ', 'variants': ['গণিত বিভাগ', 'Mathematics', 'ম্যাথ', 'Math']},
 {'canonical': 'পানি ও বন্যা ব্যবস্থাপনা ইনস্টিটিউট',
  'variants': ['পানি ও বন্যা ব্যবস্থাপনা ইনস্টিটিউট',
               'Institute of Water and Flood Management',
               'আইডব্লিউএফএম',
               'IWFM']},
 {'canonical': 'শিল্প ও উৎপাদন কৌশল বিভাগ',
  'variants': ['শিল্প ও উৎপাদন কৌশল বিভাগ', 'Industrial and Production Engineering', 'আইপিই', 'IPE']},
 {'canonical': 'তথ্য ও যোগাযোগ প্রযুক্তি ইনস্টিটিউট',
  'variants': ['তথ্য ও যোগাযোগ প্রযুক্তি ইনস্টিটিউট',
               'Institute of Information and Communication Technology',
               'আইআইসিটি',
               'IICT']},
 {'canonical': 'লাগসই প্রযুক্তি ইনস্টিটিউট',
  'variants': ['লাগসই প্রযুক্তি ইনস্টিটিউট', 'Institute of Appropriate Technology', 'আইএটি', 'IAT']},
 {'canonical': 'মানবিক বিভাগ', 'variants': ['মানবিক বিভাগ', 'Humanities', 'হিউম', 'Hum']},
 {'canonical': 'তড়িৎ ও ইলেক্ট্রনিক কৌশল বিভাগ',
  'variants': ['তড়িৎ ও ইলেক্ট্রনিক কৌশল বিভাগ', 'Electrical and Electronic Engineering', 'ইইই', 'EEE']},
 {'canonical': 'কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং বিভাগ',
  'variants': ['কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং বিভাগ',
               'Computer Science and Engineering',
               'সিএসই',
               'CSE']},
 {'canonical': 'রসায়ন বিভাগ', 'variants': ['রসায়ন বিভাগ', 'Chemistry', 'কেম', 'Chem']},
 {'canonical': 'কেমিকৌশল বিভাগ',
  'variants': ['কেমিকৌশল বিভাগ',
               'কেমিক্যাল বিভাগ',
               'কেমিক্যাল ইঞ্জিনিয়ারিং বিভাগ',
               'কেমিক্যাল ইঞ্জিনিয়ারিং বিভাগ',
               'কেমিকৌশল',
               'Chemical Engineering',
               'সিএইচই',
               'ChE']},
 {'canonical': 'পুরকৌশল বিভাগ', 'variants': ['পুরকৌশল বিভাগ', 'Civil Engineering', 'সিই', 'CE']},
 {'canonical': 'বায়োমেডিকেল ইঞ্জিনিয়ারিং বিভাগ',
  'variants': ['বায়োমেডিকেল ইঞ্জিনিয়ারিং বিভাগ', 'Biomedical Engineering', 'বিএমই', 'BME']},
 {'canonical': 'দুর্ঘটনা গবেষণা ইনস্টিটিউট',
  'variants': ['দুর্ঘটনা গবেষণা ইনস্টিটিউট', 'Accident Research Institute', 'এআরআই', 'ARI']},
 {'canonical': 'স্থাপত্য বিভাগ', 'variants': ['স্থাপত্য বিভাগ', 'Architecture', 'আর্চ', 'Arch']}]

OFFICE_REFERENCES = [{'canonical': 'বিভাগীয় প্রধান, পেট্রোলিয়াম ও মিনারেল রিসোর্সেস প্রকৌশল বিভাগ (পিএমআরই)',
  'variants': ['বিভাগীয় প্রধান, পেট্রোলিয়াম ও মিনারেল রিসোর্সেস প্রকৌশল বিভাগ (পিএমআরই)',
               'Department Head, Department of Petroleum & Mineral Resources Engineering (PMRE)']},
 {'canonical': 'ডিন, স্থাপত্য ও পরিকল্পনা অনুষদ',
  'variants': ['ডিন, স্থাপত্য ও পরিকল্পনা অনুষদ', 'Dean, Faculty of Architecture and Planning']},
 {'canonical': 'বিভাগীয় প্রধান, বস্তু ও ধাতব কৌশল বিভাগ (এমএমই)',
  'variants': ['বিভাগীয় প্রধান, বস্তু ও ধাতব কৌশল বিভাগ (এমএমই)',
               'Department Head, Department of Materials & Metallurgical Engineering (MME)']},
 {'canonical': 'ডিন, যন্ত্রকৌশল অনুষদ',
  'variants': ['ডিন, যন্ত্রকৌশল অনুষদ', 'Dean, Faculty of Mechanical Engineering']},
 {'canonical': 'বিভাগীয় প্রধান, স্থাপত্য বিভাগ (আর্চ)',
  'variants': ['বিভাগীয় প্রধান, স্থাপত্য বিভাগ (আর্চ)',
               'Department Head, Department of Architecture (ARCH)']},
 {'canonical': 'বিভাগীয় প্রধান, রসায়ন বিভাগ (কেম)',
  'variants': ['বিভাগীয় প্রধান, রসায়ন বিভাগ (কেম)', 'Department Head, Department of Chemistry (CHEM)']},
 {'canonical': 'ডিন, কেমিক্যাল ও ম্যাটেরিয়ালস কৌশল অনুষদ (এফসিএমই)',
  'variants': ['ডিন, কেমিক্যাল ও ম্যাটেরিয়ালস কৌশল অনুষদ (এফসিএমই)',
               'Dean, Faculty of Chemical & Materials Engineering (FCME)']},
 {'canonical': 'ডিন, পুরকৌশল অনুষদ',
  'variants': ['ডিন, পুরকৌশল অনুষদ', 'Dean, Faculty of Civil Engineering']},
 {'canonical': 'বিভাগীয় প্রধান, কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং বিভাগ (সিএসই)',
  'variants': ['বিভাগীয় প্রধান, কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং বিভাগ (সিএসই)',
               'Department Head, Department of Computer Science & Engineering (CSE)']},
 {'canonical': 'বিভাগীয় প্রধান, কেমিকৌশল বিভাগ (সিএইচই)',
  'variants': ['বিভাগীয় প্রধান, কেমিকৌশল বিভাগ (সিএইচই)',
               'Department Head, Department of Chemical Engineering (ChE)']},
 {'canonical': 'বিভাগীয় প্রধান, গণিত বিভাগ (ম্যাথ)',
  'variants': ['বিভাগীয় প্রধান, গণিত বিভাগ (ম্যাথ)', 'Department Head, Department of Mathematics (MATH)']},
 {'canonical': 'বিভাগীয় প্রধান, পানি সম্পদ কৌশল বিভাগ (ডব্লিউআরই)',
  'variants': ['বিভাগীয় প্রধান, পানি সম্পদ কৌশল বিভাগ (ডব্লিউআরই)',
               'Department Head, Department of Water Resources Engineering (WRE)']},
 {'canonical': 'ডিন, তড়িৎ ও ইলেক্ট্রনিক কৌশল অনুষদ',
  'variants': ['ডিন, তড়িৎ ও ইলেক্ট্রনিক কৌশল অনুষদ',
               'Dean, Faculty of Electrical & Electronic Engineering']},
 {'canonical': 'বিভাগীয় প্রধান, ন্যানোম্যাটেরিয়ালস এন্ড সিরামিক ইঞ্জিনিয়ারিং বিভাগ (এনসিই)',
  'variants': ['বিভাগীয় প্রধান, ন্যানোম্যাটেরিয়ালস এন্ড সিরামিক ইঞ্জিনিয়ারিং বিভাগ (এনসিই)',
               'Department Head, Department of Nanomaterials & Ceramics Engineering (NCE)']},
 {'canonical': 'বিভাগীয় প্রধান, শিল্প ও উৎপাদন কৌশল বিভাগ (আইপিই)',
  'variants': ['বিভাগীয় প্রধান, শিল্প ও উৎপাদন কৌশল বিভাগ (আইপিই)',
               'Department Head, Department of Industrial & Production Engineering (IPE)']},
 {'canonical': 'বিভাগীয় প্রধান, পুরকৌশল বিভাগ (সিই)',
  'variants': ['বিভাগীয় প্রধান, পুরকৌশল বিভাগ (সিই)',
               'Department Head, Department of Civil Engineering (CE)']},
 {'canonical': 'বিভাগীয় প্রধান, বায়োমেডিকেল ইঞ্জিনিয়ারিং বিভাগ (বিএমই)',
  'variants': ['বিভাগীয় প্রধান, বায়োমেডিকেল ইঞ্জিনিয়ারিং বিভাগ (বিএমই)',
               'Department Head, Department of Bio-Medical Engineering (BME)']},
 {'canonical': 'বিভাগীয় প্রধান, পদার্থবিজ্ঞান বিভাগ (ফিজিক্স)',
  'variants': ['বিভাগীয় প্রধান, পদার্থবিজ্ঞান বিভাগ (ফিজিক্স)',
               'Department Head, Department of Physics (Phy)']},
 {'canonical': 'বিভাগীয় প্রধান, মানবিক বিভাগ (হিউম)',
  'variants': ['বিভাগীয় প্রধান, মানবিক বিভাগ (হিউম)', 'Department Head, Department of Humanities (HUM)']},
 {'canonical': 'বিভাগীয় প্রধান, যন্ত্রকৌশল বিভাগ (এমই)',
  'variants': ['বিভাগীয় প্রধান, যন্ত্রকৌশল বিভাগ (এমই)',
               'Department Head, Department of Mechanical Engineering (ME)']},
 {'canonical': 'ডিন, বিজ্ঞান অনুষদ', 'variants': ['ডিন, বিজ্ঞান অনুষদ', 'Dean, Faculty of Science']},
 {'canonical': 'বিভাগীয় প্রধান, নৌযান ও নৌযন্ত্র কৌশল বিভাগ (এনএএমই)',
  'variants': ['বিভাগীয় প্রধান, নৌযান ও নৌযন্ত্র কৌশল বিভাগ (এনএএমই)',
               'Department Head, Department of Naval Arch. & Marine Engineering (NAME)']},
 {'canonical': 'বিভাগীয় প্রধান, নগর ও অঞ্চল পরিকল্পনা বিভাগ (ইউআরপি)',
  'variants': ['বিভাগীয় প্রধান, নগর ও অঞ্চল পরিকল্পনা বিভাগ (ইউআরপি)',
               'Department Head, Department of Urban & Regional Planning (URP)']},
 {'canonical': 'ডিন, স্নাতকোত্তর স্টাডিজ অনুষদ',
  'variants': ['ডিন, স্নাতকোত্তর স্টাডিজ অনুষদ', 'Dean, Faculty of Post Graduate Studies']},
 {'canonical': 'বিভাগীয় প্রধান, তড়িৎ ও ইলেক্ট্রনিক কৌশল বিভাগ (ইইই)',
  'variants': ['বিভাগীয় প্রধান, তড়িৎ ও ইলেক্ট্রনিক কৌশল বিভাগ (ইইই)',
               'Department Head, Department of Electrical & Electronic Engineering (EEE)']},
 {'canonical': 'উপাচার্য, বাংলাদেশ প্রকৌশল বিশ্ববিদ্যালয়',
  'variants': ['উপাচার্য, বাংলাদেশ প্রকৌশল বিশ্ববিদ্যালয়',
               'Vice Chancellor, Bangladesh University of Engineering and Technology']},
 {'canonical': 'উপ-উপাচার্য, বাংলাদেশ প্রকৌশল বিশ্ববিদ্যালয়',
  'variants': ['উপ-উপাচার্য, বাংলাদেশ প্রকৌশল বিশ্ববিদ্যালয়',
               'Pro-Vice Chancellor, Bangladesh University of Engineering and Technology']}]


def _entity_norm(value) -> str:
    """Normalize spelling/punctuation only for comparison, not for output."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    # OCR frequently adds/drops চন্দ্রবিন্দু (খান vs খাঁন). Ignore it for
    # comparison only — outputs keep their original spelling.
    text = text.replace("\u0981", "")
    replacements = {
        "&": " এন্ড ",
        "অ্যান্ড": "এন্ড",
        "এণ্ড": "এন্ড",
        "ডীন": "ডিন",
        "বিভাগীয়": "বিভাগীয়",
        "সায়েন্স": "সায়েন্স",
        "ম্যাটেরিয়ালস": "ম্যাটেরিয়ালস",
        "ইঞ্জিনিয়ারিং": "ইঞ্জিনিয়ারিং",
        "ইলেকট্রনিক": "ইলেক্ট্রনিক",
        "উপ উপাচার্য": "উপ-উপাচার্য",
        # Printed abbreviations used in the minutes — expanded for COMPARISON
        # only, so "ত.ই কৌশল বিভাগ" and "তড়িৎ ও ইলেক্ট্রনিক কৌশল বিভাগ"
        # canonicalize to the same SQL seed value. Keyed on the dotted/visarga
        # forms so ordinary words can never be corrupted.
        "ত.ই": "তড়িৎ ও ইলেক্ট্রনিক",
        "ত. ই": "তড়িৎ ও ইলেক্ট্রনিক",
        "ইলেকঃ": "ইলেক্ট্রনিক",
        "ইঞ্জিঃ": "ইঞ্জিনিয়ারিং",
        "মেডিক্যাল": "মেডিকেল",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[()\[\]{},.;:।/\\_|]+", " ", text)
    text = re.sub(r"[-–—]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _expanded_variants(reference: dict, entity_type: str) -> list:
    """Generate useful short variants without changing the canonical output."""
    variants = list(reference.get("variants") or [])
    canonical = reference["canonical"]

    if entity_type == "department":
        for value in list(variants):
            normalized = _entity_norm(value)
            for suffix in (" বিভাগ", " department"):
                if normalized.endswith(suffix):
                    variants.append(normalized[: -len(suffix)].strip())

    if entity_type == "office":
        no_parentheses = re.sub(r"\s*\([^)]*\)\s*", " ", canonical).strip()
        variants.append(no_parentheses)

        if canonical.startswith("বিভাগীয় প্রধান"):
            variants.append(canonical.replace("বিভাগীয় প্রধান", "প্রধান", 1))
            variants.append(no_parentheses.replace("বিভাগীয় প্রধান", "প্রধান", 1))

            alias_match = re.search(r"\(([^()]*)\)\s*$", canonical)
            if alias_match:
                alias = alias_match.group(1).strip()
                variants.extend(
                    [
                        f"বিভাগীয় প্রধান {alias}",
                        f"প্রধান {alias}",
                        f"{alias} বিভাগীয় প্রধান",
                    ]
                )

        if canonical.startswith("উপাচার্য,"):
            variants.append("উপাচার্য")
        elif canonical.startswith("উপ-উপাচার্য,"):
            variants.extend(["উপ-উপাচার্য", "উপ উপাচার্য"])

    # Preserve order while removing duplicates.
    unique = []
    seen = set()
    for value in variants:
        normalized = _entity_norm(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(value)
    return unique


def _match_score(source: str, candidate: str) -> float:
    a = _entity_norm(source)
    b = _entity_norm(candidate)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    sequence_score = SequenceMatcher(None, a, b).ratio()

    a_tokens = set(a.split())
    b_tokens = set(b.split())
    common = len(a_tokens & b_tokens)
    token_f1 = (
        2.0 * common / (len(a_tokens) + len(b_tokens))
        if a_tokens and b_tokens
        else 0.0
    )
    token_coverage = (
        common / min(len(a_tokens), len(b_tokens))
        if a_tokens and b_tokens
        else 0.0
    )
    token_score = 0.55 * token_coverage + 0.45 * token_f1
    # Every word of the input appearing in the candidate is a strong signal
    # ("Electrical Engineering" ⊆ "Electrical and Electronic Engineering");
    # score it decisively so it clears the ambiguity margin over lookalikes.
    # The ambiguity guard still protects genuinely shared subsets.
    if a_tokens and common == len(a_tokens):
        token_score = max(token_score, 0.90)

    containment_score = 0.0
    if a in b or b in a:
        length_ratio = min(len(a), len(b)) / max(len(a), len(b))
        containment_score = 0.78 + 0.22 * length_ratio

    return max(sequence_score, token_score, containment_score)


def _canonical_reference(
    value,
    references: list,
    entity_type: str,
    threshold: float,
    ambiguity_margin: float = 0.06,
):
    """Use the SQL value only for a strong, reasonably unambiguous match."""
    if value is None:
        return None
    original = str(value).strip()
    if not original or original.lower() == "null":
        return None

    ranked = []
    for reference in references:
        score = max(
            (_match_score(original, variant)
             for variant in _expanded_variants(reference, entity_type)),
            default=0.0,
        )
        ranked.append((score, reference["canonical"]))

    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best_value = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0

    is_exact = best_score >= 0.999
    is_clear = best_score - second_score >= ambiguity_margin
    if best_score >= threshold and (is_exact or is_clear):
        return best_value

    # Very different or ambiguous: preserve the extracted text.
    return original


# Meeting-role words that must never leak into a person's identity fields.
MEETING_ROLE_WORDS = {
    "সভাপতি", "সদস্য", "সদস্য-সচিব", "সদস্য সচিব",
    "আমন্ত্রিত", "আমন্ত্রিত অতিথি", "চেয়ারম্যান",
    "member", "chairman", "in the chair",
}


def _strip_meeting_role_words(value):
    """Remove a trailing/leading meeting-role label from an identity field."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if _entity_norm(text) in {_entity_norm(w) for w in MEETING_ROLE_WORDS}:
        return None
    for word in MEETING_ROLE_WORDS:
        text = re.sub(
            rf"(?:^|\s)[\(\[]?{re.escape(word)}[\)\]]?\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip(" ,;-–—।")
    return text.strip() or None


_PROFESSOR_WORDS = (
    "অধ্যাপক", "প্রফে", "প্রোফে", "prof",  # stems cover প্রফেসর, প্রফেঃ, প্রফে:, Professor, Prof.
)
_ASSOCIATE_WORDS = ("সহযোগী", "associate", "assoc")
_ASSISTANT_WORDS = ("সহকারী", "assistant", "asst")


def _mentions_professor(text: str) -> bool:
    return any(word in text for word in _PROFESSOR_WORDS)


def _canonical_designation(value, prefix=None):
    """Return only অধ্যাপক / সহযোগী অধ্যাপক; everything else becomes null.

    All professor spellings are treated as অধ্যাপক: প্রফেসর, প্রোফেসর, the old
    handwritten abbreviation প্রফেঃ, and English Professor / Prof. Combined
    with a সহযোগী/Associate marker they become সহযোগী অধ্যাপক. Any সহকারী /
    Assistant rank is rejected (null), per the schema rules.
    """
    raw = _entity_norm(value).lower()

    if raw:
        if any(word in raw for word in _ASSISTANT_WORDS):
            return None
        if _mentions_professor(raw):
            if any(word in raw for word in _ASSOCIATE_WORDS):
                return "সহযোগী অধ্যাপক"
            return "অধ্যাপক"
        return None

    # When designation is absent, infer only from an unambiguous prefix.
    prefix_norm = _entity_norm(prefix).lower()
    if not prefix_norm:
        return None
    if any(word in prefix_norm for word in _ASSISTANT_WORDS):
        return None
    if _mentions_professor(prefix_norm):
        if any(word in prefix_norm for word in _ASSOCIATE_WORDS):
            return "সহযোগী অধ্যাপক"
        return "অধ্যাপক"
    return None


GENERIC_DEPARTMENT_HEAD_OFFICES = {
    "বিভাগীয় প্রধান",
    "বিভাগীয় প্রধান",
    "বিভাগ প্রধান",
    "department head",
    "head of department",
}


def _build_department_head_maps():
    """Two exact maps built from the SQL seed data:

    1. canonical-department-norm  -> seeded department-head office
    2. seeded department-head office -> canonical department
    """
    department_norms = {
        _entity_norm(reference["canonical"]): reference["canonical"]
        for reference in DEPARTMENT_REFERENCES
    }
    office_by_department = {}
    department_by_office = {}

    for reference in OFFICE_REFERENCES:
        canonical_office = reference["canonical"]
        if not canonical_office.startswith("বিভাগীয় প্রধান,"):
            continue

        office_department = canonical_office.split(",", 1)[1].strip()
        office_department = re.sub(r"\s*\([^)]*\)\s*$", "", office_department).strip()
        office_department_norm = _entity_norm(office_department)

        canonical_department = department_norms.get(office_department_norm)
        if canonical_department:
            office_by_department[office_department_norm] = canonical_office
            department_by_office[canonical_office] = canonical_department

    return office_by_department, department_by_office


DEPARTMENT_HEAD_OFFICE_BY_DEPARTMENT, DEPARTMENT_BY_HEAD_OFFICE = (
    _build_department_head_maps()
)
CANONICAL_DEPARTMENTS = {r["canonical"] for r in DEPARTMENT_REFERENCES}
_HEAD_PREFIX_FULL = _entity_norm("বিভাগীয় প্রধান")
_HEAD_PREFIX_SHORT = _entity_norm("প্রধান")


def _department_embedded_in_office(office_value):
    """Pull the department text out of a 'বিভাগীয় প্রধান, X' style office.

    Comparison happens on normalized text, so composed/decomposed Bengali
    forms and punctuation differences never break the match. Returns the
    normalized department candidate or None.
    """
    if not office_value:
        return None
    norm = _entity_norm(office_value)
    if not norm:
        return None
    if norm.startswith(_HEAD_PREFIX_FULL + " "):
        candidate = norm[len(_HEAD_PREFIX_FULL):].strip()
    elif norm.startswith(_HEAD_PREFIX_SHORT + " "):
        candidate = norm[len(_HEAD_PREFIX_SHORT):].strip()
    else:
        return None
    return candidate or None


def _expand_department_head_office(office, department):
    """Add the department name to a generic department-head office.

    Rules:
    1. If the department closely matches a seeded SQL department, use the
       exact seeded office value, including its SQL abbreviation.
    2. If the department is too different from every SQL department, preserve
       the extracted department and build the office from that extracted text.
    3. If no department is available, keep the generic office unchanged.

    Examples:
        office="বিভাগীয় প্রধান", department="রসায়ন বিভাগ"
        -> "বিভাগীয় প্রধান, রসায়ন বিভাগ (কেম)"

        office="বিভাগীয় প্রধান", department="ধাতু কৌশল বিভাগ"
        -> "বিভাগীয় প্রধান, ধাতু কৌশল বিভাগ"
    """
    if office is None:
        return None

    original_office = str(office).strip()
    if not original_office:
        return None

    generic_norms = {_entity_norm(value) for value in GENERIC_DEPARTMENT_HEAD_OFFICES}
    if _entity_norm(original_office) not in generic_norms:
        return original_office

    if department is None:
        return original_office

    department_text = str(department).strip()
    department_norm = _entity_norm(department_text)
    if not department_norm:
        return original_office

    seeded_office = DEPARTMENT_HEAD_OFFICE_BY_DEPARTMENT.get(department_norm)
    if seeded_office:
        return seeded_office

    # The department was intentionally preserved because it was too different
    # from the SQL list. Keep that extracted department in the office as well.
    return f"বিভাগীয় প্রধান, {department_text}"


# ---------- Optional faculty name roster (deterministic name correction) ----------
_HONORIFIC_TOKENS = {
    "অধ্যাপক", "সহযোগী", "সহকারী", "ডঃ", "ড", "ডক্টর", "প্রফেসর", "প্রফেঃ", "প্রফে",
    "জনাব", "বাবু", "মিস", "মিসেস", "ইঞ্জিঃ", "ইঞ্জি", "ইঞ্জিনিয়ার",
    "dr", "prof", "professor", "mr", "mrs", "ms", "engr",
}


def _strip_name_honorifics(name: str) -> str:
    """Remove leading honorific/rank tokens; keep মোঃ etc. (part of the name)."""
    tokens = name.split()
    while tokens:
        stripped = tokens[0].strip(".:,").casefold()
        if stripped in _HONORIFIC_TOKENS:
            tokens.pop(0)
        else:
            break
    return " ".join(tokens).strip()


def parse_name_roster(raw: str) -> list:
    """Parse a faculty roster from plain lines or SQL INSERT statements.

    Accepts one name per line, or SQL where names appear as quoted strings.
    Honorifics are stripped, non-Bengali junk is skipped, order is preserved,
    duplicates are dropped.
    """
    if not raw or not raw.strip():
        return []
    text = raw.strip()

    quoted = re.findall(r"'((?:[^'\\]|\\.)*)'", text)
    candidates = quoted if quoted else text.splitlines()

    names, seen = [], set()
    for candidate in candidates:
        candidate = candidate.replace("\\'", "'").strip().strip(",;|")
        if not candidate or not re.search(r"[\u0980-\u09FF]", candidate):
            continue
        candidate = _strip_name_honorifics(candidate)
        if len(candidate) < 3:
            continue
        key = _entity_norm(candidate)
        if key and key not in seen:
            seen.add(key)
            names.append(candidate)
    return names


def build_roster_pairs(names: list) -> list:
    """Precompute (normalized, original) pairs for matching."""
    return [(_entity_norm(n), n) for n in (names or []) if _entity_norm(n)]


def _canonical_name(value, roster_pairs, threshold: float = 0.72):
    """Return the exact roster spelling for a close OCR'd name, else None.

    Safety properties: a very different name is never replaced (threshold),
    and when two DIFFERENT roster people score nearly identically the match
    is treated as ambiguous and skipped — replacing a name with the wrong
    person would be worse than leaving the OCR text as-is.
    """
    raw = _entity_norm(value)
    if not raw or not roster_pairs:
        return None

    scored = sorted(
        ((_match_score(raw, norm), original) for norm, original in roster_pairs),
        key=lambda item: -item[0],
    )
    best_score, best_name = scored[0]
    if best_score < threshold:
        return None
    if (
        len(scored) > 1
        and scored[1][1] != best_name
        and best_score - scored[1][0] < 0.04
        and best_score < 0.999
    ):
        return None  # two different people are equally close — ambiguous
    return best_name


def normalize_meeting_entities(meeting: dict, name_roster=None) -> dict:
    """Apply designation, department, and context-aware office normalization.

    Bulletproofing added here:
    - Meeting-role words (সভাপতি/সদস্য/...) are stripped from identity fields.
    - A missing department is backfilled from the person's office whenever the
      office is (or matches) a seeded 'বিভাগীয় প্রধান, X' entry, or when the
      office text itself embeds a department name that strongly matches the
      SQL seed list. Preserved (intentionally different) values are never
      overwritten.
    - When a faculty name roster is supplied, each attendee name within close
      edit distance of exactly one roster entry is replaced by that exact
      spelling; the applied corrections are recorded in '_name_corrections'
      for display (the caller pops that key before saving JSON).
    """
    normalized = dict(meeting or {})
    normalized_people = []
    seen = set()
    roster_pairs = build_roster_pairs(name_roster)
    name_corrections = []

    for person in normalized.get("presentees") or []:
        clean_person = dict(person or {})

        for field in ("prefix", "name", "department", "office"):
            clean_person[field] = _strip_meeting_role_words(clean_person.get(field))

        if clean_person.get("name") and roster_pairs:
            fixed = _canonical_name(clean_person["name"], roster_pairs)
            if fixed and fixed != clean_person["name"]:
                name_corrections.append(f'{clean_person["name"]} → {fixed}')
                clean_person["name"] = fixed

        clean_person["designation"] = _canonical_designation(
            clean_person.get("designation"),
            clean_person.get("prefix"),
        )

        canonical_department = _canonical_reference(
            clean_person.get("department"),
            DEPARTMENT_REFERENCES,
            "department",
            threshold=0.72,
        )
        canonical_office = _canonical_reference(
            clean_person.get("office"),
            OFFICE_REFERENCES,
            "office",
            threshold=0.74,
        )

        # A bare office such as "বিভাগীয় প্রধান" is ambiguous by itself.
        # Use the already-normalized department to select the exact seeded
        # office, including the department name and abbreviation.
        canonical_office = _expand_department_head_office(
            canonical_office,
            canonical_department,
        )

        # Backfill a MISSING department from the office (never overwrite a
        # present one — intentionally preserved values stay untouched).
        if canonical_department in (None, ""):
            if canonical_office in DEPARTMENT_BY_HEAD_OFFICE:
                canonical_department = DEPARTMENT_BY_HEAD_OFFICE[canonical_office]
            else:
                embedded = _department_embedded_in_office(canonical_office)
                if embedded:
                    inferred = _canonical_reference(
                        embedded,
                        DEPARTMENT_REFERENCES,
                        "department",
                        threshold=0.75,
                    )
                    if inferred in CANONICAL_DEPARTMENTS:
                        canonical_department = inferred

        clean_person["department"] = canonical_department
        clean_person["office"] = canonical_office

        # Drop rows that carry no identifying information at all.
        if not any(
            clean_person.get(f)
            for f in ("name", "office", "department", "designation")
        ):
            continue

        key = (
            _entity_norm(clean_person.get("name")),
            _entity_norm(clean_person.get("department")),
            _entity_norm(clean_person.get("office")),
        )
        if key in seen:
            continue
        seen.add(key)
        normalized_people.append(clean_person)

    normalized["presentees"] = normalized_people
    normalized.setdefault("agenda", [])
    if name_corrections:
        normalized["_name_corrections"] = name_corrections
    return normalized

EXTRACTION_RULES = """You are extracting structured data from the minutes of an academic council meeting (একাডেমিক কাউন্সিলের সভা / অধিবেশন) of a Bangladeshi university (BUET, formerly EPUET). The text is OCR output in Bengali or in English — EPUET-era minutes (1960s) are entirely in ENGLISH with headings like "MEMBERS PRESENT:" and "RESOLUTIONS:". It may be a modern typed document or the transcription of an old handwritten or typewritten one. Apply the SAME schema to both languages; copy text in its original language, never translate. Extract into JSON with these rules:

- serial: the meeting number as an integer (e.g. ৪৬০তম সভা → 460). Convert Bengali numerals to Arabic. Old minutes may state no number — then use null.
- title: e.g. "একাডেমিক কাউন্সিলের ৪৬০-তম সভা". For old minutes without a number, use the heading as written (e.g. "একাডেমিক কাউন্সিল অধিবেশনের কার্যবিবরণী" or "Proceedings of the Meeting of the Academic Council held on 14.1.69").
- date: meeting date+time in ISO 8601 with +06:00 timezone, e.g. "2021-02-28T16:30:00+06:00". Convert Bengali dates/numerals. If time is unknown use T00:00:00+06:00. Old documents may write DD-MM-YY with a two-digit year, Bengali (১২-৭-৭৪ → 1974-07-12) or English dotted (14.1.69 → 1969-01-14; the order is DAY.MONTH.YEAR): years 00–30 mean 20xx, otherwise 19xx. If the minutes cover two sitting dates (e.g. ১২-৭-৭৪ ও ১৬-৭-৭৪), use the FIRST date.
- type: "academic". status: "past".
- description: the opening narrative of the minutes (who presided, platform, welcome, condolence resolutions, etc.) verbatim, EXCLUDING the attendee list and agenda items. English minutes often have only the heading line — then use that or null.
- president: full line identifying the meeting president, e.g. "অধ্যাপক ডঃ সত্য প্রসাদ মজুমদার, উপাচার্য, বাংলাদেশ প্রকৌশল বিশ্ববিদ্যালয়, ঢাকা।". In old minutes the presiding officer may be listed as ভাইস চ্যান্সেলার / চেয়ারম্যান at the top of the attendee list, or in English as "... Vice-Chancellor in the Chair" — use that entry's full line.
- conclusion: the closing paragraph or approval block verbatim (পরিশেষে... / অনুমোদিত; English: the "Approved / Sd/- ... Vice-Chancellor" block with the Registrar signature).
- presentees: EVERY attendee from the presence list (উপস্থিত সদস্যবৃন্দ / "MEMBERS PRESENT:"), in document order. Do not skip, invent, or merge any attendee. Split each entry into:
  - prefix: honorific as written ("অধ্যাপক ডঃ", "ডঃ", "ড.", "জনাব", "বাবু", "প্রফেসর", "Dr.", "Mr.", "Mrs.", ...)
  - name: name only, without prefix. Copy it character for character from the OCR text. Never normalize, complete, shorten, re-spell or “correct” a name, and never replace it with a more familiar one. If the OCR text has [?] inside a name, keep the [?] exactly where it is — it marks a character a human must verify.
  - designation: the ONLY permitted non-null output values are exactly "অধ্যাপক" and "সহযোগী অধ্যাপক" — in BOTH languages. ALL professor spellings mean অধ্যাপক: প্রফেসর, প্রোফেসর, the abbreviation প্রফেঃ, English Professor / Prof. → output "অধ্যাপক". সহযোগী প্রফেসর / সহযোগী প্রফেঃ / সহযোগী অধ্যাপক / Associate Professor → output "সহযোগী অধ্যাপক". Any সহকারী / Assistant / Asstt. rank or anything else → null. You may infer "অধ্যাপক" from an unambiguous prefix such as "অধ্যাপক ডঃ" or "প্রফেসর".
  - If an entry's detail column is a rank plus a department (e.g. "প্রফেসর, যন্ত্রকৌশল বিভাগ", "সহযোগী প্রফেঃ, তড়িৎ কৌশল", or English "Professor of Civil Engineering", "Associate Professor of Electrical Engg."), that is a designation + department, NOT an office: set designation per the rule above, department to the department text, and office to null.
  - department: extract the department/institute as written if stated, else null. Local code will replace a close match with the exact canonical department name from the supplied SQL list; a very different or ambiguous value will remain unchanged.
  - office: extract the administrative office as written if stated (উপাচার্য, ভাইস চ্যান্সেলার, ডিন/ডীন, বিভাগীয় প্রধান, রেজিস্ট্রার, পরিচালক; English: Vice-Chancellor, Dean of the Faculty of ..., Head of ... Department, Director of Students' Welfare, Registrar), else null. If the office is only "বিভাগীয় প্রধান" and a department is present, local code will combine both and use the exact seeded office value, including the department name and abbreviation. Other close office matches are replaced with the exact SQL value; very different values remain unchanged.
  - If the office text itself names a department, faculty, or institute (e.g. "বিভাগীয় প্রধান, পুরকৌশল বিভাগ", "কেমিকৌশল বিভাগের প্রধান", or English "Head of the Department of Chemical Engineering", "Dean of the Faculty of Engineering & Head of the Department of Chemical Engineering"), fill office with the FULL text as written AND also fill department with that department name.
  - A ditto mark in an attendee row — 〃, ", or English "-do-" — means "same value as the row above". Resolve it: fill the field with the value from the entry directly above (chaining through consecutive ditto rows). Never output 〃 or -do- itself in any field.
  - Meeting-role words printed beside an attendee — সভাপতি, সদস্য, সদস্য-সচিব, চেয়ারম্যান, আমন্ত্রিত অতিথি; English: Member, Chairman, "in the Chair" — describe the person's role in THE MEETING only. NEVER copy them into prefix, name, designation, department, or office (e.g. "Vice-Chancellor in the Chair" → office is "Vice-Chancellor").
  - Each numbered entry in the attendee list is exactly ONE person. Never merge two adjacent entries, and never move a field (name, designation, department, office) from one entry into another.
  - If an entry lists only an office with no name, set name/prefix/designation/department to null and fill office.
- agenda: every proposal (প্রস্তাব নং ...) with:
  - serial: sequential integer position of the proposal (1, 2, 3, ...)
  - body: full proposal text verbatim, starting with "প্রস্তাব নং ...". Preserve line breaks as \n.
    Preserve every table as a Markdown pipe table: one header row, one separator row,
    and every data row. Never flatten a table into ordinary paragraph text. Preserve
    the EXACT number and position of columns, including blank cells. A blank printed
    top-left header is a real cell and MUST be written explicitly, e.g.
    | | 150 এর মধ্যে প্রাপ্ত নম্বর | শতকরা প্রাপ্ত নম্বর | লেটার গ্রেড | গ্রেড পয়েন্ট |
    The separator and every row must contain that same number of cells. Never delete
    the first row-label column, never shift values left, and never omit a final value
    such as Grade Point. If a printed header spans multiple columns, put its text in
    the leftmost covered cell and leave the remaining covered header cells blank.
    A ONE-COLUMN table is still a table: write its header, separator, and every value
    on separate lines, each with leading and trailing pipes (for example | Header |,
    | --- |, | Value |). Never turn a one-column table into an inline pipe sequence.
  - resolution: full "সিদ্ধান্ত : ..." text verbatim. If the resolution is missing in this text portion, use null. Preserve any table in the resolution as a Markdown pipe table too.
  - ONE PROPOSAL = ONE AGENDA ENTRY, EVEN ACROSS PAGES: a proposal begins at a "প্রস্তাব নং ..." line and continues until the NEXT "প্রস্তাব নং ..." line. Everything in between belongs to that same entry: continuation paragraphs, tables and table rows that carry on over a page break, repeated table headers, the section/department/faculty headings that label those tables (e.g. "স্থাপত্য বিভাগ", "পুরকৌশল বিভাগ", "আই.পি.ই বিভাগ", "যন্ত্রকৌশল বিভাগ"), lists of names, roll numbers or course codes, and that item's own "সিদ্ধান্ত ঃ" text. NEVER begin a new agenda entry merely because a new page starts, a new table starts, or a new heading appears. If a page begins with a table, a table header row, a heading, or any text that is not itself a "প্রস্তাব নং ..." line, it is a CONTINUATION: append it to the body of the proposal already in progress, keeping every table as a Markdown pipe table. In this format an agenda entry whose body does not begin with "প্রস্তাব নং" is always a mistake.
  - OLD FORMAT: older minutes have no প্রস্তাব নং items; instead a সিদ্ধান্তাবলী (decisions) section — or in English a "RESOLUTIONS:" section — lists numbered items (১।, ২।, ... / 1., 2., ...). Treat each numbered item as one agenda entry: body = the item's full text verbatim. If the item text itself states the decision (…সিদ্ধান্ত গ্রহণ করা হয়, …অনুমোদন করা হয়, …কনফার্ম করা হয়; English: "Confirmed ...", "... and resolved that ...", "Considered and approved ..."), also copy that deciding sentence (or the whole item if it is one sentence) into resolution; otherwise resolution = null. This next exception applies ONLY to that old format — a document that contains no "প্রস্তাব নং" item anywhere: if a page BEGINS with a short, complete, standalone decision paragraph that carries no number (its number may have been lost in a damaged margin) and is not a continuation of the previous item, treat it as its OWN agenda entry. It NEVER applies to a document that uses প্রস্তাব নং, and it never applies to a table, a table header row, or a section/department heading.
- Copy all Bengali text EXACTLY as written (do not modernize spelling, do not translate, do not transliterate). Fix only obvious OCR artifacts like stray Latin/Arabic/Devanagari characters inside Bengali words when the correct Bengali word is unambiguous — but NEVER apply this to a PERSON'S NAME. A name has no “correct” form other than the one printed, so pass every name through completely unchanged, character for character, even when it looks misspelled, unusual, or like a familiar name with one letter wrong. These names are written straight into a permanent database that identifies real people and is never re-checked against the document, so silently “improving” one is the most damaging thing you can do here. Keep [?] illegible-word placeholders exactly where the OCR placed them.
- Strip page markers like '=== PAGE n ===' and page headers/footers/page numbers from all extracted text.
- NEVER invent values. If a field is not present in this text portion, use null (or [] for lists)."""


def split_text_with_overlap(full_text: str, max_chars: int = JSON_CHUNK_CHARS):
    """Bound every request, preferring page/paragraph breaks with limited overlap."""
    if max_chars < 256:
        raise ValueError("Text chunk size must be at least 256 characters")
    if not full_text:
        return [full_text]
    chunks, start = [], 0
    overlap = min(1800, max_chars // 8)
    while start < len(full_text):
        end = min(start + max_chars, len(full_text))
        if end < len(full_text):
            lower = start + max_chars // 2
            page_breaks = [m.start() for m in PAGE_MARKER_RE.finditer(full_text, lower, end)]
            boundary = page_breaks[-1] if page_breaks else full_text.rfind("\n\n", lower, end)
            if boundary <= lower:
                boundary = full_text.rfind("\n", lower, end)
            if boundary > lower:
                end = boundary
        chunks.append(full_text[start:end])
        if end == len(full_text):
            break
        next_start = max(start + 1, end - overlap)
        newline = full_text.find("\n", next_start, end)
        if newline >= 0:
            next_start = newline + 1
        start = next_start
    return chunks


def _deep_clean_strings(obj):
    """Apply the deterministic Bengali cleanup to every string in a JSON tree."""
    if isinstance(obj, str):
        return clean_bengali_ocr_text(obj)
    if isinstance(obj, list):
        return [_deep_clean_strings(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _deep_clean_strings(value) for key, value in obj.items()}
    return obj


def _validated_meeting_partial(parsed):
    """Structurally validate one JSON chunk BEFORE caching it.

    Raises ValueError on any structural problem so the caller retries the
    request instead of silently caching a broken partial.
    """
    if not isinstance(parsed, dict):
        raise ValueError("model returned non-object JSON")

    presentees = parsed.get("presentees")
    if presentees is None:
        parsed["presentees"] = []
    elif not isinstance(presentees, list):
        raise ValueError("'presentees' is not a list")

    agenda = parsed.get("agenda")
    if agenda is None:
        parsed["agenda"] = []
    elif not isinstance(agenda, list):
        raise ValueError("'agenda' is not a list")

    for field in ("title", "date", "type", "status", "description", "president", "conclusion"):
        if parsed.get(field) is not None and not isinstance(parsed[field], str):
            raise ValueError(f"'{field}' must be text or null")
    for person in parsed["presentees"]:
        if not isinstance(person, dict):
            raise ValueError("a presentee entry is not an object")
        for field in ("name", "prefix", "designation", "department", "office"):
            if person.get(field) is not None and not isinstance(person[field], str):
                raise ValueError(f"presentee '{field}' must be text or null")

    for item in parsed["agenda"]:
        if not isinstance(item, dict):
            raise ValueError("an agenda entry is not an object")
        body = item.get("body")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("an agenda entry is missing its 'body'")
        if item.get("resolution") is not None and not isinstance(item["resolution"], str):
            raise ValueError("agenda resolution must be text or null")
        serial = item.get("serial")
        if isinstance(serial, str):
            digits = re.sub(
                r"\D", "", serial.translate(BENGALI_TO_ARABIC_DIGITS)
            )
            item["serial"] = int(digits) if digits else None

    if not any(parsed.get(f) for f in ("title", "date", "description", "president", "conclusion", "presentees", "agenda")):
        raise ValueError("model returned an empty meeting object")
    return _deep_clean_strings(parsed)


def gemini_extract_meeting(
    chunk_text: str,
    part_no: int,
    total_parts: int,
    key_pool: APIKeyPool,
    model_name: str,
) -> dict:
    prompt = (
        f"{EXTRACTION_RULES}\n\n"
        f"This is part {part_no} of {total_parts} of the document. "
        "Extract only what appears in this part.\n\n"
        f"--- DOCUMENT TEXT ---\n{chunk_text}"
    )
    last_error = None

    attempt = 0
    while attempt < JSON_ATTEMPTS:
        api_key = key_pool.current_key()
        key_pool.wait(api_key)
        try:
            if NEW_SDK:
                response = get_new_client(api_key).models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=new_sdk_config(
                        _model_name=model_name,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=MEETING_SCHEMA,
                        _thinking_level=resolve_thinking_level(
                            model_name, JSON_THINKING_LEVEL
                        ),
                    ),
                )
                raw = response.text or ""
            else:
                response = legacy_generate(
                    api_key, model_name, prompt,
                    generation_config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                        "response_schema": MEETING_SCHEMA,
                    },
                )
                raw = response.text or ""

            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            if _response_truncated(response):
                raise JSONIncompleteError("JSON output was truncated")
            if not raw:
                raise ValueError("empty JSON response")
            result = _validated_meeting_partial(json.loads(raw))
            key_pool.record_success(api_key)
            return result

        except JSONIncompleteError:
            raise
        except (json.JSONDecodeError, ValueError) as e:
            # Invalid or truncated JSON → retry; lower the JSON-chunk slider
            # if a chunk keeps failing this way.
            last_error = e
            attempt += 1
            if attempt < JSON_ATTEMPTS:
                time.sleep(retry_delay(e, attempt - 1))
        except Exception as e:
            last_error = e
            msg = str(e).upper()
            if "RESPONSE_SCHEMA" in msg or ("SCHEMA" in msg and "INVALID" in msg):
                raise RuntimeError(
                    f"response_schema not supported by {model_name}: {e}"
                ) from e
            # A key swap is not a retry — see the OCR loop.
            if key_pool.report_error(api_key, e):
                continue
            if _is_fatal_api_error(e):
                raise
            attempt += 1
            if attempt < JSON_ATTEMPTS:
                time.sleep(retry_delay(e, attempt - 1))

    raise last_error


def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def merge_meeting_partials(partials: list) -> dict:
    """Structure-aware merge for this template — no extra API call needed."""
    final = {
        "serial": None, "title": None, "date": None,
        "type": "academic", "status": "past",
        "description": None, "president": None, "conclusion": None,
        "presentees": [], "agenda": [],
    }
    # Scalars: first non-null wins (except conclusion — usually in the LAST chunk).
    for p in partials:
        for key in ("serial", "title", "date", "description", "president"):
            if final[key] in (None, "") and p.get(key) not in (None, ""):
                final[key] = p[key]
    for p in reversed(partials):
        if p.get("conclusion"):
            final["conclusion"] = p["conclusion"]
            break

    # Presentees: keep order, dedupe by (name, office).
    seen = set()
    for p in partials:
        for person in p.get("presentees") or []:
            key = (_norm(person.get("name")), _norm(person.get("office")))
            if key in seen:
                continue
            seen.add(key)
            final["presentees"].append(person)

    # Agenda: merge by proposal number found in the body (or serial as fallback).
    def proposal_key(item):
        # Merge on the printed proposal number, normalized to Arabic digits so
        # "প্রস্তাব নং এ ১৪০১০৬৫" and "1401065" are the same item. Without a
        # number, fall back to the item's own opening words — NOT its serial,
        # because chunk-local serials collide across chunks and would merge two
        # unrelated old-format items into one.
        text = (item.get("body") or "")[:300]
        match = re.search(r"প্রস্তাব\s*নং[^০-৯0-9]{0,15}([০-৯0-9]+)", text)
        if match:
            return "prop-" + match.group(1).translate(BENGALI_TO_ARABIC_DIGITS)
        opening = re.sub(r"\s+", " ", text).strip()[:60]
        return "text-" + (opening or f"serial-{item.get('serial')}")

    merged = {}
    order = []
    for p in partials:
        for item in p.get("agenda") or []:
            k = proposal_key(item)
            if k not in merged:
                merged[k] = dict(item)
                order.append(k)
            else:
                ex = merged[k]
                # Overlapping chunks may each hold a partial version — keep the longer text.
                if len(item.get("body") or "") > len(ex.get("body") or ""):
                    ex["body"] = item["body"]
                if len(item.get("resolution") or "") > len(ex.get("resolution") or ""):
                    ex["resolution"] = item["resolution"]
    final["agenda"] = [merged[k] for k in order]
    for i, item in enumerate(final["agenda"], 1):
        item["serial"] = i  # renumber sequentially after merge
    return final


_PROPOSAL_HEAD_RE = re.compile(r"প্রস্তাব\s*নং")

# Headings that legitimately open their own entry without a proposal number —
# the "any other business" block at the end of the minutes. Everything else
# without a proposal number is a continuation of the item above it.
_STANDALONE_SECTION_RE = re.compile(
    r"^\s*(?:<p>)?\s*(?:বিবিধ|অন্যান্য|বিবিধ\s+বিষয়|Miscellaneous|"
    r"Any\s+Other\s+Business)\s*[ঃ:।\-]"
)


def _agenda_uses_proposal_numbers(agenda: list) -> bool:
    """True for the modern format where every item starts with প্রস্তাব নং."""
    numbered = sum(
        1
        for item in agenda or []
        if _PROPOSAL_HEAD_RE.search((item.get("body") or "")[:200])
    )
    return numbered >= 2 and numbered >= len(agenda or []) // 2


def _normalize_overlap_text(value: str) -> str:
    """Normalize text only for overlap comparison, never for saved output.

    The original proposal text is always preserved. This comparison form merely
    ignores harmless differences in whitespace, HTML wrappers and letter case so
    the same overlapped page can be recognized when two Gemini chunks format it
    slightly differently.
    """
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = html.unescape(normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    return normalized


def _overlap_matches(previous_normalized: str, candidate_normalized: str) -> bool:
    """Return True when candidate is already represented at the previous end."""
    if not candidate_normalized:
        return True

    # An exact repeated ending is safe to remove even when it is short. Do not
    # remove a short phrase merely because the same common words appeared
    # somewhere earlier in the proposal.
    if previous_normalized.endswith(candidate_normalized):
        return True

    # Avoid treating a very short, common phrase as a duplicated page block.
    if len(candidate_normalized) < 120:
        return False

    if candidate_normalized in previous_normalized:
        return True

    suffix = previous_normalized[-len(candidate_normalized):]
    if not suffix:
        return False

    # Two extractions of the same overlapped page may differ by a character or
    # two (for example ধাতব vs धাতব) or by harmless spacing. A high fuzzy ratio
    # removes that duplicate without discarding genuinely different content.
    similarity = SequenceMatcher(
        None,
        suffix,
        candidate_normalized,
        autojunk=False,
    ).ratio()
    return similarity >= 0.97


def _continuation_cut_points(text: str) -> list:
    """Return safe raw-text boundaries for trimming a duplicated prefix.

    Most continuation blocks preserve line or paragraph boundaries. HTML closing
    tags are included too so manually supplied/generated rich text is handled
    safely. The returned offsets refer to the original string, preserving every
    character in any genuinely new tail.
    """
    points = {len(text)}
    for match in re.finditer(r"\n+", text):
        points.add(match.end())
    for match in re.finditer(
        r"</(?:p|table|div|ol|ul)>\s*",
        text,
        flags=re.IGNORECASE,
    ):
        points.add(match.end())
    return sorted(points, reverse=True)


def _append_unique_continuation(previous_text: str, continuation_text: str):
    """Append only the new part of a page/chunk continuation.

    Returns ``(combined_text, action)`` where action is one of:
      - ``duplicate``: the overlap was already present and nothing was appended;
      - ``trimmed``: a repeated prefix was removed and only its new tail appended;
      - ``appended``: no overlap was found, so the full continuation was appended;
      - ``empty``: the continuation had no text.

    This keeps the deliberate one-page overlap (which protects proposals from
    being split) while preventing that overlap from appearing twice in JSON.
    """
    previous = str(previous_text or "").rstrip()
    continuation = str(continuation_text or "").strip()

    if not continuation:
        return previous, "empty"
    if not previous:
        return continuation, "appended"

    previous_normalized = _normalize_overlap_text(previous)
    continuation_normalized = _normalize_overlap_text(continuation)

    # The whole headless block is already at the end of the numbered proposal.
    if _overlap_matches(previous_normalized, continuation_normalized):
        return previous, "duplicate"

    # Sometimes the repeated page is followed by genuinely new text. Find the
    # longest duplicated prefix and append only the remaining tail, so no part of
    # the proposal is lost and no page is repeated.
    for cut_point in _continuation_cut_points(continuation):
        if cut_point >= len(continuation):
            continue
        prefix = continuation[:cut_point].rstrip()
        prefix_normalized = _normalize_overlap_text(prefix)
        if not _overlap_matches(previous_normalized, prefix_normalized):
            continue

        new_tail = continuation[cut_point:].lstrip()
        if not new_tail:
            return previous, "duplicate"
        return (previous + "\n\n" + new_tail).strip(), "trimmed"

    # No duplicated overlap was detected. This is a genuine continuation and is
    # appended in full, ensuring the proposal remains one agenda entry.
    return (previous + "\n\n" + continuation).strip(), "appended"


def stitch_split_agenda_items(meeting: dict):
    """Re-join proposals split across page or JSON-chunk boundaries safely.

    Modern minutes require every independent agenda item to begin with
    ``প্রস্তাব নং``. A headless item is therefore a continuation of the preceding
    proposal. It is always folded back into that proposal, but duplicated text
    from the deliberate one-page overlap is removed first. Any genuinely new
    tail is preserved and appended.

    Old-format minutes without proposal numbers remain untouched. Returns
    ``(meeting, notes)`` for the Streamlit status message.
    """
    result = dict(meeting or {})
    agenda = [dict(item or {}) for item in (result.get("agenda") or [])]
    if not agenda or not _agenda_uses_proposal_numbers(agenda):
        return result, []

    stitched = []
    notes = []
    pending_leading = []

    for item in agenda:
        body = (item.get("body") or "").strip()
        starts_proposal = bool(_PROPOSAL_HEAD_RE.search(body[:200]))
        standalone = bool(_STANDALONE_SECTION_RE.match(body))

        if starts_proposal or standalone:
            # A full document normally cannot begin with a headless block. Keep
            # such a rare block pending rather than dropping it; when possible it
            # will be attached to the preceding numbered proposal below.
            if pending_leading and stitched:
                for orphan in pending_leading:
                    combined, _ = _append_unique_continuation(
                        stitched[-1].get("body"),
                        orphan.get("body"),
                    )
                    stitched[-1]["body"] = combined
                pending_leading.clear()
            stitched.append(item)
            continue

        if not stitched:
            pending_leading.append(item)
            continue

        previous = stitched[-1]
        previous_body = (previous.get("body") or "").rstrip()
        combined_body, body_action = _append_unique_continuation(
            previous_body,
            body,
        )
        previous["body"] = combined_body

        previous_resolution = (previous.get("resolution") or "").strip()
        resolution = (item.get("resolution") or "").strip()
        if resolution:
            combined_resolution, _ = _append_unique_continuation(
                previous_resolution,
                resolution,
            )
            previous["resolution"] = combined_resolution

        heading = re.search(
            r"প্রস্তাব\s*নং[^\n:ঃ]{0,28}", previous.get("body") or ""
        )
        label = (heading.group(0) if heading else previous_body[:40]).strip()
        snippet = re.sub(r"\s+", " ", body).strip()[:45]

        if body_action == "duplicate":
            notes.append(
                f'duplicate overlap starting "{snippet}…" was removed from "{label}…"'
            )
        elif body_action == "trimmed":
            notes.append(
                f'a repeated prefix was removed and the new continuation was joined to "{label}…"'
            )
        else:
            notes.append(
                f'a continuation block starting "{snippet}…" was joined to "{label}…"'
            )

    # Preserve any exceptional leading text rather than losing it. This only
    # applies to malformed/incomplete input that begins midway through a proposal.
    if pending_leading:
        stitched = pending_leading + stitched

    for position, item in enumerate(stitched, 1):
        item["serial"] = position
    result["agenda"] = stitched
    return result, notes


ISO_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?$"
)


def _finalize_scalars(meeting: dict) -> dict:
    """Deterministic local fixes for the top-level scalar fields.

    - serial: Bengali/mixed digit strings become a plain integer.
    - date: Bengali digits converted; a bare YYYY-MM-DD gains T00:00:00+06:00;
      a missing timezone gains +06:00.
    - type/status defaults enforced; agenda renumbered sequentially.
    """
    m = dict(meeting or {})

    serial = m.get("serial")
    if isinstance(serial, str):
        digits = re.sub(r"\D", "", serial.translate(BENGALI_TO_ARABIC_DIGITS))
        m["serial"] = int(digits) if digits else None

    date = m.get("date")
    if isinstance(date, str):
        d = date.strip().translate(BENGALI_TO_ARABIC_DIGITS)
        # Defensive fallback: old-format day-first dates (14.1.69 / ১২-৭-৭৪)
        # that slipped through unconverted. Years 00–30 → 20xx, else 19xx.
        old_style = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", d)
        if old_style:
            day, month, year = (int(x) for x in old_style.groups())
            if year < 100:
                year += 2000 if year <= 30 else 1900
            if 1 <= day <= 31 and 1 <= month <= 12:
                d = f"{year:04d}-{month:02d}-{day:02d}"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            d += "T00:00:00+06:00"
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", d):
            d += "+06:00"
        m["date"] = d or None

    m["type"] = m.get("type") or "academic"
    m["status"] = m.get("status") or "past"

    agenda = m.get("agenda") or []
    for i, item in enumerate(agenda, 1):
        if isinstance(item, dict):
            item["serial"] = i
    m["agenda"] = agenda
    m.setdefault("presentees", [])
    return m


def meeting_quality_report(meeting: dict) -> list:
    """Local sanity checks on the FINAL merged meeting — surfaced in the UI
    so nothing incomplete slips through unnoticed."""
    issues = []

    for field in ("serial", "title", "date", "president", "description", "conclusion"):
        if meeting.get(field) in (None, ""):
            issues.append(f"'{field}' could not be extracted")

    date = meeting.get("date")
    if isinstance(date, str) and not ISO_DATE_RE.fullmatch(date):
        issues.append(f"'date' is not valid ISO 8601: {date!r}")

    if isinstance(date, str) and ISO_DATE_RE.fullmatch(date):
        try:
            datetime.fromisoformat(date)
        except ValueError:
            issues.append("The meeting date is not a real calendar date; verify it against the PDF")
    if "[?]" in json.dumps(meeting, ensure_ascii=False):
        issues.append("Unreadable characters [?] remain in the record; review these against the PDF")
    presentees = meeting.get("presentees") or []
    if not presentees:
        issues.append("no presentees were extracted")
    else:
        nameless = sum(
            1 for p in presentees if not (p.get("name") or p.get("office"))
        )
        if nameless:
            issues.append(
                f"{nameless} presentee(s) have neither a name nor an office"
            )
        no_dept = sum(
            1 for p in presentees if not (p.get("department") or p.get("office"))
        )
        if no_dept:
            issues.append(
                f"{no_dept} presentee(s) have neither a department nor an office — "
                "verify them against the source PDF"
            )

    agenda = meeting.get("agenda") or []
    if not agenda:
        issues.append("no agenda items were extracted")
    else:
        unresolved = [
            str(item.get("serial"))
            for item in agenda
            if not (item.get("resolution") or "").strip()
        ]
        if unresolved:
            issues.append(
                "agenda item(s) without a resolution: " + ", ".join(unresolved)
            )
        if _agenda_uses_proposal_numbers(agenda):
            headless = [
                str(item.get("serial"))
                for item in agenda
                if not _PROPOSAL_HEAD_RE.search((item.get("body") or "")[:200])
                and not _STANDALONE_SECTION_RE.match((item.get("body") or "").strip())
            ]
            if headless:
                issues.append(
                    "agenda item(s) that do not start with 'প্রস্তাব নং': "
                    + ", ".join(headless)
                    + " — a proposal was probably split across a page boundary"
                )

    return issues


# ==========================================
# HTML formatting for Bengali sub-points and tables
# ==========================================
# NOTE: All generated HTML uses SINGLE-QUOTED attributes on purpose.
# JSON escapes double quotes as \" but leaves single quotes untouched, so
# single-quoted attributes keep the JSON string free of backslashes. The HTML
# then survives even a consumer that injects the raw string without a proper
# JSON decode, and CSS selectors / inline styles keep working.
BANGLA_LIST_LETTERS = "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ"

# Matches common Bengali list markers such as:
# (ক), ক), ক., ক:, ক।
BANGLA_POINT_PATTERN = re.compile(
    rf"(?<!\S)(?:\(([{BANGLA_LIST_LETTERS}])\)|"
    rf"([{BANGLA_LIST_LETTERS}])[\.\):।])\s*"
)

# These expressions commonly begin a paragraph that comes after the final
# Bengali list item. They help when OCR has removed the blank-line separator.
TRAILING_PARAGRAPH_PATTERN = re.compile(
    r"(?=(?:অতঃপর|পরিশেষে|পরবর্তীতে|উপর্যুক্ত|এতদসঙ্গে|সভায় আরও|সভাপতি মহোদয়)\s)",
    flags=re.IGNORECASE,
)

MARKDOWN_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")


def _clean_ocr_line_breaks(text):
    """Remove visual line wrapping introduced by PDF OCR."""
    if text is None:
        return None

    value = str(text).strip()
    if not value:
        return value

    value = value.replace("\r\n", "\n").replace("\r", "\n")

    # Remove page-marker lines if one survives the extraction stage.
    value = re.sub(
        r"^\s*===\s*PAGE\s+\d+\s*===\s*$",
        " ",
        value,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # A PDF line ending normally represents visual wrapping rather than a new
    # paragraph. Convert it to one ordinary space and collapse extra whitespace.
    value = re.sub(r"[ \t]*\n[ \t]*", " ", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()


def _escape_compact_html(text) -> str:
    """Escape text for HTML after removing OCR-created line wrapping.

    html.escape converts both double and single quotes to entities, so the
    escaped content can never break the single-quoted attributes used by the
    generators below, and it never adds a raw double quote to the JSON string.
    """
    cleaned = _clean_ocr_line_breaks(text)
    return html.escape(cleaned or "")



def _format_table_cell_html(text) -> str:
    """Escape table-cell text and convert Markdown bold to HTML <strong>."""
    cleaned = _clean_ocr_line_breaks(text)
    escaped = html.escape(cleaned or "")

    # Convert **text** into <strong>text</strong>.
    return re.sub(
        r"\*\*(.+?)\*\*",
        r"<strong>\1</strong>",
        escaped,
    )


def _split_final_item_and_trailing_paragraph(raw_text: str):
    """Separate a paragraph following the final Bengali list item."""
    value = str(raw_text or "").strip()
    if not value:
        return "", ""

    # Best signal: a blank line after the final item.
    blank_line_parts = re.split(r"\n[ \t]*\n+", value, maxsplit=1)
    if len(blank_line_parts) == 2:
        return blank_line_parts[0].strip(), blank_line_parts[1].strip()

    # OCR may remove the blank line, so also recognize common paragraph starts.
    match = TRAILING_PARAGRAPH_PATTERN.search(value)
    if match and match.start() > 0:
        return value[: match.start()].strip(), value[match.start() :].strip()

    return value, ""


def convert_bangla_points_to_html(text):
    """Convert (ক), (খ), (গ) points without adding numeric list markers."""
    if text is None:
        return None

    original = str(text).strip()
    if not original:
        return original

    # Prevent double conversion of a saved result.
    if re.search(
        r'<\s*(?:div|span)\b[^>]*class=["\'][^"\']*bangla-',
        original,
        flags=re.IGNORECASE,
    ):
        return original

    matches = list(BANGLA_POINT_PATTERN.finditer(original))

    # One marker may be ordinary text; require at least two list items.
    if len(matches) < 2:
        return _clean_ocr_line_breaks(original)

    introduction = original[: matches[0].start()].strip()
    items = []
    trailing_paragraph = ""

    for index, match in enumerate(matches):
        marker_text = match.group(1) or match.group(2)
        item_start = match.end()
        item_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(original)
        )
        raw_item_text = original[item_start:item_end].strip()

        if index == len(matches) - 1:
            raw_item_text, trailing_paragraph = (
                _split_final_item_and_trailing_paragraph(raw_item_text)
            )

        if not raw_item_text:
            continue

        marker_html = html.escape(marker_text)
        item_html = _escape_compact_html(raw_item_text)
        # Single-quoted attributes: no double quotes -> no \" in the JSON file.
        items.append(
            f"<div class='bangla-list-item' data-marker='{marker_html}'>"
            f"<span class='bangla-list-marker'>({marker_html})</span> "
            f"{item_html}</div>"
        )

    if len(items) < 2:
        return _clean_ocr_line_breaks(original)

    output = []
    if introduction:
        output.append(f"<p>{_escape_compact_html(introduction)}</p>")

    # A div-based list avoids the browser-generated 1., 2., 3. markers that an
    # <ol> can add. Only the Bengali markers remain visible.
    output.append(
        "<div class='bangla-letter-list'>"
        + "".join(items)
        + "</div>"
    )

    if trailing_paragraph:
        output.append(f"<p>{_escape_compact_html(trailing_paragraph)}</p>")

    return "".join(output)


def _split_markdown_table_row(line: str) -> list:
    """Split a Markdown pipe row without losing empty or escaped cells.

    Leading/trailing pipes are row borders and are removed. Interior empty cells
    are preserved, and ``\\|`` inside cell text is treated as a literal pipe.
    """
    value = str(line or "").strip()
    if not value:
        return []

    # Remove only unescaped outer row-border pipes.
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|") and not value.endswith(r"\|"):
        value = value[:-1]

    cells = []
    current = []
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\" and index + 1 < len(value) and value[index + 1] == "|":
            current.append("|")
            index += 2
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    cells.append("".join(current).strip())
    return cells


def _is_markdown_table_separator(line: str) -> bool:
    """Return True for one- or multi-column Markdown separator rows."""
    cells = _split_markdown_table_row(line)
    return bool(cells) and all(
        bool(MARKDOWN_SEPARATOR_CELL.fullmatch(cell.replace(" ", "")))
        for cell in cells
    )


def _looks_like_table_row(line: str) -> bool:
    """Return True for a one- or multi-column Markdown pipe row.

    One-column rows require their leading and trailing pipes. Multi-column rows
    may omit the outer borders, but must still contain at least one delimiter.
    """
    value = str(line or "").strip()
    if not value or "<table" in value.lower() or "</table" in value.lower():
        return False

    unescaped_pipes = len(re.findall(r"(?<!\\)\|", value))
    if unescaped_pipes < 1:
        return False

    cells = _split_markdown_table_row(value)
    if not cells or not any(cell for cell in cells):
        return False

    # A one-cell row is safe only when it has explicit outer borders.
    if len(cells) == 1:
        return value.startswith("|") and value.endswith("|")
    return True


# Gemini normally preserves Markdown row breaks, but a long continuation that
# begins on a new PDF page can occasionally arrive as one inline sequence:
# ``| Header | | --- | | Row 1 | | Row 2 |``. This form is unambiguous for
# one-column tables and is repaired before the regular line parser runs.
_FLATTENED_SINGLE_COLUMN_TABLE_RE = re.compile(
    r"(?P<header>\|\s*[^|\n<>]+?\s*\|)"
    r"[ \t\r\n]+"
    r"(?P<separator>\|\s*:?-{3,}:?\s*\|)"
    r"(?P<rows>(?:[ \t\r\n]+\|\s*(?!:?-{3,}:?\s*\|)[^|\n<>]*?\s*\|)+)",
    flags=re.IGNORECASE,
)


def _expand_flattened_single_column_tables(value: str) -> str:
    """Restore row breaks in flattened one-column Markdown tables."""
    source = str(value or "")

    def replace_match(match: re.Match) -> str:
        row_cells = re.findall(
            r"\|\s*([^|\n<>]*?)\s*\|",
            match.group("rows"),
        )
        rows = [f"| {cell.strip()} |" for cell in row_cells]
        if not rows:
            return match.group(0)
        return (
            "\n"
            + match.group("header").strip()
            + "\n"
            + match.group("separator").strip()
            + "\n"
            + "\n".join(rows)
            + "\n"
        )

    return _FLATTENED_SINGLE_COLUMN_TABLE_RE.sub(replace_match, source)


def _plain_table_text(value) -> str:
    """Normalize a cell for table-shape comparison without changing output."""
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _header_expected_kind(value: str) -> str:
    """Infer the broad value type suggested by a column heading."""
    text = _plain_table_text(value).casefold()
    if not text:
        return "unknown"

    if re.search(r"(?:letter\s*grade|লেটার\s*গ্রেড|গ্রেড\s*লেটার)", text):
        return "grade"
    if re.search(
        r"(?:name|নাম|বিবরণ|description|remarks?|মন্তব্য|বিভাগ|department|"
        r"faculty|অনুষদ|office|পদবী|designation)",
        text,
    ):
        return "text"
    if re.search(
        r"(?:course\s*(?:no|number|code)|কোর্স\s*(?:নং|নম্বর)|student\s*(?:no|number)|"
        r"স্টুডেন্ট\s*(?:নং|নম্বর)|roll\s*(?:no|number)|রোল\s*(?:নং|নম্বর))",
        text,
    ):
        return "code"
    if re.search(
        r"(?:grade\s*point|গ্রেড\s*প[য়য়]েন্ট|marks?|নম্বর|শতকরা|percentage|%|"
        r"credit|ক্রেডিট|serial|ক্রমিক|তারিখ|date|সময়|time|total|মোট|gpa|cgpa)",
        text,
    ) or re.search(r"\d", text):
        return "numeric"
    return "unknown"


def _observed_cell_kind(value: str) -> str:
    """Classify a data cell broadly for safe header/row alignment."""
    text = _plain_table_text(value)
    if not text:
        return "empty"
    compact = re.sub(r"\s+", "", text)

    if re.fullmatch(r"(?:[-–—]|n/?a|null)", compact, flags=re.IGNORECASE):
        return "numeric"
    if re.fullmatch(r"(?:A\+?|A-|B\+?|B-|C\+?|C-|D\+?|D-|F|P|PASS|FAIL)", compact, flags=re.IGNORECASE):
        return "grade"
    if re.fullmatch(r"[০-৯0-9.,:+/%()\-–—]+", compact):
        return "numeric"
    if re.fullmatch(r"[A-Za-z]{1,12}[ ._-]*[০-৯0-9][A-Za-z০-৯0-9 ._/'-]*", compact):
        return "code"
    if re.fullmatch(r"[০-৯0-9]{4,}[A-Za-z]?", compact):
        return "code"
    return "text"


def _kind_compatibility(expected: str, observed: str) -> float:
    """Score how naturally a data kind fits under a header kind."""
    if observed == "empty" or expected == "unknown":
        return 0.0
    scores = {
        "text": {"text": 3.0, "code": 0.8, "numeric": -1.3, "grade": -1.8},
        "numeric": {"numeric": 3.2, "code": 1.1, "grade": -1.0, "text": -2.0},
        "grade": {"grade": 4.0, "text": -1.2, "numeric": -2.0, "code": -1.5},
        "code": {"code": 3.2, "numeric": 1.0, "text": 0.2, "grade": -1.5},
    }
    return scores.get(expected, {}).get(observed, 0.0)


def _column_kind(rows: list, column_index: int) -> str:
    """Return the most common non-empty kind in one observed data column."""
    counts = {}
    for row in rows:
        if column_index >= len(row):
            continue
        kind = _observed_cell_kind(row[column_index])
        if kind == "empty":
            continue
        counts[kind] = counts.get(kind, 0) + 1
    if not counts:
        return "empty"
    return max(counts, key=lambda kind: (counts[kind], kind == "text"))


def _align_header_to_width(header: list, rows: list, target_width: int) -> list:
    """Insert missing blank header cells without shifting or deleting data.

    OCR/model output often omits an empty top-left heading. Dynamic alignment
    places the surviving headings over the body columns whose value types fit
    best. It can also recover a missing blank header in the middle or at right.
    """
    cells = list(header)
    if target_width <= 0:
        return cells
    if len(cells) >= target_width:
        return cells + [""] * (target_width - len(cells))

    column_kinds = [_column_kind(rows, index) for index in range(target_width)]
    count = len(cells)
    neg_inf = float("-inf")
    dp = [[neg_inf] * (target_width + 1) for _ in range(count + 1)]
    choice = [[None] * (target_width + 1) for _ in range(count + 1)]
    dp[0][0] = 0.0

    for i in range(count + 1):
        for j in range(target_width):
            current = dp[i][j]
            if current == neg_inf:
                continue

            # Leave this output column blank. A tiny penalty avoids unnecessary
            # leading blanks when the evidence is tied.
            skip_score = current - 0.08
            if skip_score > dp[i][j + 1]:
                dp[i][j + 1] = skip_score
                choice[i][j + 1] = (i, j, "skip")

            if i < count:
                expected = _header_expected_kind(cells[i])
                place_score = current + _kind_compatibility(expected, column_kinds[j])
                # Prefer placing over skipping when scores are exactly tied.
                if place_score >= dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = place_score
                    choice[i + 1][j + 1] = (i, j, "place")

    aligned = [""] * target_width
    i, j = count, target_width
    while j > 0:
        step = choice[i][j]
        if step is None:
            # Defensive fallback: preserve order and pad on the right.
            return cells + [""] * (target_width - len(cells))
        prev_i, prev_j, action = step
        if action == "place":
            aligned[j - 1] = cells[i - 1]
        i, j = prev_i, prev_j
    return aligned


def _align_short_row_to_header(row: list, header: list) -> list:
    """Insert omitted empty cells into a short row using header semantics."""
    target_width = len(header)
    cells = list(row)
    if len(cells) >= target_width:
        return cells + [""] * (target_width - len(cells))

    count = len(cells)
    neg_inf = float("-inf")
    dp = [[neg_inf] * (target_width + 1) for _ in range(count + 1)]
    choice = [[None] * (target_width + 1) for _ in range(count + 1)]
    dp[0][0] = 0.0

    for i in range(count + 1):
        for j in range(target_width):
            current = dp[i][j]
            if current == neg_inf:
                continue

            skip_score = current - 0.08
            if skip_score > dp[i][j + 1]:
                dp[i][j + 1] = skip_score
                choice[i][j + 1] = (i, j, "skip")

            if i < count:
                expected = _header_expected_kind(header[j])
                observed = _observed_cell_kind(cells[i])
                place_score = current + _kind_compatibility(expected, observed)
                if place_score >= dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = place_score
                    choice[i + 1][j + 1] = (i, j, "place")

    aligned = [""] * target_width
    i, j = count, target_width
    while j > 0:
        step = choice[i][j]
        if step is None:
            return cells + [""] * (target_width - len(cells))
        prev_i, prev_j, action = step
        if action == "place":
            aligned[j - 1] = cells[i - 1]
        i, j = prev_i, prev_j
    return aligned


def _normalize_table_shape(header: list, rows: list, separator_width: int = 0):
    """Return a rectangular table while preserving every extracted cell.

    Width is based on the widest of the header, separator and all body rows.
    No row is ever sliced. Missing cells are inserted as blanks and aligned by
    broad column semantics when possible.
    """
    clean_header = list(header or [])
    clean_rows = [list(row or []) for row in (rows or [])]
    widths = [len(clean_header), int(separator_width or 0)]
    widths.extend(len(row) for row in clean_rows)
    target_width = max(widths or [0])
    if target_width <= 0:
        return [], []

    # Use widest body rows as the most reliable column profiles.
    profile_rows = [row for row in clean_rows if len(row) == target_width]
    if not profile_rows:
        profile_rows = [row + [""] * (target_width - len(row)) for row in clean_rows]

    normalized_header = _align_header_to_width(
        clean_header,
        profile_rows,
        target_width,
    )
    normalized_rows = [
        _align_short_row_to_header(row, normalized_header)
        if len(row) < target_width
        else list(row)
        for row in clean_rows
    ]

    # A table header repeated after a page break is structural, not a data row.
    header_key = tuple(_plain_table_text(cell).casefold() for cell in normalized_header)
    deduped_rows = []
    for row in normalized_rows:
        row_key = tuple(_plain_table_text(cell).casefold() for cell in row)
        if any(header_key) and row_key == header_key:
            continue
        deduped_rows.append(row)

    return normalized_header, deduped_rows


def _render_html_table(header: list, rows: list, separator_width: int = 0) -> str:
    """Render a safe rectangular HTML table without ever dropping a cell."""
    normalized_header, normalized_rows = _normalize_table_shape(
        header,
        rows,
        separator_width=separator_width,
    )
    if not normalized_header:
        return ""

    header_html = "".join(
        f"<th style='border:1px solid #000;padding:6px;text-align:left;'>{_format_table_cell_html(cell)}</th>"
        for cell in normalized_header
    )

    body_rows = []
    for row in normalized_rows:
        cells_html = "".join(
            f"<td style='border:1px solid #000;padding:6px;'>{_format_table_cell_html(cell)}</td>"
            for cell in row
        )
        body_rows.append(f"<tr>{cells_html}</tr>")

    return (
        "<table class='meeting-table' border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse;width:100%;border:1px solid #000;'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )

def _contains_generated_rich_html(value: str) -> bool:
    """Detect HTML already created by this local formatter."""
    return bool(
        re.search(
            r'<\s*(?:table|div)\b[^>]*class=["\']'
            r'[^"\']*(?:meeting-table|bangla-letter-list|bangla-list-item)',
            value or "",
            flags=re.IGNORECASE,
        )
    )


def convert_tables_and_lists_to_html(text):
    """Convert Markdown tables and Bengali lettered points into compact HTML.

    Markdown tables are detected before OCR line breaks are cleaned. A table is
    recognized either from a classic '| --- |' separator row or from two or
    more consecutive pipe rows (separator omitted). Text outside tables is also
    checked for (ক), (খ), (গ) lists. This is entirely local and makes no
    Gemini request.
    """
    if text is None:
        return None

    original = str(text).strip()
    if not original:
        return original

    if _contains_generated_rich_html(original):
        return original

    # Repair the occasional inline representation of a one-column table before
    # the ordinary Markdown line parser examines it.
    original = _expand_flattened_single_column_tables(original)

    lines = original.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    segments = []
    text_buffer = []
    found_table = False

    def flush_text_buffer():
        if not text_buffer:
            return
        buffered = "\n".join(text_buffer).strip()
        text_buffer.clear()
        if buffered:
            segments.append(("text", buffered))

    index = 0
    while index < len(lines):
        header_line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""

        has_separator = _is_markdown_table_separator(next_line)
        is_table_start = _looks_like_table_row(header_line) and (
            has_separator or _looks_like_table_row(next_line)
        )

        if is_table_start:
            flush_text_buffer()
            found_table = True

            header = _split_markdown_table_row(header_line)
            separator_width = (
                len(_split_markdown_table_row(next_line)) if has_separator else 0
            )
            index += 2 if has_separator else 1  # skip separator only if present
            rows = []

            while index < len(lines):
                row_line = lines[index]
                if _is_markdown_table_separator(row_line):
                    index += 1
                    continue
                if not _looks_like_table_row(row_line):
                    break
                rows.append(_split_markdown_table_row(row_line))
                index += 1

            segments.append(
                (
                    "table",
                    _render_html_table(
                        header,
                        rows,
                        separator_width=separator_width,
                    ),
                )
            )
            continue

        text_buffer.append(header_line)
        index += 1

    flush_text_buffer()

    if not found_table:
        return convert_bangla_points_to_html(original)

    output = []
    for segment_type, value in segments:
        if segment_type == "table":
            output.append(value)
            continue

        converted = convert_bangla_points_to_html(value)
        if _contains_generated_rich_html(converted):
            output.append(converted)
        else:
            cleaned = _clean_ocr_line_breaks(converted)
            if cleaned:
                output.append(f"<p>{html.escape(cleaned)}</p>")

    return "".join(output)


def format_agenda_content_as_html(meeting: dict) -> dict:
    """Format tables and Bengali subpoints in agenda body and resolution."""
    formatted = dict(meeting or {})
    formatted_agenda = []

    for agenda_item in formatted.get("agenda") or []:
        clean_item = dict(agenda_item or {})
        clean_item["body"] = convert_tables_and_lists_to_html(
            clean_item.get("body")
        )
        clean_item["resolution"] = convert_tables_and_lists_to_html(
            clean_item.get("resolution")
        )
        formatted_agenda.append(clean_item)

    formatted["agenda"] = formatted_agenda
    return formatted



# ============================================================
# Reliable processing, saved results, and restart recovery
# ============================================================
PIPELINE_VERSION = "ecouncil-2026-09-v2"
CACHE_ROOT = Path(os.getenv("ECOUNCIL_CACHE_DIR", str(Path(__file__).resolve().parent / "processed_cache")))
PDF_LOCK = threading.RLock()
MAX_CACHE_BYTES = 50 * 1024 * 1024


class JSONIncompleteError(Exception):
    """Structured output reached its token limit and needs smaller sections."""


def stable_digest(value) -> str:
    raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def job_identity(stage, source_digest, settings):
    settings = {k: v for k, v in settings.items() if k not in ("workers", "safe_rpm")}
    rules = CHUNK_PROMPT if stage == "ocr" else (EXTRACTION_RULES + json.dumps(MEETING_SCHEMA, sort_keys=True))
    return stable_digest({"version": PIPELINE_VERSION, "stage": stage, "source": source_digest, "settings": settings, "rules": stable_digest(rules)})


def atomic_json_write(path, data):
    """Each writer gets a unique temporary file; replace is atomic on one disk."""
    temp_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".pending-", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        return True
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def read_json_file(path):
    try:
        if path.stat().st_size > MAX_CACHE_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None


def marker_sequence(text):
    return [int(m.group(1)) for m in PAGE_MARKER_RE.finditer(text or "")]


def valid_ocr(text, start, end):
    return isinstance(text, str) and marker_sequence(text) == list(range(start, end + 1)) and "COULD NOT BE READ — press Continue reading" not in text


def make_entry(stage, key, source_digest, settings, payload, **metadata):
    return {"version": PIPELINE_VERSION, "stage": stage, "key": key,
            "source_digest": source_digest, "settings": settings,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "complete": True, "payload": payload, "payload_digest": stable_digest(payload),
            **metadata}


def entry_valid(entry, stage, key):
    if not isinstance(entry, dict) or entry.get("version") != PIPELINE_VERSION:
        return False
    if entry.get("key") != key or entry.get("stage") != stage or entry.get("complete") is not True:
        return False
    try:
        if job_identity(stage, entry["source_digest"], entry["settings"]) != key:
            return False
        payload = entry["payload"]
        if entry.get("payload_digest") != stable_digest(payload):
            return False
        if stage == "ocr":
            count = entry.get("pages")
            if type(count) is not int or count <= 0 or not valid_ocr(payload, 1, count):
                return False
            checks = entry.get("page_checks", {})
            if not isinstance(checks, dict):
                return False
            for number, check in checks.items():
                if not str(number).isdigit() or not 1 <= int(number) <= count or not isinstance(check, dict):
                    return False
                if "needs_review" in check and type(check["needs_review"]) is not bool:
                    return False
                if not isinstance(check.get("readings", []), list) or not isinstance(check.get("differences", []), list):
                    return False
                for reading in check.get("readings", []):
                    if not isinstance(reading, dict) or not isinstance(reading.get("label"), str) or not valid_ocr(reading.get("text"), int(number), int(number)):
                        return False
            if entry["settings"].get("accuracy_strategy") and set(checks) != {str(n) for n in range(1, count + 1)}:
                return False
            return True
        if not isinstance(payload, dict) or not all(isinstance(payload.get(k), list) for k in ("presentees", "agenda")):
            return False
        _validated_meeting_partial(copy.deepcopy(payload))
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def cache_relative(stage, key):
    return f"v2/{stage}/{key}.json"


def load_entry(stage, key, imported=None):
    entry = (imported or {}).get(cache_relative(stage, key))
    if entry_valid(entry, stage, key):
        return entry
    entry = read_json_file(CACHE_ROOT / cache_relative(stage, key))
    return entry if entry_valid(entry, stage, key) else None


def save_entry(entry):
    return atomic_json_write(CACHE_ROOT / cache_relative(entry["stage"], entry["key"]), entry)


def checkpoint_path(stage, key, part):
    return CACHE_ROOT / "checkpoints" / stage / key / f"{part}.json"


def load_checkpoint(stage, key, part):
    saved = read_json_file(checkpoint_path(stage, key, part))
    if isinstance(saved, dict) and saved.get("key") == key and saved.get("checksum") == stable_digest(saved.get("payload")):
        return saved.get("payload")
    return None


def save_checkpoint(stage, key, part, payload):
    return atomic_json_write(checkpoint_path(stage, key, part), {"key": key, "payload": payload, "checksum": stable_digest(payload)})


def safe_error(error, keys=()):
    message = str(error)
    message = re.sub(r"\bJSON\b", "meeting record", message, flags=re.IGNORECASE)
    for key in keys:
        if key:
            message = message.replace(key, "[hidden]")
    message = re.sub(r"AIza[\w-]+", "[hidden]", message)
    upper = message.upper()
    if "429" in upper or "RESOURCE_EXHAUSTED" in upper:
        return "The Gemini request limit was reached. Wait for the quota to reset, then resume. Completed work is saved."
    if "401" in upper or "403" in upper or "API KEY" in upper:
        return "Gemini could not authorize this request. Check the configured API key and its permissions."
    if "404" in upper or "NOT_FOUND" in upper:
        return "The selected model was not found. Check the Gemini model name in Advanced settings."
    if "TIMEOUT" in upper:
        return "The request timed out. Resume to retry the unfinished section."
    return message[:320] or type(error).__name__


# PyMuPDF calls are serialized; network requests still run concurrently.
# Separate documents are used for each operation and never shared by threads.
def locked_pdf_function(function):
    def wrapper(*args, **kwargs):
        with PDF_LOCK:
            return function(*args, **kwargs)
    return wrapper


for _pdf_function_name in ("split_pdf_into_chunks", "render_chunk_page_images", "render_page_detail_views", "_split_pdf_bytes_in_half", "extract_single_page_pdf", "extract_text_layer_pages"):
    globals()[_pdf_function_name] = locked_pdf_function(globals()[_pdf_function_name])


def pdf_metadata(data):
    with PDF_LOCK, fitz.open(stream=data, filetype="pdf") as document:
        if document.needs_pass:
            raise ValueError("This PDF is password protected. Upload an unlocked copy.")
        if len(document) == 0:
            raise ValueError("This PDF has no pages.")
        return {"pages": len(document), "bytes": len(data)}


def page_pdf(data, start, end):
    with PDF_LOCK, fitz.open(stream=data, filetype="pdf") as source, fitz.open() as part:
        part.insert_pdf(source, from_page=start - 1, to_page=end - 1)
        return part.tobytes(garbage=1, deflate=True)


@st.cache_data(show_spinner=False, max_entries=6, ttl=600)
def _cached_pdf_preview(source_digest, page_number, _data):
    """Keep a few high-resolution previews, keyed by the exact document digest."""
    with PDF_LOCK, fitz.open(stream=_data, filetype="pdf") as document:
        page = document[page_number - 1]
        scale = min(4.0, 4096 / max(page.rect.width, page.rect.height))
        return page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png")


def pdf_preview(data, page_number):
    return _cached_pdf_preview(stable_digest(data), page_number, data)


def pdf_viewer_html(preview, page_number):
    """Local-only zoom/pan controls: changing zoom never reruns the text editor."""
    image_data = base64.b64encode(preview).decode("ascii")
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<style>
*{box-sizing:border-box}html,body{margin:0;height:100%;font:16px system-ui,sans-serif;color:#30292b;background:#faf6f7}
body{display:flex;flex-direction:column;border:1px solid #ead5d8;border-radius:10px;overflow:hidden}
.toolbar{display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding:9px;background:#fff0f2;border-bottom:1px solid #ead5d8}
button{border:1px solid #ce8e99;border-radius:6px;background:white;color:#8c1423;min-width:34px;min-height:34px;padding:4px 9px;font:inherit;cursor:pointer}
button:hover{background:#f8e5e8}button:focus-visible,input:focus-visible,.viewport:focus-visible{outline:3px solid #ad1024;outline-offset:-3px}
button:disabled{opacity:.45;cursor:default}input{accent-color:#ad1024;width:90px;min-width:50px;flex:1;max-width:145px}
output{min-width:43px;font-variant-numeric:tabular-nums}.viewport{flex:1;min-height:0;overflow:auto;background:#f0e7e9;padding:12px;overscroll-behavior:contain}
img{display:block;max-width:none;height:auto;background:#fff;box-shadow:0 1px 5px #0002;cursor:grab;user-select:none;-webkit-user-drag:none}
.viewport.dragging img{cursor:grabbing}.hint{font-size:14px;padding:5px 10px;background:#fff0f2;color:#625559}
</style></head><body>
<div class="toolbar" role="toolbar" aria-label="Original PDF zoom controls">
<button id="out" aria-label="Zoom out" title="Zoom out">−</button>
<button id="fit" title="Fit page width">Fit</button>
<button id="in" aria-label="Zoom in" title="Zoom in">+</button>
<input id="zoom" aria-label="Original PDF zoom" type="range" min="50" max="400" step="25" value="100">
<output id="value" aria-live="polite">100%</output>
<button id="full" hidden>Full screen</button></div>
<div class="viewport" id="viewport" tabindex="0" role="region" aria-label="Zoomable original page">
<img id="page" alt="Original PDF, page ''' + str(int(page_number)) + '''" draggable="false" src="data:image/png;base64,''' + image_data + '''"></div>
<div class="hint">100% fits the width. Zoom in, then scroll or drag to read small text.</div>
<script>
const viewport=document.getElementById('viewport'), picture=document.getElementById('page');
const slider=document.getElementById('zoom'), output=document.getElementById('value');
const minus=document.getElementById('out'), plus=document.getElementById('in'), full=document.getElementById('full');
let zoom=100,drag=null;
function render(){picture.style.width=Math.max(1,(viewport.clientWidth-24)*zoom/100)+'px';slider.value=zoom;output.textContent=zoom+'%';minus.disabled=zoom<=50;plus.disabled=zoom>=400;}
function change(value){
 const oldWidth=picture.clientWidth||1,oldHeight=picture.clientHeight||1;
 const centerX=(viewport.scrollLeft+viewport.clientWidth/2)/oldWidth;
 const centerY=(viewport.scrollTop+viewport.clientHeight/2)/oldHeight;
 zoom=Math.min(400,Math.max(50,Number(value)));render();
 viewport.scrollLeft=centerX*picture.clientWidth-viewport.clientWidth/2;
 viewport.scrollTop=centerY*picture.clientHeight-viewport.clientHeight/2;
}
plus.onclick=()=>change(zoom+25);minus.onclick=()=>change(zoom-25);
document.getElementById('fit').onclick=()=>{change(100);viewport.scrollTo(0,0);};
slider.oninput=()=>change(slider.value);
viewport.addEventListener('keydown',event=>{if(event.key==='+'||event.key==='='){event.preventDefault();change(zoom+25);}else if(event.key==='-'){event.preventDefault();change(zoom-25);}else if(event.key==='0'){event.preventDefault();change(100);viewport.scrollTo(0,0);}});
viewport.addEventListener('pointerdown',event=>{if(event.pointerType!=='mouse'||event.button!==0)return;drag={x:event.clientX,y:event.clientY,left:viewport.scrollLeft,top:viewport.scrollTop};viewport.setPointerCapture(event.pointerId);viewport.classList.add('dragging');});
viewport.addEventListener('pointermove',event=>{if(!drag)return;viewport.scrollLeft=drag.left+drag.x-event.clientX;viewport.scrollTop=drag.top+drag.y-event.clientY;});
function endDrag(){drag=null;viewport.classList.remove('dragging');}
viewport.addEventListener('pointerup',endDrag);viewport.addEventListener('pointercancel',endDrag);
if(document.fullscreenEnabled){full.hidden=false;full.onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch(error){full.hidden=true;}};}
document.addEventListener('fullscreenchange',()=>{full.textContent=document.fullscreenElement?'Exit full screen':'Full screen';render();});
new ResizeObserver(render).observe(viewport);picture.onload=render;render();
</script></body></html>'''


class ProcessingJob:
    """Background workers do not call Streamlit or mutate session state."""
    def __init__(self, stage, key, source_digest, settings):
        self.id = uuid.uuid4().hex
        self.stage, self.key = stage, key
        self.source_digest, self.settings = source_digest, dict(settings)
        self.lock = threading.RLock()
        self.pause = threading.Event()
        self.status = "running"
        self.started = time.monotonic()
        self.finished = None
        self.total = self.done = self.restored = self.local = 0
        self.message = "Preparing the document…"
        self.errors = {}
        self.notes = []
        self.page_checks = {}
        self.result = None
        self.partial = ""

    def note(self, message):
        with self.lock:
            if message not in self.notes:
                self.notes.append(message)

    def snapshot(self):
        with self.lock:
            return {k: copy.copy(getattr(self, k)) for k in (
                "id", "stage", "key", "status", "started", "finished", "total", "done",
                "restored", "local", "message", "errors", "notes", "result", "partial")}

    def run(self, function, *args):
        try:
            function(self, *args)
        except Exception as error:
            with self.lock:
                self.errors["Document"] = safe_error(error, args[-1] if args and isinstance(args[-1], list) else [])
                self.status = "paused" if self.pause.is_set() else "failed"
        finally:
            with self.lock:
                self.finished = time.monotonic()
                if self.status == "running":
                    self.status = "paused" if self.pause.is_set() else "failed"


@st.cache_resource(show_spinner=False)
def job_registry():
    return {"lock": threading.Lock(), "jobs": {}}


def start_job(job, function, *args):
    registry = job_registry()
    with registry["lock"]:
        for previous in registry["jobs"].values():
            if previous.key == job.key and previous.stage == job.stage and previous.status == "running" and not previous.pause.is_set():
                return previous.id  # Reconnect to the same work after a browser refresh.
        for job_id, previous in list(registry["jobs"].items()):
            if previous.finished and time.monotonic() - previous.finished > 3600:
                registry["jobs"].pop(job_id, None)
        finished_jobs = sorted((j for j in registry["jobs"].values() if j.finished), key=lambda j: j.finished)
        for previous in finished_jobs[:-16]:
            registry["jobs"].pop(previous.id, None)
        if sum(j.status == "running" for j in registry["jobs"].values()) >= 4:
            raise ValueError("The app is processing several documents. Please try again shortly.")
        registry["jobs"][job.id] = job
    threading.Thread(target=job.run, args=(function, *args), daemon=True).start()
    return job.id


def get_active_job():
    return job_registry()["jobs"].get(st.session_state.get("active_job_id"))


def bounded_process(job, items, worker, accept, workers):
    """Only hold as many page payloads/futures as there are active workers."""
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {}
        exhausted = False
        while futures or not exhausted:
            while not exhausted and not job.pause.is_set() and len(futures) < workers:
                item = next(iterator, None)
                if item is None:
                    exhausted = True
                    break
                futures[executor.submit(worker, item)] = item
            if job.pause.is_set():
                exhausted = True
            if not futures:
                break
            completed, _ = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in completed:
                item = futures.pop(future)
                try:
                    accept(item, future.result())
                except Exception as error:
                    with job.lock:
                        job.errors[str(item[0])] = safe_error(error)
                    if _is_fatal_api_error(error):
                        job.pause.set()  # do not charge for every remaining page on a bad configuration


ACCURACY_STRATEGY = "independent-page-views-v1"


def reading_signature(text):
    # Ignore layout whitespace only. Preserve punctuation, glyphs and ALL digits.
    body = PAGE_MARKER_RE.sub("", text or "")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", body)).strip()


def reading_differences(first, second):
    """Bound the comparison size; retain short, literal disagreements for review."""
    a, b = reading_signature(first).split(), reading_signature(second).split()
    differences = []
    for tag, i, j, k, l in SequenceMatcher(None, a[:6000], b[:6000], autojunk=False).get_opcodes():
        if tag != "equal":
            differences.append({"First reading": " ".join(a[i:j])[:240] or "(missing)",
                                "Other reading": " ".join(b[k:l])[:240] or "(missing)"})
            if len(differences) >= 12:
                break
    return differences


def compare_page_readings(readings, page):
    """Agreement helps triage; it is not a correctness score or human approval."""
    signatures = [reading_signature(text) for text in readings]
    first, second = signatures[:2]
    disagreed = first != second
    selected = 0
    reason = "Two independent readings agree."
    if disagreed:
        reason = "Independent readings differ; compare the highlighted details with the PDF."
        # Prefer corroborated text only if it does not drop substantial text or
        # numeric fields. Always retain ALL readings and flag the disagreement.
        if len(signatures) > 2 and second == signatures[2] and second:
            digits_a = re.findall(r"\d+", first)
            digits_b = re.findall(r"\d+", second)
            if len(second) >= len(first) * .9 and len(digits_b) >= len(digits_a):
                selected = 1
                reason = "Readings 2 and 3 agree. Their text was chosen initially; all differences remain available for review."
        elif len(signatures) > 2 and first == signatures[2]:
            reason = "Readings 1 and 3 agree. The first text was chosen initially; review the disagreement."
    chosen = readings[selected]
    blank = not signatures[selected]
    number_difference = any(re.findall(r"\d+", value) != re.findall(r"\d+", first) for value in signatures[1:])
    check = {
        "status": "disagreement" if disagreed else "agreement",
        "needs_review": disagreed or "[?]" in chosen or blank,
        "reason": "No text was returned; confirm this page is blank." if blank else reason,
        "number_difference": number_difference,
        "selected_reading": selected + 1,
        "passes": len(readings),
        "differences": reading_differences(readings[0], readings[1]) if disagreed else [],
        "readings": [{"label": f"Reading {i + 1}", "text": renumber_pages(t, page)}
                     for i, t in enumerate(readings)] if disagreed else [],
    }
    return renumber_pages(chosen, page), check


def read_page_with_verification(job, raw, page, pool, reuse):
    settings = job.settings

    def read(pass_number):
        part = f"{page}-read-{pass_number}"
        previous = load_checkpoint("ocr_reads", job.key, part) if reuse else None
        if valid_ocr(previous, 1, 1):
            return previous
        with job.lock:
            job.message = f"Checking page {page}: reading {pass_number} of up to 3…"
        # Each request is independent: no previous transcription is in its prompt.
        # The second uses unaltered color and crops to preserve faint annotations.
        result = ocr_chunk_bulletproof(
            raw, 1, pool, settings["model"],
            input_mode=settings["input_mode"] if pass_number == 1 else "images",
            dpi=settings["dpi"],
            preprocess=settings["preprocess"] if pass_number == 1 else ("original" if pass_number == 2 else "standard"),
            detail_views=pass_number > 1, verification=True,
        )
        result = clean_bengali_ocr_text(result)
        if not valid_ocr(result, 1, 1):
            raise OCRIncompleteError("The verification response has invalid page markers.")
        if not save_checkpoint("ocr_reads", job.key, part, result):
            job.note("The server could not save verification progress. Keep this tab open and download the completed text and reading checks.")
        return result

    readings = [read(1), read(2)]
    if reading_signature(readings[0]) != reading_signature(readings[1]):
        readings.append(read(3))
    return compare_page_readings(readings, page)


def run_ocr_job(job, pdf_data, reuse, keys):
    settings = job.settings
    total_pages = pdf_metadata(pdf_data)["pages"]
    accuracy = bool(settings.get("accuracy_strategy"))
    size = 1 if accuracy else settings["chunk_size"]
    ranges = [(start, min(start + size - 1, total_pages)) for start in range(1, total_pages + 1, size)]
    results = {}
    with job.lock:
        job.total = total_pages
        job.message = "Checking saved progress and embedded text…"
    for start, end in ranges:
        saved = load_checkpoint("ocr", job.key, f"{start}-{end}") if reuse else None
        saved_text = saved.get("text") if isinstance(saved, dict) else saved
        checks = saved.get("page_checks", {}) if isinstance(saved, dict) else {}
        if valid_ocr(saved_text, start, end) and (not accuracy or str(start) in checks):
            results[start] = saved_text
            job.page_checks.update(checks)
            job.done += end - start + 1
            job.restored += end - start + 1
    # Avoid re-scanning text layers when every page is already checkpointed.
    local_pages, _ = extract_text_layer_pages(pdf_data) if settings["use_text_layer"] and len(results) < len(ranges) else ({}, {})
    pool = APIKeyPool(keys, settings["safe_rpm"]) if keys else None
    if pool:
        pool.ocr_thinking = settings["thinking"]

    def worker(item):
        start, end = item
        pieces, page, checks = [], start, {}
        while page <= end:
            if page - 1 in local_pages:
                pieces.append(f"=== PAGE {page} ===\n" + clean_bengali_ocr_text(local_pages[page - 1]))
                checks[str(page)] = {"status": "digital_text", "needs_review": False,
                                     "reason": "Extracted from embedded PDF text; no AI verification was performed."}
                with job.lock:
                    job.local += 1
                page += 1
                continue
            if pool is None:
                raise ValueError("Reading scanned pages needs GEMINI_API_KEYS in Streamlit Secrets. Digital text pages and saved results can be used without a key.")
            last = page
            while last < end and last not in local_pages:
                last += 1
            raw = page_pdf(pdf_data, page, last)
            if accuracy:
                text, check = read_page_with_verification(job, raw, page, pool, reuse)
                checks[str(page)] = check
                pieces.append(text)
            else:
                text = ocr_chunk_bulletproof(raw, last - page + 1, pool, settings["model"],
                                            input_mode=settings["input_mode"], dpi=settings["dpi"], preprocess=settings["preprocess"])
                pieces.append(renumber_pages(clean_bengali_ocr_text(text), page))
            page = last + 1
        text = "\n\n".join(pieces)
        if not valid_ocr(text, start, end):
            raise OCRIncompleteError("The page sequence is incomplete or duplicated. Resume to retry.")
        # Preserve the existing isolated-page cross-check. Report differences;
        # never claim the second probabilistic reading proves correctness.
        if settings["cross_check"] and pool and not accuracy:
            for issue in page_boundary_item_report(text):
                flagged = issue["page"]
                try:
                    solo = ocr_chunk_bulletproof(page_pdf(pdf_data, flagged, flagged), 1, pool, settings["model"],
                                                input_mode=settings["input_mode"], dpi=settings["dpi"], preprocess=settings["preprocess"])
                    solo = renumber_pages(clean_bengali_ocr_text(solo), flagged)
                    old, new = page_item_numbers(text, flagged), page_item_numbers(solo, flagged)
                    if old and new and old != new:
                        original = page_blocks(text)[flagged]
                        checks[str(flagged)] = {"status": "disagreement", "needs_review": True,
                            "reason": "Item numbers differ on a second reading. The first text is kept for review.",
                            "number_difference": True, "passes": 2, "selected_reading": 1,
                            "differences": reading_differences(original, solo),
                            "readings": [{"label": "Reading 1", "text": original}, {"label": "Reading 2", "text": solo}]}
                        job.note(f"Page {flagged}: readings disagree on item numbers ({old} / {new}). Check the original PDF.")
                except Exception as error:
                    checks[str(flagged)] = {"status": "check_failed", "needs_review": True,
                                           "reason": "The second reading could not finish. Review this page manually."}
                    job.note(f"Page {flagged}: the second reading could not finish; review this page manually. " + safe_error(error, keys))
        return {"text": text, "page_checks": checks}

    def accept(item, result):
        start, end = item
        text = result["text"]
        results[start] = text
        if not save_checkpoint("ocr", job.key, f"{start}-{end}", result):
            job.note("Automatic saving is unavailable on this server. Download your text before leaving.")
        with job.lock:
            job.page_checks.update(result["page_checks"])
            job.done += end - start + 1
            job.message = f"{job.done} of {total_pages} pages read"
            job.partial = "\n\n".join(results[n] for n in sorted(results))

    pending = [item for item in ranges if item[0] not in results]
    bounded_process(job, pending, worker, accept, settings["workers"] if NEW_SDK else 1)
    combined = "\n\n".join(results[n] if n in results else f"[PAGES {n}–{end} COULD NOT BE READ — press Continue reading to try again]" for n, end in ranges)
    with job.lock:
        job.partial = combined
    if len(results) != len(ranges):
        job.status = "paused" if job.pause.is_set() else "failed"
        return
    if not valid_ocr(combined, 1, total_pages):
        raise OCRIncompleteError("Document page order failed its final check.")
    entry = make_entry("ocr", job.key, job.source_digest, settings, combined, pages=total_pages,
                       notes=job.notes, page_checks=job.page_checks)
    if not save_entry(entry):
        job.note("The server could not save the finished result. Download the extracted text below.")
    with job.lock:
        job.result, job.status = entry, "complete"
        job.message = "Text ready for review"


def extract_json_resilient(text, index, total, pool, model, depth=0):
    try:
        return gemini_extract_meeting(text, index, total, pool, model)
    except JSONIncompleteError:
        if depth >= 3 or len(text) < 4000:
            raise ValueError("This section still exceeds the output limit. Lower Text size per request and retry.")
        parts = split_text_with_overlap(text, max(2000, len(text) // 2))
        partials = [extract_json_resilient(part, index, total, pool, model, depth + 1) for part in parts]
        return merge_meeting_partials(partials)


def run_json_job(job, source, reuse, keys):
    if not keys:
        raise ValueError("Generating a new meeting record needs GEMINI_API_KEYS in Streamlit Secrets.")
    settings = job.settings
    chunks = split_text_with_overlap(source, settings["chunk_chars"])
    results = {}
    job.total = len(chunks)
    for index, chunk in enumerate(chunks):
        saved = load_checkpoint("json", job.key, index) if reuse else None
        if isinstance(saved, dict):
            try:
                results[index] = _validated_meeting_partial(saved)
            except (TypeError, ValueError):
                continue
    job.done = job.restored = len(results)
    job.message = "Building the meeting record…"
    pool = APIKeyPool(keys, settings["safe_rpm"])

    def worker(item):
        index, chunk = item
        return extract_json_resilient(chunk, index + 1, len(chunks), pool, settings["model"])

    def accept(item, result):
        index, _ = item
        results[index] = result
        if not save_checkpoint("json", job.key, index, result):
            job.note("Automatic saving is unavailable; keep this tab open until the record is ready.")
        with job.lock:
            job.done += 1
            job.message = f"{job.done} of {len(chunks)} sections processed"

    bounded_process(job, [(i, chunk) for i, chunk in enumerate(chunks) if i not in results], worker, accept, settings["workers"] if NEW_SDK else 1)
    if len(results) != len(chunks):
        job.status = "paused" if job.pause.is_set() else "failed"
        return
    partials = [results[i] for i in range(len(chunks))]
    final = merge_meeting_partials(partials) if len(partials) > 1 else dict(partials[0])
    final, stitch_notes = stitch_split_agenda_items(final)
    final = normalize_meeting_entities(_finalize_scalars(final), name_roster=settings["roster"])
    corrections = final.pop("_name_corrections", [])
    final = format_agenda_content_as_html(final)
    _validated_meeting_partial(copy.deepcopy(final))
    entry = make_entry("json", job.key, job.source_digest, settings, final, stitch_notes=stitch_notes, corrections=corrections, incomplete_source="COULD NOT BE READ — press Continue reading" in source)
    if not save_entry(entry):
        job.note("The server could not save this record. Download the meeting record below.")
    with job.lock:
        job.result, job.status = entry, "complete"
        job.message = "Meeting record ready"


def make_backup(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            if entry and entry_valid(entry, entry.get("stage"), entry.get("key")):
                archive.writestr("processed_cache/" + cache_relative(entry["stage"], entry["key"]), json.dumps(entry, ensure_ascii=False, indent=2))
        archive.writestr("README.txt", "Place the processed_cache folder beside your Streamlit Python file and commit it to the repository. Use the same PDF and processing settings. Cloud runtime files are not automatically committed to GitHub.\n")
    return output.getvalue()


def parse_backup(data):
    if len(data) > MAX_CACHE_BYTES:
        raise ValueError("Backup is too large (maximum 50 MB).")
    result = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > 100 or sum(info.file_size for info in infos) > MAX_CACHE_BYTES:
            raise ValueError("Backup has too many entries or exceeds 50 MB when opened.")
        for info in infos:
            match = re.fullmatch(r"processed_cache/(v2/(ocr|json)/([0-9a-f]{64})\.json)", info.filename)
            if not match:
                continue  # No filesystem extraction; traversal paths are never used.
            entry = json.loads(archive.read(info))
            if not entry_valid(entry, match[2], match[3]):
                raise ValueError("This backup contains an invalid or incomplete record.")
            result[match[1]] = entry
    if not result:
        raise ValueError("No valid processed-document results were found in this ZIP.")
    return result


# ============================================================
# Interface — two deliberate steps, with review between them
# ============================================================
APP_CSS = """
<style>
/* Native controls use the matching light palette in .streamlit/config.toml. */
:root { --ec-red:#ad1024; --ec-border:#ead5d8; }
html { font-size:17px; }
.block-container { max-width:1440px; padding:3rem 2rem; }
h1,h2,h3 { letter-spacing:-.025em; }
h2 { font-size:1.65rem !important; }
h3 { font-size:1.18rem !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stButton"] p,
[data-testid="stDownloadButton"] p, [data-testid="stFormSubmitButton"] p,
[data-testid="stRadio"] label p, [data-testid="stCheckbox"] label p { font-size:1rem; }
.ec-header { display:flex; align-items:center; justify-content:space-between; gap:18px; background:#fff; color:#30292b; border:1px solid var(--ec-border); border-top:4px solid var(--ec-red); border-radius:0 0 15px 15px; padding:20px; box-shadow:0 6px 20px #65101d08; margin-bottom:20px; }
.ec-brand { display:flex; align-items:center; gap:15px; min-width:0; }
.ec-logo { display:grid; place-items:center; flex-shrink:0; width:54px; height:54px; background:var(--ec-red); color:#fff; border-radius:12px; font-size:.8rem; font-weight:800; box-shadow:0 4px 12px #ad102426; }
.ec-header h1 { font-size:1.6rem !important; line-height:1.25 !important; color:#a30d20; padding:0; margin:0; }
.ec-brand-subtitle { color:#625559; font-size:.95rem; margin-top:4px; }
.ec-pill { display:flex; align-items:center; gap:8px; color:#8c1423; background:#fff5f6; border:1px solid #ebc4ca; padding:8px 12px; border-radius:24px; font-size:.88rem; font-weight:650; white-space:nowrap; }
.ec-dot { width:8px; height:8px; border-radius:50%; background:#18845b; box-shadow:0 0 0 3px #18845b15; }
.ec-pill.off .ec-dot { background:#996b18; box-shadow:none; }
.ec-intro { color:#45383b; background:#fff; border:1px solid var(--ec-border); border-left:5px solid var(--ec-red); border-radius:12px; padding:17px 20px; margin-bottom:30px; line-height:1.7; box-shadow:0 5px 18px #65101d04; }
.ec-intro strong { color:#8f1424; }
.ec-section { display:flex; gap:14px; align-items:center; margin:30px 0 8px; }
.ec-number { color:#fff; background:var(--ec-red); width:42px; height:44px; flex-shrink:0; border-radius:11px; display:grid; place-items:center; font-weight:750; box-shadow:0 5px 12px #ad102420; }
.ec-section h2 { padding:0; margin:0; }
.ec-sidebar-title { color:#a30d20; font-size:1.5rem; font-weight:750; margin-bottom:14px; }
.ec-file { color:#30292b; background:#fff; border:1px solid var(--ec-border); padding:14px 17px; border-radius:10px; margin:4px 0 14px; }
.ec-file small { display:block; color:#625559; margin-top:5px; font-size:.94rem; }
.stButton button, .stDownloadButton button { border-radius:9px; min-height:44px; font-weight:600; }
.stButton button[kind="primary"]:not(:disabled), [data-testid="stFormSubmitButton"] button[kind="primary"]:not(:disabled) { background:var(--ec-red); border-color:var(--ec-red); color:#fff; }
.stButton button[kind="primary"]:not(:disabled) p, [data-testid="stFormSubmitButton"] button[kind="primary"]:not(:disabled) p { color:#fff; }
.stButton button[kind="primary"]:not(:disabled):hover { background:#890d1c; border-color:#890d1c; }
[data-testid="stExpander"] { border-radius:12px; }
[data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:hover p { color:inherit !important; }
button[kind="secondary"]:not(:disabled):hover { color:inherit; }
button:focus-visible, textarea:focus-visible, input:focus-visible { outline:3px solid #ad1024 !important; outline-offset:2px; }
[data-testid="stMetric"] { border:1px solid var(--ec-border); border-radius:10px; padding:12px 16px; }
[data-testid="stMetricValue"] { font-size:1.5rem; }
[data-testid="stTextArea"] textarea { line-height:1.8; font-size:1.06rem; resize:vertical; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:inherit !important; font-size:.94rem; }
[data-testid="stCaptionContainer"] { opacity:1 !important; }
[data-testid="stSliderThumbValue"] { color:inherit; }
[data-testid="stFileUploaderDropzone"] { border:2px dashed #ce8e99; border-radius:12px; }
[data-testid="stAlert"] { border-radius:10px; }
@media (max-width:700px) {
 .block-container { padding:3rem 1rem 2rem; }
 .ec-header { flex-wrap:wrap; padding:16px; }
 .ec-header h1 { font-size:1.3rem !important; }
 .ec-intro { padding:14px 16px; }
 .ec-section h2 { font-size:1.4rem !important; }
}
</style>
"""


def clear_record():
    for name in ("json_entry", "json_result", "json_job_key", "json_job_complete", "json_partials_done", "record_source_key"):
        st.session_state.pop(name, None)


def set_ocr_result(entry, filename, origin):
    st.session_state.pop("last_job_issue", None)
    st.session_state.pop("partial_ocr", None)
    old = st.session_state.get("ocr_result")
    st.session_state.update(ocr_entry=entry, ocr_result=entry["payload"], ocr_editor=entry["payload"],
                            ocr_filename=filename.rsplit(".", 1)[0], ocr_origin=origin,
                            job_complete=True, job_key=entry["key"])
    st.session_state["edit_revision"] = st.session_state.get("edit_revision", 0) + 1
    if old != entry["payload"]:
        clear_record()


def set_json_result(entry, origin):
    st.session_state.pop("last_job_issue", None)
    st.session_state.update(json_entry=entry, json_result=json.dumps(entry["payload"], ensure_ascii=False, indent=4),
                            json_origin=origin, json_job_complete=True, json_job_key=entry["key"])


def page_blocks(text):
    markers = list(PAGE_MARKER_RE.finditer(text or ""))
    return {int(m.group(1)): text[m.start():markers[i + 1].start() if i + 1 < len(markers) else len(text)].strip()
            for i, m in enumerate(markers)}


def apply_text_edit(new_text):
    entry = copy.deepcopy(st.session_state["ocr_entry"])
    if not valid_ocr(new_text, 1, entry["pages"]):
        st.error("Keep every === PAGE n === marker exactly once, in page order. Your edits have not been applied.")
        return False
    before, after = page_blocks(entry["payload"]), page_blocks(new_text)
    for number, check in entry.get("page_checks", {}).items():
        if before.get(int(number)) != after.get(int(number)) and isinstance(check, dict):
            check.pop("reviewed_digest", None)
            check.pop("reviewed_at", None)
    entry["payload"], entry["payload_digest"] = new_text, stable_digest(new_text)
    entry["edited"] = True
    entry["created_at"] = datetime.now(timezone.utc).isoformat()
    if not save_entry(entry):
        st.session_state["save_warning"] = "Edits are available in this session. Download the edited text because the server could not save it."
    st.session_state.setdefault("imported_cache", {})[cache_relative("ocr", entry["key"])] = entry
    set_ocr_result(entry, st.session_state.get("ocr_filename", "meeting") + ".pdf", "Edited text")
    st.session_state["flash"] = "Edits saved. The next meeting record will use the corrected text."
    return True


def page_needs_review(block, check):
    return "[?]" in block or (isinstance(check, dict) and check.get("needs_review", False)
                              and check.get("reviewed_digest") != stable_digest(block))


def mark_page_reviewed(page):
    entry = copy.deepcopy(st.session_state["ocr_entry"])
    check = entry.setdefault("page_checks", {}).setdefault(str(page), {})
    check["reviewed_digest"] = stable_digest(page_blocks(entry["payload"])[page])
    check["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    if not save_entry(entry):
        st.session_state["save_warning"] = "Review status is available in this session. Download the reading checks to keep a copy."
    st.session_state["ocr_entry"] = entry
    st.session_state.setdefault("imported_cache", {})[cache_relative("ocr", entry["key"])] = entry
    st.session_state["flash"] = f"Page {page} marked as checked against the PDF."
    if "[?]" in page_blocks(entry["payload"])[page]:
        st.session_state["flash"] += " Remaining [?] markers stay on the review list."


def render_sidebar(busy):
    with st.sidebar:
        st.markdown('<div class="ec-sidebar-title">Document options</div>', unsafe_allow_html=True)
        st.caption("Choose your document type and reading preferences.")
        st.divider()
        kind = st.radio("What kind of document is this?", ["Modern printed document", "Old, faded or handwritten document"], disabled=busy)
        old = kind.startswith("Old")
        st.caption("One page per request helps keep difficult names and numbers in context." if old else "Digital text is read locally when suitable. Scans are read by Gemini.")
        accuracy = st.radio("Reading mode", ["Accuracy first (recommended)", "Faster reading"], index=1, disabled=busy).startswith("Accuracy")
        if accuracy:
            st.caption("Scans are checked twice, with a third reading on disagreement. Slower; uses 2–3 AI requests per page, plus any retries. Conflicting details stay visible for review.")
        reuse = st.checkbox("Reuse saved results", value=True, disabled=busy,
                            help="Return saved results for the same document and settings. Turn off for a fresh run; this may use Gemini credits.")
        with st.expander("Advanced settings", expanded=False):
            model = st.text_input("Gemini model", value=MODEL_NAME, disabled=busy).strip()
            mode = st.radio("OCR processing method", ["High-DPI page images (recommended)", "Raw PDF chunks"], disabled=busy)
            dpi = st.slider("Image quality (DPI)", 200, 400, OCR_IMAGE_DPI, 50, disabled=busy)
            chunk_size = st.slider("Pages read at a time", 1, 40, 1 if old or accuracy else 4, key=f"batch_{old}_{accuracy}", disabled=busy or accuracy,
                                   help="Accuracy first reads one page at a time to reduce omissions and name mixing.")
            layer = st.checkbox("Use embedded PDF text when available", True, disabled=busy)
            strong = st.checkbox("Stronger contrast for badly faded scans", False, disabled=busy,
                                   help="Try this only if gentle cleanup leaves text unreadable. Strong contrast can remove faint strokes.")
            care = st.selectbox("Reading care", ["Quick transcription", "More reasoning"], index=1 if old or accuracy else 0, key=f"care_{old}_{accuracy}", disabled=busy,
                                help="More reasoning may help difficult pages and can take longer. Always review uncertain names and numbers.")
            cross_check = st.checkbox("Cross-check suspicious page boundaries", True, disabled=busy or accuracy,
                                     help="Retains isolated re-reading for pages that may have lost an item number. May make extra Gemini requests.")
            chars = st.slider("Text size per request", 30_000, 100_000, JSON_CHUNK_CHARS, 10_000, disabled=busy)
            workers = st.slider("Parallel requests", 1, 24, DEFAULT_MAX_WORKERS, disabled=busy,
                                help="Four is a conservative cloud default. Higher values use more memory; the request limit still applies.")
            rpm = st.number_input("Request limit per minute", 1, 1000, DEFAULT_SAFE_RPM, disabled=busy)
        if not api_keys:
            st.info("Saved results and suitable digital PDFs work now. New AI processing needs an API key in Streamlit Secrets.")
        st.caption("Progress is saved on this server. Keep downloaded copies of your text and meeting record; cloud storage can reset.")
    base = {"model": model, "workers": int(workers), "safe_rpm": int(rpm)}
    ocr = dict(base, input_mode="images" if mode.startswith("High") else "pdf", dpi=dpi, chunk_size=chunk_size,
               use_text_layer=layer, preprocess="strong" if strong else ("degraded" if old else "standard"),
               thinking="low" if care == "More reasoning" else "minimal", cross_check=cross_check)
    if accuracy:
        ocr.update(accuracy_strategy=ACCURACY_STRATEGY, chunk_size=1)
    js = dict(base, chunk_chars=chars)
    return reuse, ocr, js


# Older installations can still run the app without upgrading dependencies.
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)


def progress_fragment(function):
    return _fragment(run_every=1.0)(function) if _fragment else function


@progress_fragment
def render_progress():
    job = get_active_job()
    if job is None:
        return
    snap = job.snapshot()
    elapsed = (snap["finished"] or time.monotonic()) - snap["started"]
    unit = "pages" if snap["stage"] == "ocr" else "sections"
    if snap["status"] == "running":
        st.progress(min(1., snap["done"] / max(1, snap["total"])), text=snap["message"])
        details = f"{snap['done']} / {snap['total'] or '…'} {unit} · {int(elapsed)//60}m {int(elapsed)%60:02d}s elapsed · {snap['restored']} restored"
        if snap["stage"] == "ocr":
            details += f" · {snap['local']} read locally"
        st.caption(details)
        if job.pause.is_set():
            st.info("Pausing after active batches finish. Completed work will be kept.")
        elif st.button("Pause after current batches", key="pause_job"):
            job.pause.set()
            st.rerun()
        if _fragment is None and st.button("Refresh progress", key="refresh_progress"):
            st.rerun()
        return
    if snap["status"] == "complete":
        if snap["stage"] == "ocr":
            set_ocr_result(snap["result"], st.session_state.get("uploaded_name", "meeting.pdf"), "New extraction")
        else:
            set_json_result(snap["result"], "New record")
        st.session_state["flash"] = f"{'Text' if snap['stage'] == 'ocr' else 'Meeting record'} ready in {elapsed:.1f}s."
    else:
        st.session_state["last_job_issue"] = {"stage": snap["stage"], "done": snap["done"], "total": snap["total"], "errors": snap["errors"], "status": snap["status"]}
        if snap["stage"] == "ocr" and snap["partial"]:
            st.session_state["partial_ocr"] = snap["partial"]
    st.session_state["job_notes"] = snap["notes"]
    st.session_state.pop("active_job_id", None)
    st.rerun()


def legacy_ocr_entry(data, settings, page_count):
    """Recognize the previous version's default cache without masking setting changes."""
    if settings.get("accuracy_strategy"):
        return None  # An earlier single reading is not a verified extraction.
    expected = {"model": MODEL_NAME, "input_mode": "images", "dpi": OCR_IMAGE_DPI, "chunk_size": 4,
                "preprocess": "standard", "use_text_layer": True, "thinking": "minimal", "cross_check": True}
    if any(settings[k] != v for k, v in expected.items()):
        return None
    path = CACHE_ROOT / "ocr" / f"{stable_digest(data)}.txt"
    try:
        if path.stat().st_size > MAX_CACHE_BYTES:
            return None
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if not valid_ocr(text, 1, page_count):
        return None
    key = job_identity("ocr", stable_digest(data), settings)
    return make_entry("ocr", key, stable_digest(data), settings, text, pages=page_count,
                      notes=["Restored from the earlier app. Its original processing settings were not recorded; turn off Reuse saved results for a fresh extraction."], legacy=True)


def legacy_json_entry(text, settings):
    if settings["model"] != MODEL_NAME or settings["chunk_chars"] != JSON_CHUNK_CHARS:
        return None
    digest = stable_digest({"version": "meeting-record-v1", "source_text": text, "roster_names": settings["roster"]})
    value = read_json_file(CACHE_ROOT / "json" / f"{digest}.json")
    if not isinstance(value, dict) or not all(isinstance(value.get(k), list) for k in ("agenda", "presentees")):
        return None
    key = job_identity("json", stable_digest(text.encode("utf-8")), settings)
    entry = make_entry("json", key, stable_digest(text.encode("utf-8")), settings, value, legacy=True)
    return entry if entry_valid(entry, "json", key) else None


def render_ocr_review(pdf_data, busy):
    entry = st.session_state.get("ocr_entry")
    if not entry:
        return
    text = entry["payload"]
    blocks = page_blocks(text)
    uncertain = [n for n, value in blocks.items() if "[?]" in value]
    checks = entry.get("page_checks", {})
    flagged = [n for n, value in blocks.items() if page_needs_review(value, checks.get(str(n), {}))]
    cols = st.columns(3)
    cols[0].metric("Pages read", entry["pages"])
    cols[1].metric("Characters", f"{len(text):,}")
    cols[2].metric("Pages to review", len(flagged))
    st.caption(f"{st.session_state.get('ocr_origin', 'Saved result')} · {'Edited transcription' if entry.get('edited') else 'Ready to review'}")
    if uncertain:
        st.warning("Review [?] on page(s): " + ", ".join(map(str, uncertain)) + ". The app kept uncertainty visible instead of filling it in.")
    disagreements = [n for n in flagged if checks.get(str(n), {}).get("status") == "disagreement"]
    if disagreements:
        st.warning("Independent readings disagree on page(s): " + ", ".join(map(str, disagreements)) + ". Check names, numbers and missing text against the original.")
    if entry["settings"].get("accuracy_strategy"):
        agreed = sum(c.get("status") == "agreement" for c in checks.values() if isinstance(c, dict))
        st.caption(f"{agreed} page(s) have matching AI readings. Agreement is a review aid, not an accuracy percentage. Embedded digital text is extracted directly.")
    issues = ocr_consistency_report(text) + [i["message"] for i in page_boundary_item_report(text)]
    if issues or entry.get("notes"):
        with st.expander(f"Review notes ({len(issues) + len(entry.get('notes', []))})"):
            for issue in issues + entry.get("notes", []):
                st.write("• " + issue)
    with st.expander("Review and edit extracted text", expanded=True):
        view = st.radio("Review mode", ["Page by page", "Complete text"], horizontal=True, disabled=busy)
        if view == "Page by page":
            only_flagged = st.checkbox("Show only pages needing review", False, disabled=busy or not flagged)
            options = flagged if only_flagged and flagged else list(blocks)
            page = st.selectbox("Page", options, format_func=lambda n: f"Page {n}" + (" · needs review" if n in flagged else ""), disabled=busy)
            left, right = st.columns([1, 1])
            with left:
                st.caption("Original document")
                if pdf_data:
                    try:
                        components.html(pdf_viewer_html(pdf_preview(pdf_data, page), page),
                                        height=570, scrolling=False)
                    except Exception:
                        st.info("Page preview is unavailable; check the original PDF.")
            with right:
                st.caption("Extracted text · save edits before changing pages")
                with st.form(f"page_edit_{page}_{st.session_state.get('edit_revision', 0)}"):
                    edited = st.text_area("Page text", value=blocks[page], height=520, disabled=busy)
                    if st.form_submit_button("Save page edits", type="primary", disabled=busy):
                        updated = replace_page_text(text, page, edited)
                        if apply_text_edit(updated):
                            st.rerun()
            check = checks.get(str(page), {})
            if check:
                st.caption(check.get("reason", ""))
            if check.get("readings"):
                with st.expander("Compare independent readings", expanded=page in flagged):
                    if check.get("number_difference"):
                        st.warning("Numbers differ between readings. Verify every affected date and number in the PDF.")
                    if check.get("differences"):
                        st.table(check["differences"])
                        st.caption("Up to 12 differences between readings 1 and 2 are shown. Full readings are below.")
                    for i, reading in enumerate(check["readings"], 1):
                        st.text_area(reading["label"], value=reading["text"], height=180, disabled=True,
                                     key=f"alternative_{entry['key']}_{page}_{i}")
                        if st.button(f"Use reading {i} for page {page}", key=f"use_reading_{page}_{i}", disabled=busy):
                            if apply_text_edit(replace_page_text(text, page, reading["text"])):
                                st.rerun()
            if check and page in flagged:
                if st.button("I checked this page against the PDF", key=f"confirm_page_{page}", disabled=busy):
                    mark_page_reviewed(page)
                    st.rerun()
        else:
            with st.form("full_text_edit"):
                edited = st.text_area("Combined OCR output", value=text, height=440, disabled=busy)
                if st.form_submit_button("Apply edits", type="primary", disabled=busy):
                    if apply_text_edit(edited):
                        st.rerun()
    a, b = st.columns(2)
    base = st.session_state.get("ocr_filename", "meeting")
    a.download_button("Download text (.txt)", text, file_name=f"{base}_ocr.txt", mime="text/plain", use_container_width=True)
    b.download_button("Download Markdown (.md)", text, file_name=f"{base}_ocr.md", mime="text/markdown", use_container_width=True)
    if checks:
        st.download_button("Download reading checks", json.dumps(checks, ensure_ascii=False, indent=2),
                           file_name=f"{base}_reading_checks.json", mime="application/json")


def render_record():
    entry = st.session_state.get("json_entry")
    if not entry:
        return
    meeting = entry["payload"]
    st.subheader("Meeting record")
    if entry.get("incomplete_source"):
        st.warning("This record was created from incomplete source text; missing pages are not represented.")
    st.caption(st.session_state.get("json_origin", "Saved result"))
    a, b, c = st.columns(3)
    a.metric("Attendees", len(meeting.get("presentees", [])))
    b.metric("Agenda items", len(meeting.get("agenda", [])))
    issues = meeting_quality_report(meeting)
    c.metric("Checks to review", len(issues))
    if issues:
        with st.expander("Check these details before using the record", expanded=True):
            for issue in issues:
                st.write("• " + issue)
    else:
        st.success("Structural checks passed. Review names, dates, and numbers against the source.")
    for message in entry.get("corrections", []):
        st.info("Roster correction: " + message)
    if entry.get("stitch_notes"):
        with st.expander("Page-boundary continuations"):
            for note in entry["stitch_notes"]:
                st.write(note)
    with st.expander("Preview the meeting record", expanded=True):
        view = st.radio("Record view", ["Summary", "Raw text", "Collapsible tree"], horizontal=True)
        if view == "Summary":
            st.write(meeting.get("title") or "Meeting title not found")
            st.caption(str(meeting.get("date") or "Date not found"))
            if meeting.get("presentees"):
                st.dataframe(meeting["presentees"], use_container_width=True, hide_index=True)
            st.caption("Download the meeting record to get every agenda item, formatted table, and resolution.")
        elif view == "Collapsible tree":
            with scroll_box(460):
                st.json(meeting, expanded=False)
        else:
            code_block(st.session_state["json_result"], "json", 460)
    st.download_button("Download meeting record", st.session_state["json_result"],
                       file_name=st.session_state.get("ocr_filename", "meeting") + ".json", mime="application/json", use_container_width=True)


def render_app():
    st.markdown(APP_CSS, unsafe_allow_html=True)
    job = get_active_job()
    busy = job is not None  # commit a just-finished job before enabling another one
    reuse, ocr_settings, json_settings = render_sidebar(busy)
    status = "OCR service ready" if api_keys else "Local reading available"
    badge_class = "" if api_keys else "off"
    st.markdown(f'''<div class="ec-header"><div class="ec-brand"><div class="ec-logo" aria-hidden="true">BUET</div><div><h1>BUET E-Council Document Processor</h1><div class="ec-brand-subtitle">Academic Council document workspace</div></div></div><span class="ec-pill {badge_class}"><span class="ec-dot" aria-hidden="true"></span>{status}</span></div><div class="ec-intro"><strong>Turn a scanned meeting document into text and a structured record.</strong><br><b>Step 1</b> — upload your PDF and press <b>Read the document</b>, then review the extracted text.<br><b>Step 2</b> — press <b>Create the meeting record</b> to prepare your download.</div>''', unsafe_allow_html=True)
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    if st.session_state.get("save_warning"):
        st.warning(st.session_state.pop("save_warning"))
    if st.session_state.get("last_job_issue"):
        issue = st.session_state["last_job_issue"]
        st.warning(f"{'Reading' if issue['stage'] == 'ocr' else 'Record generation'} paused with {issue['done']} of {issue['total']} completed. Keep Reuse saved results on and press the step button to continue.")
        if issue["errors"]:
            with st.expander("What happened"):
                for part, error in issue["errors"].items():
                    st.write(f"Section {part}: {error}")
    if st.session_state.get("job_notes"):
        with st.expander("Processing notes"):
            for note in st.session_state["job_notes"]:
                st.write(note)
    if job is not None and job.stage == "ocr":
        render_progress()
    st.markdown('<div class="ec-section"><span class="ec-number">1</span><h2>Extract text from the document</h2></div>', unsafe_allow_html=True)
    st.caption("Upload the PDF you want to process. Previously saved results open without another AI request.")
    uploaded = st.file_uploader("Choose a PDF document", type=["pdf"], key="source_pdf", disabled=busy)
    pdf_data = uploaded.getvalue() if uploaded is not None else None
    digest = stable_digest(pdf_data) if pdf_data is not None else None
    if digest != st.session_state.get("selected_pdf_digest") and not busy:
        for name in ("ocr_entry", "ocr_result", "ocr_editor", "partial_ocr", "last_job_issue", "job_notes", "ocr_metadata", "ocr_filename"):
            st.session_state.pop(name, None)
        clear_record()
        st.session_state["selected_pdf_digest"] = digest
    page_count = None
    if uploaded is not None:
        st.session_state["uploaded_name"] = uploaded.name
        try:
            if "ocr_metadata" not in st.session_state:
                st.session_state["ocr_metadata"] = pdf_metadata(pdf_data)
            page_count = st.session_state["ocr_metadata"]["pages"]
        except Exception as error:
            st.error("Cannot open this PDF. " + safe_error(error))
        if page_count:
            key = job_identity("ocr", digest, ocr_settings)
            cached = load_entry("ocr", key, st.session_state.get("imported_cache")) if reuse else None
            if cached is None and reuse:
                cached = legacy_ocr_entry(pdf_data, ocr_settings, page_count)
            if st.session_state.get("ocr_entry", {}).get("key") not in (None, key):
                st.info("The text below was read with earlier settings. Press Read the document to use your current settings.")
            label = "Saved text is ready" if cached else "Ready to read"
            st.markdown(f'<div class="ec-file"><b>{html.escape(uploaded.name)}</b><small>{page_count} pages · {len(pdf_data)/1048576:.1f} MB · {label}</small></div>', unsafe_allow_html=True)
            issue = st.session_state.get("last_job_issue", {})
            button_label = "Continue reading" if issue.get("stage") == "ocr" and reuse else "Read the document"
            if st.button(button_label, type="primary", use_container_width=True, disabled=busy, key="read_document"):
                st.session_state.pop("last_job_issue", None)
                if cached:
                    set_ocr_result(cached, uploaded.name, "Saved result")
                    st.session_state["flash"] = "Saved text loaded. Review it below, then create the meeting record."
                else:
                    new_job = ProcessingJob("ocr", key, digest, ocr_settings)
                    try:
                        st.session_state["active_job_id"] = start_job(new_job, run_ocr_job, pdf_data, reuse, api_keys)
                    except ValueError as error:
                        st.error(str(error))
                        return
                st.rerun()
    render_ocr_review(pdf_data, busy)
    if not st.session_state.get("ocr_entry") and st.session_state.get("partial_ocr"):
        st.download_button("Download completed text so far", st.session_state["partial_ocr"], "partial_ocr.txt", "text/plain")
    st.divider()
    st.markdown('<div class="ec-section"><span class="ec-number">2</span><h2>Create the meeting record</h2></div>', unsafe_allow_html=True)
    source_choice = st.radio("Choose the text source", ["Use the text from Step 1", "Upload or paste my own text"], horizontal=True, disabled=busy)
    source = st.session_state.get("ocr_result", "")
    if source_choice.startswith("Upload"):
        txt = st.file_uploader("Upload extracted text (.txt / .md)", type=["txt", "md"], key="txt_up", disabled=busy)
        if txt is not None:
            signature = stable_digest(txt.getvalue())
            if signature != st.session_state.get("manual_json_upload_signature"):
                st.session_state["manual_json_text"] = txt.getvalue().decode("utf-8", errors="replace")
                st.session_state["manual_json_upload_signature"] = signature
        source = st.text_area("Paste or edit the extracted meeting text here", key="manual_json_text", height=260, disabled=busy)
    with st.expander("Optional: improve faculty-name spelling"):
        st.caption("Use a trusted roster to correct close, unambiguous name matches. Applied corrections are listed with the finished record.")
        roster_file = st.file_uploader("Upload roster (.txt / .sql)", type=["txt", "sql"], key="roster_up", disabled=busy)
        if roster_file is not None:
            signature = stable_digest(roster_file.getvalue())
            if signature != st.session_state.get("roster_signature"):
                st.session_state["roster_text"] = roster_file.getvalue().decode("utf-8", errors="replace")
                st.session_state["roster_signature"] = signature
        roster = st.text_area("Faculty names, one per line, or SQL INSERT statements", key="roster_text", height=130, disabled=busy)
    json_settings["roster"] = parse_name_roster(roster)
    if json_settings["roster"]:
        st.caption(f"{len(json_settings['roster'])} roster names loaded")
    full_text = clean_bengali_ocr_text(source)
    source_digest = stable_digest(full_text.encode("utf-8"))
    json_key = job_identity("json", source_digest, json_settings)
    if st.session_state.get("json_entry", {}).get("key") not in (None, json_key) and not busy:
        clear_record()  # never show a previous document's record under new text/settings
    ready = bool(full_text.strip())
    incomplete = "COULD NOT BE READ — press Continue reading" in full_text
    allow_partial = False
    if incomplete:
        st.warning("This text has missing pages. Finish reading it for a complete meeting record.")
        allow_partial = st.checkbox("Create a record from this incomplete text", False, disabled=busy)
    if not ready:
        st.info("Complete Step 1 above, or upload/paste text to begin.")
    json_cached = load_entry("json", json_key, st.session_state.get("imported_cache")) if ready and reuse else None
    if json_cached is None and ready and reuse:
        json_cached = legacy_json_entry(full_text, json_settings)
    if json_cached:
        st.caption("A saved meeting record is ready for this text and roster.")
    button_label = "Continue building the record" if st.session_state.get("last_job_issue", {}).get("stage") == "json" and reuse else "Create the meeting record"
    if st.button(button_label, type="primary", use_container_width=True, disabled=busy or not ready or (incomplete and not allow_partial), key="create_record"):
        st.session_state.pop("last_job_issue", None)
        if json_cached:
            set_json_result(json_cached, "Saved result")
            st.session_state["flash"] = "Saved meeting record loaded."
        elif not api_keys:
            st.error("Add GEMINI_API_KEYS in Streamlit Secrets to generate a new record.")
            return
        else:
            st.progress(0.0, text="Starting the meeting record…")
            new_job = ProcessingJob("json", json_key, source_digest, json_settings)
            try:
                st.session_state["active_job_id"] = start_job(new_job, run_json_job, full_text, reuse, api_keys)
            except ValueError as error:
                st.error(str(error))
                return
        st.rerun()
    if job is not None and job.stage == "json":
        render_progress()  # Keep record progress beside its action and result.
    render_record()



if __name__ == "__main__":
    render_app()
