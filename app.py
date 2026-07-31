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
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

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
    _secret_keys = list(st.secrets.get("GEMINI_API_KEYS", []))
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
DEFAULT_MAX_WORKERS = 18  # concurrency only; the limiter still controls request starts
DEFAULT_SAFE_RPM = 15
MAX_INLINE_MB = 19  # inline request payload safety limit (~20 MB hard cap)
OCR_THINKING_LEVEL = "minimal"
JSON_THINKING_LEVEL = "low"
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
    "and it is NEVER re-checked against this page afterwards. A name that is fluent, plausible and "
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
    "knowledge of Bengali names, departments and places. Do not compare entries against each "
    "other in EITHER direction. Never change a value so that two entries agree. Never carry a value "
    "FORWARD from the entry above, and never pull a value BACKWARD from the entry below — a "
    "neighbouring line is not evidence about this line, whichever side it sits on. Every entry is "
    "decided only by its own printed lines. This single rule prevents most serious errors.",

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
    "\n\n=== WORKED EXAMPLE — FORMAT ONLY ==="
    "\nEverything in «guillemets» below is a PLACEHOLDER. These are not real names, departments "
    "or faculties, they must NEVER appear in your output, and nothing in them tells you anything "
    "about the document you are reading. The example teaches SHAPE, not content."
    "\nA page printing this attendee list:"
    "\n     ৩। অধ্যাপক ডঃ «নাম-এক»              সদস্য"
    "\n        প্রধান, «বিভাগ-এক»"
    "\n     ৪। অধ্যাপক ডঃ «নাম-দুই»             সদস্য"
    "\n        ডীন, «অনুষদ-দুই»"
    "\n     ৫। 〃                                সদস্য"
    "\nis transcribed as exactly this and nothing else:"
    "\n=== PAGE 7 ==="
    "\n৩। অধ্যাপক ডঃ «নাম-এক» সদস্য"
    "\nপ্রধান, «বিভাগ-এক»"
    "\n৪। অধ্যাপক ডঃ «নাম-দুই» সদস্য"
    "\nডীন, «অনুষদ-দুই»"
    "\n৫। 〃 সদস্য"
    "\nWhat this shows: entry ৩ takes its affiliation from the line printed under ৩, and entry ৪ "
    "from the line printed under ৪ — the two are read completely independently; the role label "
    "stays on the same line as the person; the ditto mark is transcribed as printed and never "
    "expanded; the page marker sits on its own line; no commentary is added."
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
    "\nL4. In an attendee or member list each numbered entry is ONE record. Work the list ONE ENTRY "
    "AT A TIME using this procedure: (a) locate this entry's printed number; (b) locate the NEXT "
    "entry's printed number; (c) everything between those two numbers — name, designation, "
    "department, office, role, whether on the number's own line or on the indented lines beneath "
    "it — belongs to THIS entry, and nothing outside that span does; (d) transcribe it "
    "left-to-right then top-to-bottom; (e) only then move on. An affiliation line such as "
    "'ডীন, ... অনুষদ' or 'প্রধান, ... বিভাগ' belongs to the entry whose number is printed "
    "directly above it — never to the entry before or after it. Do not keep one entry's "
    "affiliation in mind while writing another's."
    "\nL4b. LOOK-ALIKE NEIGHBOURS ARE THE HIGHEST-RISK MOMENT IN THE WHOLE DOCUMENT. In these "
    "lists, two entries standing next to each other often share a leading name word and share an "
    "affiliation template that differs by exactly ONE word — «পদ», «ক» অনুষদ printed directly "
    "above «পদ», «খ» অনুষদ, with «ক» and «খ» the only difference. That single differing word is "
    "the one you are most likely to get wrong, because everything around it matches. So before "
    "writing it, go back to the image and re-read THAT WORD on THIS entry's own line. Do not let "
    "the word you just wrote for the entry above supply it, and do not let the word you can "
    "already see on the entry below supply it. Every faculty and department name in the list is "
    "read independently, even when the surrounding words are identical."
    "\nL4c. THE ONE-HOLDER CONSTRAINT — this is a fact about the institution, not a guess. A "
    "faculty (অনুষদ) has exactly ONE ডীন, and a department (বিভাগ) has exactly ONE প্রধান. So if "
    "you are about to write the SAME faculty for two different ডীন entries, or the same "
    "department for two different প্রধান entries, that is a signal that you have mis-read one of "
    "them. Do NOT resolve it by deciding which one to change and inventing a difference. Resolve "
    "it by going back to the image and re-reading BOTH lines word by word before you write "
    "either one. If after looking again the page genuinely does print the same value twice, then "
    "write it twice — the page is always the authority. This constraint applies ONLY to ডীন and "
    "প্রধান affiliation lines. Ordinary members of a department repeat that department freely, "
    "and two members sharing a department is completely normal."
    "\nL5. Headings, titles, dates and any full-width text above the columns come before them; "
    "full-width text below comes after."
    "\nL6. Tables, tabular rows, aligned lists and forms are never split into columns. Keep each "
    "table together as a Markdown pipe table with one header row, one separator row and every "
    "data row, in printed row order."
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
    "\n  - every attendee's affiliation came from the line directly beneath that attendee's own number, and no affiliation was reused between entries;"
    "\n  - wherever two neighbouring entries have affiliations differing by a single word, that word was re-read separately for each of them;"
    "\n  - EVERY printed number was verified digit by digit directly from the page image a second time;"
    "\n  - this includes dates, proposal numbers, agenda numbers, serial numbers, student IDs, registration numbers, credit values, page numbers and list item numbers;"
    "\n  - no digit was inferred from context, sequence, neighbouring entries or what would look plausible;"
    "\n  - every date had its fields counted separately, with no digit lost against a danda;"
    "\n  - anything unreadable is marked [?] rather than filled in with a plausible guess;"
    "\n  - nothing has been added that is not on the page."
)


def build_chunk_prompt(handwritten: bool = True) -> str:
    """Assemble the OCR prompt. Handwriting and typewriter rules are included
    only for old/handwritten documents, where they earn their tokens."""
    parts = [PROMPT_MISSION, *PROMPT_NON_NEGOTIABLE, PROMPT_EXAMPLE,
             PROMPT_COMPLETENESS, PROMPT_LAYOUT, PROMPT_BENGALI, PROMPT_SIGNATURE]
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
    prompt = build_chunk_prompt(handwritten=(preprocess == "degraded"))
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
st.set_page_config(
    page_title="BUET E-Council Document Processor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
        /* ---------------------------------------------------------
           BUET E-COUNCIL — COMPACT RESPONSIVE LIGHT THEME
           --------------------------------------------------------- */
        :root {
            color-scheme: light !important;
            --ec-red: #a80714;
            --ec-red-dark: #820810;
            --ec-red-soft: #fff3f4;
            --ec-bg: #f8f5f5;
            --ec-card: #ffffff;
            --ec-border: #ead9da;
            --ec-text: #2f2526;
            --ec-muted: #746466;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--ec-bg) !important;
            color: var(--ec-text) !important;
        }

        [data-testid="stAppViewContainer"] > .main {
            background:
                radial-gradient(circle at 92% 0%, rgba(168, 7, 20, 0.04), transparent 25rem),
                var(--ec-bg) !important;
        }

        [data-testid="stHeader"] {
            height: 3.25rem !important;
            background: rgba(255, 255, 255, 0.98) !important;
            border-top: 3px solid var(--ec-red) !important;
            border-bottom: 1px solid var(--ec-border) !important;
            box-shadow: 0 2px 10px rgba(69, 18, 23, 0.05) !important;
        }

        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
            color: #7f1019 !important;
        }

        .block-container {
            width: min(100%, 1160px) !important;
            max-width: 1160px !important;
            /* Keep the branded header below Streamlit's fixed top toolbar. */
            padding: 4.35rem 1.35rem 2rem !important;
        }

        .main [data-testid="stVerticalBlock"] {
            gap: 0.72rem !important;
        }

        /* Compact, readable typography */
        .stApp h1, .stApp h2, .stApp h3,
        .stApp h4, .stApp h5, .stApp h6 {
            color: var(--ec-text) !important;
            letter-spacing: -0.025em;
        }

        .stApp h1 {
            font-size: clamp(1.8rem, 2.5vw, 2.35rem) !important;
        }

        .stApp h2 {
            font-size: clamp(1.45rem, 2vw, 1.9rem) !important;
            margin-top: 0.65rem !important;
            margin-bottom: 0.2rem !important;
        }

        .stApp h3 {
            font-size: clamp(1.15rem, 1.5vw, 1.4rem) !important;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"],
        .stCaption {
            color: #67595b !important;
        }

        [data-testid="stCaptionContainer"],
        .stCaption {
            font-size: 0.86rem !important;
            line-height: 1.45 !important;
        }

        /* Ensure plain Streamlit status text stays dark on the light theme. */
        [data-testid="stText"],
        [data-testid="stText"] p,
        [data-testid="stText"] span,
        [data-testid="stText"] div {
            color: #2f2526 !important;
        }

        /* Branded page header */
        .ec-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.85rem;
            padding: 0.72rem 0.9rem;
            margin: 0 0 0.85rem 0;
            background: var(--ec-card);
            border: 1px solid var(--ec-border);
            border-top: 3px solid var(--ec-red);
            border-radius: 0 0 12px 12px;
            box-shadow: 0 5px 16px rgba(74, 20, 25, 0.055);
        }

        .ec-brand-wrap {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            min-width: 0;
        }

        .ec-logo-box {
            width: 40px;
            height: 40px;
            display: grid;
            place-items: center;
            flex: 0 0 40px;
            border-radius: 10px;
            background: var(--ec-red);
            color: #ffffff !important;
            font-weight: 800;
            font-size: 0.67rem;
            letter-spacing: 0.04em;
            box-shadow: 0 4px 10px rgba(168, 7, 20, 0.18);
        }

        .ec-brand-title {
            margin: 0;
            color: var(--ec-red) !important;
            font-size: clamp(1.12rem, 1.8vw, 1.38rem);
            line-height: 1.08;
            font-weight: 800;
            letter-spacing: 0.012em;
        }

        .ec-brand-subtitle {
            margin-top: 0.13rem;
            color: var(--ec-muted) !important;
            font-size: 0.8rem;
        }

        .ec-status {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.38rem 0.62rem;
            background: var(--ec-red-soft);
            border: 1px solid #efd1d4;
            border-radius: 999px;
            color: #7f1019 !important;
            font-size: 0.73rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .ec-status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #16a06a;
            box-shadow: 0 0 0 3px rgba(22, 160, 106, 0.12);
        }

        .app-intro {
            padding: 0.78rem 0.95rem;
            margin: 0 0 1rem 0;
            background: var(--ec-card);
            border: 1px solid var(--ec-border);
            border-left: 4px solid var(--ec-red);
            border-radius: 10px;
            box-shadow: 0 4px 13px rgba(74, 20, 25, 0.035);
        }

        .app-intro strong {
            color: #7f1019 !important;
        }

        .app-intro p {
            margin: 0.05rem 0 0.18rem 0;
            line-height: 1.4;
        }

        /* Custom compact step headings */
        .ec-section-heading {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 1.05rem 0 0.18rem;
        }

        .ec-section-badge {
            display: grid;
            place-items: center;
            width: 34px;
            height: 34px;
            flex: 0 0 34px;
            border-radius: 9px;
            background: linear-gradient(180deg, #b51220, #8f0b17);
            color: #ffffff !important;
            font-size: 1rem;
            font-weight: 800;
            box-shadow: 0 4px 10px rgba(168, 7, 20, 0.17);
        }

        .ec-section-title {
            color: var(--ec-text) !important;
            font-size: clamp(1.45rem, 2.15vw, 1.95rem);
            line-height: 1.15;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .ec-section-copy {
            margin: 0 0 0.72rem 2.95rem;
            color: #7d7072 !important;
            font-size: 0.87rem;
            line-height: 1.45;
        }

        /* Sidebar: narrower on desktop, automatic overlay on small screens */
        [data-testid="stSidebar"] {
            min-width: 270px !important;
            max-width: 270px !important;
            background: #fffafa !important;
            border-right: 1px solid var(--ec-border) !important;
        }

        [data-testid="stSidebar"] > div {
            background: #fffafa !important;
            padding-top: 0.75rem !important;
        }

        [data-testid="stSidebar"] * {
            color: #3d3031 !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #8f0b17 !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--ec-border) !important;
            margin: 0.8rem 0 !important;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            font-size: 0.8rem !important;
        }

        /* Cards, expanders and metrics */
        [data-testid="stExpander"],
        [data-testid="stForm"],
        [data-testid="stMetric"],
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--ec-card) !important;
            border-color: var(--ec-border) !important;
            border-radius: 10px !important;
            box-shadow: 0 3px 10px rgba(74, 20, 25, 0.032) !important;
        }

        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {
            background: var(--ec-card) !important;
            color: #3d3031 !important;
        }

        [data-testid="stExpander"] summary {
            padding-top: 0.65rem !important;
            padding-bottom: 0.65rem !important;
        }

        [data-testid="stMetricValue"] {
            color: #8f0b17 !important;
            font-size: 1.05rem !important;
        }

        [data-testid="stMetricLabel"] {
            color: var(--ec-muted) !important;
            font-size: 0.78rem !important;
        }

        /* Inputs */
        input, textarea,
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="base-input"],
        [data-testid="stNumberInput"] input {
            background: #ffffff !important;
            color: var(--ec-text) !important;
            border-color: #d9c5c7 !important;
        }

        input::placeholder, textarea::placeholder {
            color: #9b8a8c !important;
            opacity: 1 !important;
        }

        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [role="listbox"] {
            background: #ffffff !important;
            color: var(--ec-text) !important;
        }

        [role="option"]:hover {
            background: #fbecee !important;
        }

        /* File uploader: shorter and better proportioned */
        div[data-testid="stFileUploader"] {
            padding: 0.15rem 0 !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            min-height: 72px !important;
            background: #ffffff !important;
            border: 1.5px dashed #c98f95 !important;
            border-radius: 10px !important;
            padding: 0.65rem 0.8rem !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: #57494b !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background: #fff7f7 !important;
            color: #8f0b17 !important;
            border: 1px solid #c98f95 !important;
            min-height: 36px !important;
        }

        [data-testid="stFileUploaderDropzone"] button:hover {
            background: #fbe8ea !important;
            border-color: var(--ec-red) !important;
        }

        /* Buttons */
        [data-testid="stBaseButton-primary"],
        div.stButton > button[kind="primary"],
        div.stDownloadButton > button[kind="primary"] {
            min-height: 40px !important;
            background: var(--ec-red) !important;
            color: #ffffff !important;
            border: 1px solid var(--ec-red) !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 10px rgba(168, 7, 20, 0.15) !important;
        }

        [data-testid="stBaseButton-primary"] *,
        div.stButton > button[kind="primary"] *,
        div.stDownloadButton > button[kind="primary"] * {
            color: #ffffff !important;
        }

        [data-testid="stBaseButton-primary"]:hover,
        div.stButton > button[kind="primary"]:hover,
        div.stDownloadButton > button[kind="primary"]:hover {
            background: var(--ec-red-dark) !important;
            border-color: var(--ec-red-dark) !important;
        }

        [data-testid="stBaseButton-secondary"],
        div.stButton > button:not([kind="primary"]),
        div.stDownloadButton > button:not([kind="primary"]) {
            min-height: 38px !important;
            background: #ffffff !important;
            color: #8f0b17 !important;
            border: 1px solid #d7aeb2 !important;
            border-radius: 8px !important;
            font-weight: 650 !important;
        }

        [data-testid="stBaseButton-secondary"] *,
        div.stButton > button:not([kind="primary"]) *,
        div.stDownloadButton > button:not([kind="primary"]) * {
            color: #8f0b17 !important;
        }

        /* Radio, checkbox, slider, progress */
        [data-baseweb="radio"] div[aria-checked="true"],
        [data-baseweb="checkbox"] div[aria-checked="true"],
        [role="checkbox"][aria-checked="true"] {
            background-color: var(--ec-red) !important;
            border-color: var(--ec-red) !important;
        }

        [data-testid="stSlider"] [role="slider"] {
            background: var(--ec-red) !important;
            border-color: var(--ec-red) !important;
        }

        [data-testid="stProgress"] > div > div > div > div {
            background-color: var(--ec-red) !important;
        }

        [data-testid="stAlert"] {
            background: #ffffff !important;
            color: #3d3031 !important;
            border: 1px solid var(--ec-border) !important;
            border-left: 4px solid var(--ec-red) !important;
        }

        [data-testid="stCode"], pre, code {
            background: #f8f1f2 !important;
            color: #4b3033 !important;
            border-color: var(--ec-border) !important;
        }

        hr {
            border: 0 !important;
            border-top: 1px solid #e5d4d6 !important;
            margin: 1.05rem 0 !important;
        }

        /* Keep highlighted/selected text readable in Safari and other browsers. */
        .stApp ::selection {
            background: #cfe3ff !important;
            color: #171717 !important;
            -webkit-text-fill-color: #171717 !important;
        }

        .stApp ::-moz-selection {
            background: #cfe3ff !important;
            color: #171717 !important;
        }

        /* Large desktop */
        @media (min-width: 1500px) {
            .block-container {
                max-width: 1240px !important;
            }
        }

        /* Laptop / tablet */
        @media (max-width: 1100px) {
            [data-testid="stSidebar"] {
                min-width: 245px !important;
                max-width: 245px !important;
            }

            .block-container {
                width: 100% !important;
                max-width: 100% !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }

        /* Mobile */
        @media (max-width: 760px) {
            .block-container {
                /* Streamlit keeps the toolbar fixed on narrow screens too. */
                padding: 4.1rem 0.75rem 1.5rem !important;
            }

            .ec-topbar {
                align-items: flex-start;
                flex-direction: column;
                padding: 0.65rem 0.75rem;
            }

            .ec-status {
                margin-left: 2.95rem;
            }

            .app-intro {
                padding: 0.7rem 0.8rem;
            }

            .ec-section-heading {
                gap: 0.52rem;
                margin-top: 0.85rem;
            }

            .ec-section-badge {
                width: 30px;
                height: 30px;
                flex-basis: 30px;
                border-radius: 8px;
                font-size: 0.9rem;
            }

            .ec-section-title {
                font-size: 1.35rem;
            }

            .ec-section-copy {
                margin-left: 0;
                font-size: 0.82rem;
            }

            [data-testid="stFileUploaderDropzone"] {
                min-height: 64px !important;
                padding: 0.55rem !important;
            }
        }

        /* ===========================================================
           THEME-PROOF READABILITY  (single-file: no config.toml needed)
           Streamlit paints widget labels, alert bodies, code tokens and
           expander contents from ITS OWN theme. If that theme resolves to
           dark — system preference, browser setting or host default — they
           render white-on-white against this light app. Rather than chase
           each element, force readable ink on everything first, then
           restore the few places that are meant to be light or coloured.
           ORDER MATTERS: the catch-all must come before the exceptions.
           =========================================================== */

        /* 1. Catch-all ink. `color` only — NOT -webkit-text-fill-color,
              which would also flatten colour emoji into dark silhouettes. */
        .stApp, .stApp *, [data-testid="stSidebar"] * {
            color: #2f2526 !important;
        }

        /* 2. Light surfaces, so forced-dark ink never lands on a dark box. */
        [data-testid="stJson"], [data-testid="stAlert"],
        [data-testid="stAlertContainer"], [data-testid="stNotification"],
        [data-testid="stExpander"], [data-testid="stExpanderDetails"],
        [data-testid="stDataFrame"], [data-testid="stTable"],
        [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
            background-color: #ffffff !important;
        }
        [data-testid="stCode"], pre, code {
            background-color: #f8f1f2 !important;
        }

        /* 3. Form controls need the fill colour too (Safari / autofill). */
        input, textarea,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-baseweb="base-input"] input {
            color: #2f2526 !important;
            -webkit-text-fill-color: #2f2526 !important;
            background-color: #ffffff !important;
        }
        input::placeholder, textarea::placeholder {
            color: #9b8a8c !important;
            -webkit-text-fill-color: #9b8a8c !important;
        }

        /* 4. EXCEPTIONS — everything that is meant to be light or coloured. */
        .ec-logo-box, .ec-logo-box *,
        .ec-section-badge, .ec-section-badge * {
            color: #ffffff !important;
        }
        .ec-brand-title { color: var(--ec-red) !important; }
        .ec-brand-subtitle { color: var(--ec-muted) !important; }
        .ec-status, .ec-status * { color: #7f1019 !important; }
        .app-intro strong { color: #7f1019 !important; }
        .ec-section-copy { color: #7d7072 !important; }
        [data-testid="stMetricValue"] { color: #8f0b17 !important; }
        [data-testid="stMetricLabel"] { color: var(--ec-muted) !important; }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 { color: #8f0b17 !important; }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        .stCaption { color: #67595b !important; }

        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-primary"] *,
        div.stButton > button[kind="primary"],
        div.stButton > button[kind="primary"] *,
        div.stDownloadButton > button[kind="primary"],
        div.stDownloadButton > button[kind="primary"] * {
            color: #ffffff !important;
        }
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-secondary"] *,
        div.stButton > button:not([kind="primary"]),
        div.stButton > button:not([kind="primary"]) *,
        div.stDownloadButton > button:not([kind="primary"]),
        div.stDownloadButton > button:not([kind="primary"]) * {
            color: #8f0b17 !important;
        }
        .stApp a, .stApp a * {
            color: #8f0b17 !important;
            text-decoration: underline;
        }
        [data-baseweb="tooltip"], [data-baseweb="tooltip"] * {
            background: #2f2526 !important;
            color: #ffffff !important;
        }

        /* 5. A disabled button must still read as disabled, not invisible. */
        button:disabled, button:disabled * {
            color: #9b8a8c !important;
            background: #f4eded !important;
            border-color: #e5d4d6 !important;
            cursor: not-allowed !important;
        }

        /* A plain, friendly help box */
        .ec-help {
            padding: 0.7rem 0.9rem;
            margin: 0.4rem 0 0.8rem 0;
            background: #fffdfd;
            border: 1px solid var(--ec-border);
            border-left: 4px solid #16a06a;
            border-radius: 10px;
            color: #3d3031 !important;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        /* ===========================================================
           6. ALWAYS-VISIBLE SCROLLBARS FOR PREVIEW AREAS
           macOS (and iOS) draw "overlay" scrollbars: invisible until
           the moment you actually scroll. A preview box therefore looks
           like a dead end — the reader sees the first screenful and has
           no signal that more text exists below. -webkit-appearance:none
           opts out of the overlay style and pins a real, permanent bar
           in the app's own colours. scrollbar-width/-color do the same
           on Firefox.
           =========================================================== */

        .stApp, .stApp *, [data-testid="stSidebar"] * {
            scrollbar-width: thin;
            scrollbar-color: #cf9aa0 #f4ebec;
        }

        .stApp ::-webkit-scrollbar,
        .stApp *::-webkit-scrollbar,
        [data-testid="stSidebar"] ::-webkit-scrollbar {
            -webkit-appearance: none !important;
            width: 11px !important;
            height: 11px !important;
        }

        .stApp ::-webkit-scrollbar-track,
        .stApp *::-webkit-scrollbar-track,
        [data-testid="stSidebar"] ::-webkit-scrollbar-track {
            background: #f4ebec !important;
            border-radius: 8px !important;
        }

        .stApp ::-webkit-scrollbar-thumb,
        .stApp *::-webkit-scrollbar-thumb,
        [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
            background: #cf9aa0 !important;
            border: 2px solid #f4ebec !important;
            border-radius: 8px !important;
        }

        .stApp ::-webkit-scrollbar-thumb:hover,
        .stApp *::-webkit-scrollbar-thumb:hover {
            background: var(--ec-red) !important;
        }

        .stApp ::-webkit-scrollbar-corner,
        .stApp *::-webkit-scrollbar-corner {
            background: #f4ebec !important;
        }

        /* Preview frames. Streamlit's own `height=` argument does the
           bounding and the scrolling now, so this only paints the frame —
           no CSS max-height, which would fight it and produce two nested
           scrollbars. */
        [data-testid="stCode"] {
            border: 1px solid var(--ec-border) !important;
            border-radius: 10px !important;
        }

        [data-testid="stCode"] pre {
            margin-bottom: 0 !important;
        }

        [data-testid="stTextArea"] textarea {
            overflow: auto !important;
            resize: vertical !important;   /* drag the corner for more room */
        }

        /* Generic helper for any custom scrolling block. */
        .ec-scroll {
            max-height: 420px;
            overflow: auto;
            padding: 0.6rem 0.8rem;
            background: #ffffff;
            border: 1px solid var(--ec-border);
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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


st.markdown(
    """
    <div class="ec-topbar">
        <div class="ec-brand-wrap">
            <div class="ec-logo-box">BUET</div>
            <div>
                <div class="ec-brand-title">BUET E-COUNCIL</div>
                <div class="ec-brand-subtitle">OCR Document Processor</div>
            </div>
        </div>
        <div class="ec-status">
            <span class="ec-status-dot"></span>
            OCR service ready
        </div>
    </div>

    <div class="app-intro">
        <p><strong>Turn a scanned meeting document into text and a structured record.</strong></p>
        <p><b>Step 1</b> — upload your PDF and press <b>Read the document</b>.
           <b>Step 2</b> — press <b>Create the meeting record</b> to turn that text into JSON.</p>
        <p>Everything is already set up. The only thing you need to choose is what kind
           of document you have, in the panel on the left.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📄 Document options")
    st.caption("Gemini access is already configured by the administrator.")

    document_kind = st.radio(
        "What kind of document is this?",
        [
            "Modern printed document",
            "Old, faded or handwritten document",
        ],
        help=(
            "This single choice sets everything else for you: how much the page "
            "images are cleaned up, and how many pages are read at a time."
        ),
    )
    is_old_document = document_kind.startswith("Old")
    degraded_scan = is_old_document  # kept for the processing-details panel
    preprocess_profile = "degraded" if is_old_document else "standard"
    recommended_pages_per_batch = 1 if is_old_document else 4

    st.caption(
        "Reading **one page at a time** with extra image cleanup. Slower, but the "
        "most accurate setting for difficult handwriting."
        if is_old_document
        else "Reading **4 pages at a time**. A good balance of speed and accuracy."
    )

    st.divider()
    st.caption("The recommended settings work well for most documents.")

    with st.expander("⚙️ Advanced settings", expanded=False):
        st.caption("Change these only when you understand their effect.")

        model_name = st.text_input(
            "Gemini model",
            value=MODEL_NAME,
            help="The model used for both OCR and JSON extraction.",
        )

        ocr_mode_choice = st.radio(
            "OCR processing method",
            ["High-DPI page images (recommended)", "Raw PDF chunks"],
            help="Image mode is more accurate for scans. Raw PDF mode usually costs less for clean documents.",
        )
        input_mode = "images" if ocr_mode_choice.startswith("High") else "pdf"

        if input_mode == "images" and not PIL_AVAILABLE:
            st.warning(
                "Pillow is not installed, so contrast enhancement is unavailable. "
                "Run: pip install pillow"
            )

        ocr_dpi = st.slider(
            "Image quality (DPI)",
            min_value=200,
            max_value=400,
            value=OCR_IMAGE_DPI,
            step=50,
            help="Higher DPI can improve small or faded text but increases cost and payload size.",
        )
        chunk_size = st.slider(
            "Pages read at a time",
            min_value=1,
            max_value=40,
            value=recommended_pages_per_batch,
            step=1,
            key=f"pages_per_batch_{preprocess_profile}",
            help=(
                "Fewer pages at a time is more accurate, because the model sees less "
                "at once and cannot copy details between pages. More pages is faster "
                "and cheaper. Changing the document type above resets this."
            ),
        )
        use_text_layer = st.checkbox(
            "Use embedded PDF text when available",
            value=True,
            help="Only genuinely digital pages are extracted locally (free). Scanned pages — including scans that hide a legacy OCR layer — always go to Gemini.",
        )
        json_chunk_chars = st.slider(
            "Characters per JSON request",
            min_value=30_000,
            max_value=100_000,
            value=JSON_CHUNK_CHARS,
            step=10_000,
            help="Reduce this if a very dense meeting produces incomplete JSON.",
        )
        max_workers = st.slider(
            "Parallel requests",
            min_value=1,
            max_value=24,
            value=DEFAULT_MAX_WORKERS,
            help="Controls how many requests may overlap. The RPM limit still controls request starts.",
        )
        safe_rpm = st.number_input(
            "Request limit per minute",
            min_value=1,
            max_value=1000,
            value=DEFAULT_SAFE_RPM,
            step=1,
            help="Set this to a safe per-project RPM for the configured Gemini projects.",
        )

        st.divider()
        st.caption(
            f"SDK: {'google-genai (new)' if NEW_SDK else 'google-generativeai (legacy)'}"
        )
        st.caption(
            "Temporary API errors are retried automatically. Incomplete OCR chunks are "
            "validated, retried, and split when necessary. Requests use temperature 0."
        )
        st.caption(
            "Cost note: higher image DPI uses more tokens. Raw PDF mode is cheaper for "
            "clean prints, and usable embedded text is extracted locally for free."
        )

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
                   these locally is free and character-perfect.
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
            text = (page.get_text("text") or "").strip()

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
    """Resolution of the scan actually embedded in this page, in DPI.

    A scanned page holds one raster image; rendering it above that resolution
    only interpolates. Both sample corpora are 150 DPI scans, so a 300 DPI
    render is already 2x upsampled and a 400 DPI render adds nothing but
    payload — and payload is what triggers the quality ladder below.
    Returns 0.0 when the page has no raster image (a digital page).
    """
    try:
        width_pt = float(page.rect.width)
        height_pt = float(page.rect.height)
    except Exception:
        return 0.0
    if width_pt <= 0 or height_pt <= 0:
        return 0.0

    best = 0.0
    try:
        infos = page.get_image_info()
    except Exception:
        return 0.0
    for info in infos or ():
        try:
            px_w = float(info.get("width") or 0)
            px_h = float(info.get("height") or 0)
        except Exception:
            continue
        if px_w <= 0 or px_h <= 0:
            continue
        best = max(best, px_w / (width_pt / 72.0), px_h / (height_pt / 72.0))
    return best


def effective_render_dpi(page, requested_dpi: int) -> int:
    """Clamp the requested DPI to at most 2x the page's native resolution."""
    native = _native_raster_dpi(page)
    if native <= 0:
        return int(requested_dpi)
    return int(min(float(requested_dpi), native * 2.0))


def _image_mime(data: bytes) -> str:
    """PNG or JPEG, decided from the bytes rather than assumed."""
    return "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"


def _render_page_image(page, dpi: int, preprocess: str = "standard") -> bytes:
    """Render one PDF page for OCR, preserving every stroke the scan contains.

    Two properties matter for Bengali at this scan resolution, where the
    difference between শ and স, or ত and ৎ, is one or two pixels wide:

    1. NO LOSSY RE-ENCODE. The page is saved as PNG. Measured on this corpus
       a 300 DPI grayscale page is 0.93 MB as PNG versus 0.90 MB as JPEG q88,
       so the lossless path is effectively free and removes the JPEG ringing
       that used to smear those thin strokes.
    2. NO HISTOGRAM CLIPPING by default. autocontrast(cutoff=1) discards the
       darkest and lightest 1% of pixels, which is where a faint matra lives.
       The standard profile now stretches without discarding (cutoff=0); the
       'degraded' profile keeps the aggressive clipping for stained or faded
       handwritten pages, where it genuinely helps.

    The size ladder now degrades QUALITY before it ever degrades RESOLUTION,
    and the render DPI is clamped to the page's native resolution beforehand,
    so a higher slider setting can no longer produce a lower-resolution image.
    """
    cap = int(PER_PAGE_IMAGE_MB * 1024 * 1024)
    render_dpi = effective_render_dpi(page, dpi)
    data = b""

    # (dpi, encoder) — lossless first, resolution reduced only as a last resort.
    ladder = (
        (render_dpi, "png"),
        (render_dpi, 92),
        (render_dpi, 85),
        (max(200, int(render_dpi * 0.75)), 85),
        (180, 72),
    )
    for attempt_dpi, encoder in ladder:
        if PIL_AVAILABLE:
            pix = page.get_pixmap(dpi=attempt_dpi, colorspace=fitz.csGRAY)
            img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            if preprocess == "degraded":
                # Stained/tinted background: clip it back and push ink-vs-paper
                # separation so faint strokes survive.
                img = ImageOps.autocontrast(img, cutoff=3)
                img = ImageEnhance.Contrast(img).enhance(1.6)
            else:
                # Stretch to full range WITHOUT discarding extreme pixels.
                img = ImageOps.autocontrast(img, cutoff=0)
            buf = io.BytesIO()
            if encoder == "png":
                img.save(buf, "PNG", optimize=True)
            else:
                img.save(buf, "JPEG", quality=encoder, optimize=True)
            data = buf.getvalue()
        else:
            pix = page.get_pixmap(dpi=attempt_dpi, colorspace=fitz.csGRAY)
            data = pix.tobytes("png" if encoder == "png" else "jpg")
        if len(data) <= cap:
            return data
    return data  # smallest achievable — let the payload check decide


def render_chunk_page_images(
    chunk_pdf_bytes: bytes, dpi: int, preprocess: str = "standard"
) -> list:
    """Render every page of a chunk PDF as a preprocessed JPEG (in order)."""
    src = fitz.open(stream=chunk_pdf_bytes, filetype="pdf")
    try:
        return [_render_page_image(page, dpi, preprocess) for page in src]
    finally:
        src.close()


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
    # OCR stutter: a duplicated মোঃ initial ("মো মোঃ" / "মোঃ মোঃ") is never a
    # legitimate sequence in a Bengali name — collapse it deterministically.
    value = re.sub(r"(?<!\S)মো[ঃ:]?\s+মোঃ(?!\S)", "মোঃ", value)
    return value


_thread_local = threading.local()


def get_new_client(api_key: str):
    """Create one reusable client per worker thread."""
    client = getattr(_thread_local, "new_client", None)
    client_key = getattr(_thread_local, "new_client_key", None)
    if client is None or client_key != api_key:
        client = genai_new.Client(api_key=api_key)
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


def new_sdk_config(**kwargs):
    """Add low-thinking configuration when supported by the installed SDK."""
    if hasattr(genai_types, "ThinkingConfig"):
        kwargs["thinking_config"] = genai_types.ThinkingConfig(
            thinking_level=kwargs.pop("_thinking_level", "minimal")
        )
    else:
        kwargs.pop("_thinking_level", None)
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
) -> str:
    """Send one chunk (as a PDF or as preprocessed page images) and VALIDATE
    the result before accepting it.

    A response is rejected (and retried) when it is empty, truncated by the
    output-token limit, or missing any '=== PAGE n ===' marker for the pages
    this chunk contains. Only a complete, verified transcription is returned.
    """
    prompt = chunk_prompt_for(input_mode, expected_pages, preprocess)

    # Build payload parts ONCE, before the retry loop.
    if input_mode == "images":
        images = render_chunk_page_images(chunk_pdf_bytes, dpi, preprocess)
        total_bytes = sum(len(img) for img in images)
        if total_bytes > MAX_INLINE_MB * 1024 * 1024:
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
                        temperature=0.0,
                        _thinking_level=OCR_THINKING_LEVEL,
                    ),
                )
            else:
                response = get_legacy_model(api_key, model_name).generate_content(
                    [prompt] + payload_parts,
                    generation_config={"temperature": 0.0},
                )

            text = (response.text or "").strip()
            if not text:
                raise OCRIncompleteError("empty OCR response")
            if _response_truncated(response):
                raise OCRIncompleteError(
                    "OCR response truncated by the output-token limit"
                )
            missing = missing_page_numbers(text, expected_pages)
            if missing:
                raise OCRIncompleteError(
                    f"OCR output is missing page marker(s) {missing} "
                    f"out of {expected_pages} expected page(s)"
                )
            key_pool.record_success(api_key)
            return text

        except OCRIncompleteError as e:
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
    )
    right_text = ocr_chunk_bulletproof(
        right_bytes, right_pages, key_pool, model_name,
        input_mode=input_mode, dpi=dpi, preprocess=preprocess,
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
                earlier_entry = section_names.get(name_key)
                if earlier_entry is not None and earlier_entry != entry_number:
                    try:
                        gap = abs(
                            int(entry_number.translate(BENGALI_TO_ARABIC_DIGITS))
                            - int(earlier_entry.translate(BENGALI_TO_ARABIC_DIGITS))
                        )
                    except ValueError:
                        gap = 1
                    # Far-apart repeats are almost always two real namesakes;
                    # reporting them trains the reader to ignore this warning.
                    if gap <= NAME_REPEAT_MAX_GAP:
                        issues.append(
                            f"\"{person}\" appears at entries {earlier_entry} and "
                            f"{entry_number} of the same section — only {gap} apart, so "
                            f"one line may have been copied over another entry's name. "
                            f"Verify both against the PDF."
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


# A duplicate AFFILIATION is logically impossible (one ডীন per অনুষদ), so it is
# always reported. A duplicate NAME is not: the 463rd minutes genuinely list
# অধ্যাপক ডঃ মোঃ মনিরুল ইসলাম at CSE entries ৩ and ৯ — two different people.
# Copying, by contrast, lands on a NEIGHBOURING line, so only a near-adjacent
# repeat is worth a human's attention.
NAME_REPEAT_MAX_GAP = 2

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


# ==========================================
# Main UI
# ==========================================
st.markdown(
    """
    <div class="ec-section-heading">
        <div class="ec-section-badge">1</div>
        <div class="ec-section-title">Extract text from the document</div>
    </div>
    <div class="ec-section-copy">
        Upload a PDF and press Run OCR. The app automatically checks every processed page.
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_pdf = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    help="Supported format: PDF",
)

if uploaded_pdf is not None:
    file_col, size_col = st.columns(2)
    file_col.metric("Selected file", uploaded_pdf.name)
    size_col.metric("File size", f"{uploaded_pdf.size / (1024 * 1024):.1f} MB")

    # Job key ties resume-state to the exact PDF content and OCR settings.
    uploaded_pdf_bytes = uploaded_pdf.getvalue()
    pdf_digest = hashlib.sha256(uploaded_pdf_bytes).hexdigest()[:20]
    mode_tag = (
        f"{input_mode}{ocr_dpi if input_mode == 'images' else ''}_{preprocess_profile}"
        + ("_tl" if use_text_layer else "")
    )
    job_key = f"ocr_{pdf_digest}_{chunk_size}_{model_name}_{mode_tag}"
    resuming = (
        st.session_state.get("job_key") == job_key
        and st.session_state.get("chunks_done")
        and not st.session_state.get("job_complete", False)
    )

    button_label = "🔁 Continue reading" if resuming else "🚀 Read the document"
    if st.button(button_label, type="primary", use_container_width=True):
        if not api_key:
            st.error("The reading service is not set up yet. Please ask the administrator to add a Gemini API key, then try again.")
            st.stop()

        with st.spinner("Splitting PDF into chunks..."):
            try:
                chunks, total_pages = split_pdf_into_chunks(uploaded_pdf_bytes, chunk_size)
            except Exception as e:
                st.error(f"Failed to read PDF: {e}")
                st.stop()

        total_chunks = len(chunks)
        mode_desc = (
            f"high-DPI page images ({ocr_dpi} DPI, grayscale + "
            + ("degraded-scan cleanup" if degraded_scan else "contrast")
            + ")"
            if input_mode == "images"
            else "raw PDF chunks"
        )
        st.success(
            f"Ready — {total_pages} page(s) will be read in {total_chunks} batch(es)."
        )
        if input_mode == "images":
            # The scan's own resolution is the hard ceiling on OCR accuracy;
            # rendering above 2x it only interpolates.
            _probe = fitz.open(stream=uploaded_pdf_bytes, filetype="pdf")
            try:
                _native = _native_raster_dpi(_probe[0]) if len(_probe) else 0.0
                _effective = (
                    effective_render_dpi(_probe[0], int(ocr_dpi))
                    if len(_probe)
                    else int(ocr_dpi)
                )
            finally:
                _probe.close()
            if _native > 0:
                st.caption(
                    f"Scan resolution: **{_native:.0f} DPI**. Rendering at "
                    f"**{_effective} DPI** (capped at 2x the scan). Detail above the "
                    "scan's own resolution cannot be recovered by a higher setting — "
                    "if names are still misread, the source scan is the limit."
                )
        with st.expander("Technical details (optional)", expanded=False):
            st.markdown(
                f"""
                **Input method:** {mode_desc}  
                **Parallel requests:** up to {min(max_workers, total_chunks)}  
                **Request limit:** {safe_rpm} per minute per active key  
                **Configured keys:** {len(api_keys)} (failover only)  
                **Validation:** every page is checked before acceptance
                """
            )

        # Initialize / reset resume state for a new file.
        if st.session_state.get("job_key") != job_key:
            st.session_state["job_key"] = job_key
            st.session_state["chunks_done"] = {}
            st.session_state["job_complete"] = False

        chunks_done = st.session_state["chunks_done"]

        # FREE PATH: pages with a TRUSTWORTHY embedded text layer are
        # extracted locally. A chunk whose entire page range qualifies never
        # touches the API. Scanned pages that merely carry a hidden legacy OCR
        # layer are deliberately excluded — see extract_text_layer_pages.
        if use_text_layer:
            with st.spinner("Checking for an embedded text layer..."):
                text_layer_pages, rejected_layer_pages = extract_text_layer_pages(
                    uploaded_pdf_bytes
                )
            if text_layer_pages:
                local_chunks = 0
                local_pages = 0
                for idx, (start, end, _bytes) in enumerate(chunks):
                    if idx in chunks_done:
                        continue
                    if all(p in text_layer_pages for p in range(start - 1, end)):
                        parts = [
                            f"=== PAGE {p} ===\n"
                            + clean_bengali_ocr_text(text_layer_pages[p - 1])
                            for p in range(start, end + 1)
                        ]
                        chunks_done[idx] = (start, end, "\n\n".join(parts))
                        local_chunks += 1
                        local_pages += end - start + 1
                if local_chunks:
                    st.success(
                        f"💰 {local_pages} page(s) had a trustworthy embedded text "
                        f"layer and were extracted locally — {local_chunks} chunk(s) "
                        "will cost nothing."
                    )
                else:
                    st.info(
                        f"{len(text_layer_pages)} page(s) have a usable text layer "
                        "but share chunks with scanned pages — reduce 'Pages per OCR "
                        "request' to let them skip the API."
                    )
            if rejected_layer_pages:
                order = sorted(rejected_layer_pages)
                sample = ", ".join(str(p + 1) for p in order[:8])
                if len(order) > 8:
                    sample += ", ..."
                st.info(
                    f"{len(rejected_layer_pages)} page(s) carry an embedded text "
                    f"layer that is NOT real document text — {rejected_layer_pages[order[0]]}. "
                    f"Affected page(s): {sample}. These pages are being read by "
                    "Gemini instead; trusting the built-in layer would return a "
                    "scrambled transcription for free rather than a correct one."
                )

        # Input-token estimate for the requests that will actually be sent.
        remote_chunks = [
            (start, end, b)
            for idx, (start, end, b) in enumerate(chunks)
            if idx not in chunks_done
        ]
        if remote_chunks:
            est_total, est_per_page = estimate_ocr_input_tokens(
                remote_chunks, input_mode, int(ocr_dpi)
            )
            pdf_alt, _ = estimate_ocr_input_tokens(remote_chunks, "pdf", 0)
            comparison = (
                f" (PDF mode would be ≈ {pdf_alt:,})"
                if input_mode == "images"
                else ""
            )
            st.caption(
                f"Estimated OCR input ≈ **{est_total:,} tokens** "
                f"(~{est_per_page:,} tokens/page{comparison}). Output tokens ≈ "
                "the document's text length. Estimates exclude retries and "
                "implicit-cache discounts on the repeated prompt."
            )

        progress = st.progress(len(chunks_done) / total_chunks)
        status = st.empty()
        failed_chunks = []

        _skip_keys = remembered_exhausted_keys()
        key_pool = APIKeyPool(api_keys, int(safe_rpm), unavailable=_skip_keys)
        if _skip_keys and len(_skip_keys) < len(api_keys):
            st.caption(
                f"Skipping {len(_skip_keys)} key(s) that ran out of quota in the "
                "last 10 minutes — they will be tried again after that."
            )
        pending = [
            (idx, start, end, chunk_bytes)
            for idx, (start, end, chunk_bytes) in enumerate(chunks)
            if idx not in chunks_done
        ]

        def run_ocr(item):
            idx, start, end, chunk_bytes = item
            expected = end - start + 1
            text = ocr_chunk_bulletproof(
                chunk_bytes, expected, key_pool, model_name,
                input_mode=input_mode, dpi=int(ocr_dpi),
                preprocess=preprocess_profile,
            )
            text = renumber_pages(clean_bengali_ocr_text(text), start)
            return idx, start, end, text

        if pending:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(pending))) as executor:
                future_map = {
                    executor.submit(run_ocr, item): item for item in pending
                }
                for future in as_completed(future_map):
                    idx, start, end, _ = future_map[future]
                    try:
                        idx, start, end, text = future.result()
                        chunks_done[idx] = (start, end, text)
                    except Exception as e:
                        st.error(
                            f"Chunk {idx + 1} (pages {start}–{end}) failed after all "
                            f"retries and splits: {e}"
                        )
                        failed_chunks.append((idx + 1, start, end))

                    processed = len(chunks_done) + len(failed_chunks)
                    progress.progress(min(1.0, processed / total_chunks))
                    status.text(
                        f"Processed {processed}/{total_chunks} chunks "
                        f"(verified OK: {len(chunks_done)}; latest: pages {start}–{end})..."
                    )

        progress.progress(1.0)
        st.session_state["job_complete"] = (
            len(chunks_done) == total_chunks and not failed_chunks
        )

        if failed_chunks:
            status.text(f"Finished with {len(failed_chunks)} failed chunk(s) ⚠️")
            st.warning(
                "Failed chunks: "
                + ", ".join(f"chunk {c} (pages {s}–{e})" for c, s, e in failed_chunks)
                + ". Failed chunks are NOT cached, so pressing 🔁 Resume OCR re-sends "
                "only these chunks."
            )
        else:
            status.text("Done ✅")

        combined_parts = []
        for i in range(total_chunks):
            if i in chunks_done:
                combined_parts.append(chunks_done[i][2])
            else:
                s_pg, e_pg, _bytes = chunks[i]
                combined_parts.append(
                    f"[OCR FAILED FOR PAGES {s_pg}–{e_pg} — press Resume OCR to retry this chunk]"
                )
        combined = "\n\n".join(combined_parts)

        # Illegible-word report: the prompt writes [?] instead of guessing.
        unreadable = combined.count("[?]")
        if unreadable:
            st.info(
                f"The model marked **{unreadable}** word(s) as illegible ([?]) "
                "rather than guessing — search the downloaded text for '[?]' and "
                "fill them in from the source document."
            )

        # Whole-document audit: every absolute page marker must be present exactly.
        if st.session_state["job_complete"]:
            found_pages = pages_in_text(combined)
            absolute_missing = [
                n for n in range(1, total_pages + 1) if n not in found_pages
            ]
            if absolute_missing:
                st.session_state["job_complete"] = False
                st.warning(
                    "Page audit: markers missing for page(s) "
                    + ", ".join(map(str, absolute_missing))
                    + ". Press Resume OCR to retry."
                )
            else:
                st.success(
                    f"Page audit passed ✅ — all {total_pages} page markers are present."
                )

        # Deterministic duplicate-affiliation audit (catches attention-drift
        # errors like two deans sharing the same faculty).
        consistency_issues = ocr_consistency_report(combined)
        if consistency_issues:
            st.warning(
                f"⚠️ {len(consistency_issues)} attendee line(s) look like they may have "
                "been copied from a neighbouring entry. Worth a quick check."
            )
            with st.expander("Show the lines to check", expanded=False):
                with scroll_box(300):
                    st.markdown("- " + "\n- ".join(consistency_issues))

        # Page-boundary audit: catches an item number lost in a damaged margin
        # followed by silent renumbering of the rest of the list.
        boundary_issues = page_boundary_item_report(combined)
        if boundary_issues:
            st.warning(
                f"⚠️ {len(boundary_issues)} page(s) may have lost an item number at the "
                "page edge. The app re-reads those pages on their own to check."
            )
            with st.expander("Show the pages to check", expanded=False):
                with scroll_box(300):
                    st.markdown(
                        "- "
                        + "\n- ".join(issue["message"] for issue in boundary_issues)
                    )

            # AUTO CROSS-CHECK: renumbering happens because the model remembers
            # the previous page's last item number ("after ৭ comes ৮") and that
            # counting prior outvotes an ambiguous digit. Re-OCR the flagged
            # page in ISOLATION — with no previous page in context there is no
            # counting bias, so the isolated read reports the printed numbers
            # faithfully. If the two reads disagree, the isolated one wins.
            if st.session_state.get("job_complete") and api_keys:
                for issue in boundary_issues:
                    flagged_page = issue["page"]
                    try:
                        solo_bytes = extract_single_page_pdf(
                            uploaded_pdf_bytes, flagged_page
                        )
                        solo_text = ocr_chunk_bulletproof(
                            solo_bytes, 1, key_pool, model_name,
                            input_mode=input_mode, dpi=int(ocr_dpi),
                            preprocess=preprocess_profile,
                        )
                        solo_text = renumber_pages(
                            clean_bengali_ocr_text(solo_text), flagged_page
                        )
                        old_numbers = page_item_numbers(combined, flagged_page)
                        new_numbers = page_item_numbers(solo_text, flagged_page)
                        if old_numbers and new_numbers and old_numbers != new_numbers:
                            combined = replace_page_text(
                                combined, flagged_page, solo_text
                            )
                            st.info(
                                f"🔧 Page {flagged_page} was re-read in isolation "
                                f"(no cross-page counting bias): its printed item "
                                f"numbers came back as {new_numbers} instead of "
                                f"{old_numbers}. The isolated read was adopted — "
                                f"it reflects what is actually on the page."
                            )
                        elif old_numbers and new_numbers:
                            st.caption(
                                f"Isolated re-read of page {flagged_page} produced "
                                f"the same item numbers {old_numbers} — the "
                                "numbering is consistent across two independent "
                                "reads; only the margin question remains."
                            )
                    except Exception as repair_error:
                        st.caption(
                            f"Isolated re-read of page {flagged_page} failed "
                            f"({repair_error}) — verify that page manually."
                        )

        remember_exhausted_keys(key_pool)
        st.session_state["ocr_result"] = combined
        st.session_state["ocr_filename"] = uploaded_pdf.name.rsplit(".", 1)[0]

# ==========================================
# Download section (persists after rerun)
# ==========================================
if "ocr_result" in st.session_state:
    st.divider()
    st.subheader("✅ OCR result")
    base_name = st.session_state.get("ocr_filename", "ocr_output")
    st.download_button(
        "Download extracted text (.txt)",
        data=st.session_state["ocr_result"],
        file_name=f"{base_name}_ocr.txt",
        mime="text/plain",
    )
    st.download_button(
        "Download as Markdown (.md)",
        data=st.session_state["ocr_result"],
        file_name=f"{base_name}_ocr.md",
        mime="text/markdown",
    )
    with st.expander("Preview extracted text", expanded=False):
        ocr_preview_text = st.session_state["ocr_result"]
        st.caption(
            f"{len(ocr_preview_text):,} characters — scroll inside the box below. "
            "Drag its bottom-right corner to make it taller."
        )
        st.text_area(
            "Combined OCR output",
            ocr_preview_text,
            height=400,
        )

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
    and every data row. Never flatten a table into ordinary paragraph text.
  - resolution: full "সিদ্ধান্ত : ..." text verbatim. If the resolution is missing in this text portion, use null. Preserve any table in the resolution as a Markdown pipe table too.
  - ONE PROPOSAL = ONE AGENDA ENTRY, EVEN ACROSS PAGES: a proposal begins at a "প্রস্তাব নং ..." line and continues until the NEXT "প্রস্তাব নং ..." line. Everything in between belongs to that same entry: continuation paragraphs, tables and table rows that carry on over a page break, repeated table headers, the section/department/faculty headings that label those tables (e.g. "স্থাপত্য বিভাগ", "পুরকৌশল বিভাগ", "আই.পি.ই বিভাগ", "যন্ত্রকৌশল বিভাগ"), lists of names, roll numbers or course codes, and that item's own "সিদ্ধান্ত ঃ" text. NEVER begin a new agenda entry merely because a new page starts, a new table starts, or a new heading appears. If a page begins with a table, a table header row, a heading, or any text that is not itself a "প্রস্তাব নং ..." line, it is a CONTINUATION: append it to the body of the proposal already in progress, keeping every table as a Markdown pipe table. In this format an agenda entry whose body does not begin with "প্রস্তাব নং" is always a mistake.
  - OLD FORMAT: older minutes have no প্রস্তাব নং items; instead a সিদ্ধান্তাবলী (decisions) section — or in English a "RESOLUTIONS:" section — lists numbered items (১।, ২।, ... / 1., 2., ...). Treat each numbered item as one agenda entry: body = the item's full text verbatim. If the item text itself states the decision (…সিদ্ধান্ত গ্রহণ করা হয়, …অনুমোদন করা হয়, …কনফার্ম করা হয়; English: "Confirmed ...", "... and resolved that ...", "Considered and approved ..."), also copy that deciding sentence (or the whole item if it is one sentence) into resolution; otherwise resolution = null. This next exception applies ONLY to that old format — a document that contains no "প্রস্তাব নং" item anywhere: if a page BEGINS with a short, complete, standalone decision paragraph that carries no number (its number may have been lost in a damaged margin) and is not a continuation of the previous item, treat it as its OWN agenda entry. It NEVER applies to a document that uses প্রস্তাব নং, and it never applies to a table, a table header row, or a section/department heading.
- Copy all Bengali text EXACTLY as written (do not modernize spelling, do not translate, do not transliterate). Fix only obvious OCR artifacts like stray Latin/Arabic/Devanagari characters inside Bengali words when the correct Bengali word is unambiguous — but NEVER apply this to a PERSON'S NAME. A name has no “correct” form other than the one printed, so pass every name through completely unchanged, character for character, even when it looks misspelled, unusual, or like a familiar name with one letter wrong. These names are written straight into a permanent database that identifies real people and is never re-checked against the document, so silently “improving” one is the most damaging thing you can do here. Keep [?] illegible-word placeholders exactly where the OCR placed them.
- Strip page markers like '=== PAGE n ===' and page headers/footers/page numbers from all extracted text.
- NEVER invent values. If a field is not present in this text portion, use null (or [] for lists)."""


def split_text_with_overlap(full_text: str, max_chars: int = JSON_CHUNK_CHARS):
    """Split at '=== PAGE n ===' markers with a 1-page overlap between chunks,
    so an agenda item cut at a boundary is seen whole by the next chunk."""
    pages = re.split(r"(?=^=== PAGE \d+ ===\s*$)", full_text, flags=re.MULTILINE)
    pages = [p for p in pages if p.strip()]
    if len(pages) <= 1:  # no markers — hard-split
        return [full_text[i : i + max_chars] for i in range(0, len(full_text), max_chars)] or [full_text]

    chunks, current, prev_page = [], "", ""
    for page in pages:
        if current and len(current) + len(page) > max_chars:
            chunks.append(current)
            current = prev_page + page  # overlap: repeat last page of previous chunk
        else:
            current += page
        prev_page = page
    if current.strip():
        chunks.append(current)
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
    if isinstance(parsed, list):
        parsed = next((item for item in parsed if isinstance(item, dict)), None)
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

    for person in parsed["presentees"]:
        if not isinstance(person, dict):
            raise ValueError("a presentee entry is not an object")

    for item in parsed["agenda"]:
        if not isinstance(item, dict):
            raise ValueError("an agenda entry is not an object")
        body = item.get("body")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("an agenda entry is missing its 'body'")
        serial = item.get("serial")
        if isinstance(serial, str):
            digits = re.sub(
                r"\D", "", serial.translate(BENGALI_TO_ARABIC_DIGITS)
            )
            item["serial"] = int(digits) if digits else None

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
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=MEETING_SCHEMA,
                        _thinking_level=JSON_THINKING_LEVEL,
                    ),
                )
                raw = response.text or ""
            else:
                model = get_legacy_model(api_key, model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                        "response_schema": MEETING_SCHEMA,
                    },
                )
                raw = response.text or ""

            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            if not raw:
                raise ValueError("empty JSON response")
            result = _validated_meeting_partial(json.loads(raw))
            key_pool.record_success(api_key)
            return result

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


def stitch_split_agenda_items(meeting: dict):
    """Re-join a proposal that was split across a page or chunk boundary.

    In the modern format every agenda entry starts with "প্রস্তাব নং ...".
    When a proposal's tables or trailing paragraphs continue onto the next
    page, the model sometimes emits that continuation as a SEPARATE entry with
    no proposal number at all — which is how প্রস্তাব নং এ ১৪০১০৬৫ came back as
    two agenda items, the second one headless.

    Any entry without its own proposal heading is therefore folded back into
    the entry above it: body appended, resolution carried over. Purely local
    and deterministic, so it costs nothing and cannot invent text. Old-format
    minutes (numbered সিদ্ধান্তাবলী items, English RESOLUTIONS) are left alone,
    because there every item legitimately lacks a proposal number.

    Returns (meeting, notes) where notes describes each re-join for the UI.
    """
    result = dict(meeting or {})
    agenda = [dict(item or {}) for item in (result.get("agenda") or [])]
    if not agenda or not _agenda_uses_proposal_numbers(agenda):
        return result, []

    stitched = []
    notes = []
    for item in agenda:
        body = (item.get("body") or "").strip()
        if (
            not stitched
            or _PROPOSAL_HEAD_RE.search(body[:200])
            or _STANDALONE_SECTION_RE.match(body)
        ):
            stitched.append(item)
            continue

        previous = stitched[-1]
        previous_body = (previous.get("body") or "").rstrip()
        if body:
            previous["body"] = (previous_body + "\n\n" + body).strip()

        previous_resolution = (previous.get("resolution") or "").strip()
        resolution = (item.get("resolution") or "").strip()
        if resolution and resolution not in previous_resolution:
            previous["resolution"] = (
                (previous_resolution + "\n\n" + resolution).strip()
                if previous_resolution
                else resolution
            )

        heading = re.search(
            r"প্রস্তাব\s*নং[^\n:ঃ]{0,28}", previous.get("body") or ""
        )
        label = (heading.group(0) if heading else previous_body[:40]).strip()
        snippet = re.sub(r"\s+", " ", body).strip()[:45]
        notes.append(
            f'a continuation block starting "{snippet}…" was re-joined to "{label}…"'
        )

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
    """Split one Markdown pipe-table row into cells."""
    value = str(line or "").strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    return [cell.strip() for cell in value.split("|")]


def _is_markdown_table_separator(line: str) -> bool:
    """Return True for rows such as | --- | :---: | ---: |."""
    cells = _split_markdown_table_row(line)
    return len(cells) >= 2 and all(
        bool(MARKDOWN_SEPARATOR_CELL.fullmatch(cell.replace(" ", "")))
        for cell in cells
    )


def _looks_like_table_row(line: str) -> bool:
    """A row like '| ক | খ | গ |' — at least 2 pipes and 2 non-empty cells.

    Lets tables be detected even when the '| --- | --- |' separator row is
    missing from the extracted text.
    """
    value = str(line or "").strip()
    if value.count("|") < 2:
        return False
    cells = _split_markdown_table_row(value)
    return len(cells) >= 2 and any(cell for cell in cells)


def _render_html_table(header: list, rows: list) -> str:
    """Render parsed table cells as compact, safe HTML.

    Attributes are single-quoted so json.dumps never has to escape them, and
    border='1' is included as a fallback for HTML sanitizers that strip
    inline style attributes.
    """
    column_count = len(header)
    header_html = "".join(
        f"<th style='border:1px solid #000;padding:6px;text-align:left;'>{_format_table_cell_html(cell)}</th>"
        for cell in header
    )

    body_rows = []
    for row in rows:
        normalized_row = list(row[:column_count])
        if len(normalized_row) < column_count:
            normalized_row.extend([""] * (column_count - len(normalized_row)))

        cells_html = "".join(
            f"<td style='border:1px solid #000;padding:6px;'>{_format_table_cell_html(cell)}</td>"
            for cell in normalized_row
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

            segments.append(("table", _render_html_table(header, rows)))
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

# ==========================================
# Stage 2 UI
# ==========================================
st.divider()
st.markdown(
    """
    <div class="ec-section-heading">
        <div class="ec-section-badge">2</div>
        <div class="ec-section-title">Convert meeting text to JSON</div>
    </div>
    <div class="ec-section-copy">
        Use the OCR result from Step 1, or provide previously extracted text.
        The app validates and combines the meeting information automatically.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("What the JSON converter does", expanded=False):
    st.markdown(
        """
        - Keeps only the supported Bengali academic designations.
        - Standardizes close department and office matches while preserving ambiguous values.
        - Converts Bengali subpoints and Markdown tables into compact HTML.
        - Validates every JSON part, supports resume, and creates a final quality report.
        """
    )

source_choice = st.radio(
    "Choose the text source",
    ["Use OCR result from Step 1", "Upload or paste extracted text"],
    horizontal=True,
)

manual_text = ""
if source_choice == "Upload or paste extracted text":
    txt_file = st.file_uploader(
        "Upload extracted text (.txt / .md)",
        type=["txt", "md"],
        key="txt_up",
    )
    if txt_file is not None:
        manual_text = txt_file.read().decode("utf-8", errors="replace")
    manual_text = st.text_area("...or paste text here", value=manual_text, height=150)

current_json_source = (
    manual_text
    if source_choice.startswith("Upload")
    else st.session_state.get("ocr_result", "")
)

with st.expander("Optional: improve faculty-name spelling", expanded=False):
    st.caption(
        "Paste faculty names — one per line, or SQL INSERT statements containing "
        "the names as quoted strings. An OCR'd attendee name within close edit "
        "distance of exactly one roster entry is replaced by that exact spelling "
        "(e.g. শওকতয়ার্দী → সরওয়ার্দী). Very different names and ambiguous "
        "matches between two people are left unchanged, and every applied "
        "correction is listed for review."
    )
    roster_file = st.file_uploader(
        "Upload roster (.txt / .sql)", type=["txt", "sql"], key="roster_up"
    )
    roster_prefill = (
        roster_file.read().decode("utf-8", errors="replace") if roster_file else ""
    )
    roster_raw = st.text_area(
        "...or paste roster here", value=roster_prefill, height=140, key="roster_text"
    )

roster_names = parse_name_roster(roster_raw)
if roster_names:
    st.caption(
        f"Roster loaded ✅ — {len(roster_names)} unique names will be used to "
        "correct close OCR variants."
    )

json_job_key = None
json_resuming = False
if current_json_source.strip():
    json_digest = hashlib.sha256(current_json_source.encode("utf-8")).hexdigest()[:20]
    json_job_key = f"json_{json_digest}_{int(json_chunk_chars)}_{model_name}"
    json_resuming = (
        st.session_state.get("json_job_key") == json_job_key
        and bool(st.session_state.get("json_partials_done"))
        and not st.session_state.get("json_job_complete", False)
    )

json_button_label = (
    "🔁 Continue building the record" if json_resuming
    else "🧠 Create the meeting record"
)

_step_two_ready = bool(current_json_source.strip())
if not _step_two_ready:
    st.markdown(
        "<div class='ec-help'>Step 2 unlocks once Step 1 has produced text. Read a document above, or switch the source to <b>Upload or paste extracted text</b> and provide your own.</div>",
        unsafe_allow_html=True,
    )

if st.button(
    json_button_label,
    type="primary",
    use_container_width=True,
    disabled=not _step_two_ready,
):
    # Deterministic Bengali cleanup also protects manually pasted text.
    full_text = clean_bengali_ocr_text(current_json_source)

    if not full_text.strip():
        st.error("No text to process. Run OCR first or paste text.")
        st.stop()
    if not api_key:
        st.error("The reading service is not set up yet. Please ask the administrator to add a Gemini API key, then try again.")
        st.stop()
    if "[OCR FAILED FOR PAGES" in full_text:
        st.warning(
            "The OCR text still contains failed-chunk placeholders. The JSON will "
            "miss that content — resume the OCR stage first for a complete result."
        )

    chunks = split_text_with_overlap(full_text, int(json_chunk_chars))
    total = len(chunks)

    if st.session_state.get("json_job_key") != json_job_key:
        st.session_state["json_job_key"] = json_job_key
        st.session_state["json_partials_done"] = {}
        st.session_state["json_job_complete"] = False
        st.session_state.pop("json_result", None)

    partials_done = st.session_state["json_partials_done"]
    pending_items = [
        (i, chunk)
        for i, chunk in enumerate(chunks)
        if i not in partials_done
    ]

    st.info(
        f"Preparing {total} text batch(es). {len(pending_items)} batch(es) still need processing."
    )
    with st.expander("Technical details (optional)", expanded=False):
        st.markdown(
            f"""
            **Validated batches already saved:** {len(partials_done)}  
            **Requests remaining:** {len(pending_items)}  
            **Parallel requests:** up to {min(max_workers, max(1, len(pending_items)))}  
            **Request limit:** {safe_rpm} per minute per active key  
            **Configured keys:** {len(api_keys)} (failover only)
            """
        )

    progress = st.progress(len(partials_done) / total)
    status = st.empty()
    failed = []
    key_pool = APIKeyPool(
        api_keys, int(safe_rpm), unavailable=remembered_exhausted_keys()
    )

    def run_json(item):
        i, chunk = item
        result = gemini_extract_meeting(
            chunk,
            i + 1,
            total,
            key_pool,
            model_name,
        )
        return i, result

    if pending_items:
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(pending_items))
        ) as executor:
            future_map = {
                executor.submit(run_json, item): item[0]
                for item in pending_items
            }
            completed_this_run = 0
            for future in as_completed(future_map):
                i = future_map[future]
                try:
                    result_i, result = future.result()
                    partials_done[result_i] = result
                except Exception as e:
                    failed.append(i + 1)
                    st.error(f"Chunk {i + 1} failed: {e}")

                completed_this_run += 1
                progress.progress(len(partials_done) / total)
                status.text(
                    f"Finished {completed_this_run}/{len(pending_items)} requests in this run; "
                    f"cached {len(partials_done)}/{total} validated JSON chunks..."
                )

    remember_exhausted_keys(key_pool)

    if not partials_done:
        st.error("All chunks failed — nothing to merge.")
        st.stop()

    missing = [i + 1 for i in range(total) if i not in partials_done]
    if missing:
        st.session_state["json_job_complete"] = False
        st.warning(
            "The JSON is not final because these chunk(s) are still missing: "
            + ", ".join(map(str, missing))
            + ". Press Resume Meeting JSON to send only those chunks again."
        )
        status.text("Paused with missing chunks ⚠️")
    else:
        partials = [partials_done[i] for i in range(total)]
        status.text("Merging partial results locally (no extra request)...")

        final = (
            merge_meeting_partials(partials)
            if len(partials) > 1
            else dict(partials[0])
        )
        final, stitch_notes = stitch_split_agenda_items(final)
        final = _finalize_scalars(final)
        final = normalize_meeting_entities(final, name_roster=roster_names)
        applied_corrections = final.pop("_name_corrections", [])
        final = format_agenda_content_as_html(final)

        issues = meeting_quality_report(final)
        if issues:
            st.warning(
                f"⚠️ {len(issues)} thing(s) to verify against the original document "
                "before you use this record."
            )
            with st.expander("Show what to verify", expanded=False):
                with scroll_box(300):
                    st.markdown("- " + "\n- ".join(issues))
        else:
            st.success("Quality report passed ✅ — all key fields present and consistent.")

        if stitch_notes:
            st.info(
                f"🔧 {len(stitch_notes)} agenda continuation(s) were re-joined to "
                "their proposal (a proposal split across a page boundary):\n\n- "
                + "\n- ".join(stitch_notes)
            )

        if applied_corrections:
            st.info(
                f"Roster corrections applied to {len(applied_corrections)} "
                "attendee name(s):\n\n- " + "\n- ".join(applied_corrections)
            )

        st.session_state["json_job_complete"] = True
        st.session_state["json_result"] = json.dumps(
            final,
            ensure_ascii=False,
            indent=4,
        )
        status.markdown(
            f"**Done ✅ — {len(final.get('presentees', []))} presentees, "
            f"{len(final.get('agenda', []))} agenda items.**"
        )

if "json_result" in st.session_state:
    st.subheader("✅ Meeting JSON result")
    base_name = st.session_state.get("ocr_filename", "meeting")
    st.download_button(
        "Download meeting JSON",
        data=st.session_state["json_result"],
        file_name=f"{base_name}.json",
        mime="application/json",
    )
    with st.expander("Preview JSON", expanded=False):
        json_preview_text = st.session_state["json_result"]
        st.caption(
            f"{len(json_preview_text.splitlines()):,} lines — scroll inside the "
            "box below."
        )
        json_view = st.radio(
            "How would you like to see it?",
            ["Raw text", "Collapsible tree"],
            horizontal=True,
            key="json_preview_view",
            label_visibility="collapsed",
        )
        with_tree = json_view == "Collapsible tree"
        if with_tree:
            with scroll_box(460):
                try:
                    st.json(json.loads(json_preview_text), expanded=False)
                except json.JSONDecodeError:
                    code_block(json_preview_text, "json", 440)
        else:
            code_block(json_preview_text, "json", 460)
