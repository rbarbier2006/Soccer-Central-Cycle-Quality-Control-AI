

# pdf_report.py
import os
import re
import textwrap
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from profiles import SurveyProfile, PROFILES
import json
import math
from dataclasses import dataclass
import matplotlib.patches as patches

from openai import OpenAI

import json
from collections import defaultdict
import os
import streamlit as st

from typing import Literal
from pydantic import BaseModel, Field

# Load key from Streamlit secrets into env var so the OpenAI SDK can find it
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# -----------------------------
# Constants
# -----------------------------
YES_SET = {"YES", "Y", "TRUE", "1"}
NO_SET = {"NO", "N", "FALSE", "0"}


# -----------------------------
# Small helpers
# -----------------------------
def _clean_series_as_str_dropna(s: pd.Series) -> pd.Series:
    return s.dropna().astype(str).str.strip()


def _clean_series_as_str_keep_len(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _get_unique_respondent_count(df_group: pd.DataFrame, name_idx: int) -> int:
    if name_idx < 0 or name_idx >= len(df_group.columns):
        return int(len(df_group))
    names = _clean_series_as_str_dropna(df_group.iloc[:, name_idx])
    names = names[names != ""]
    return int(names.nunique())


def _compose_group_title(profile: SurveyProfile, title_label: str, cycle_label: str) -> str:
    base = str(title_label).strip()

    if base == "All Teams":
        return f"All Teams - {cycle_label}"

    if " - " in base:
        return f"{base} - {cycle_label}"

    coach = (profile.team_coach_map or {}).get(base, "?")
    return f"{base} - {coach} - {cycle_label}"


def _format_count_pct_cell(profile: SurveyProfile, team_name: str, count: int) -> str:
    denom_map = profile.denominator_map
    if not denom_map:
        return str(count)

    total = denom_map.get(team_name)
    if not total or total <= 0:
        return str(count)

    pct = (count / float(total)) * 100.0
    return f"{count} ({pct:.0f}%)"


def _fraction_numeric_between_1_and_5(sample: pd.Series) -> float:
    if sample is None or len(sample) == 0:
        return 0.0
    s = sample.dropna().astype(str).str.strip()
    if len(s) == 0:
        return 0.0
    nums = pd.to_numeric(s, errors="coerce")
    return float(nums.between(1, 5).mean())


def _assert_profile_and_df_make_sense(profile: SurveyProfile, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("The Excel sheet is empty or could not be read.")

    cols = list(df.columns)

    if profile.group_col_index < 0 or profile.group_col_index >= len(cols):
        raise ValueError(
            f"Group column index is outside available columns. "
            f"group_col_index={profile.group_col_index}, ncols={len(cols)}"
        )

    if profile.respondent_name_index < 0 or profile.respondent_name_index >= len(cols):
        raise ValueError(
            f"Respondent name index is outside available columns. "
            f"respondent_name_index={profile.respondent_name_index}, ncols={len(cols)}"
        )

    group_col_name = df.columns[profile.group_col_index]
    sample = df[group_col_name].dropna().astype(str).str.strip().head(80)
    frac_between_1_5 = _fraction_numeric_between_1_and_5(sample)
    if frac_between_1_5 > 0.60:
        raise ValueError(
            "BUG: Your grouping column looks like a 1-5 rating column. "
            f"group_col_index={profile.group_col_index}, group_col_name='{group_col_name}', "
            f"rating_like_fraction={frac_between_1_5:.2f}. "
            "Fix profiles.py (group_col_index) or ensure you selected the correct survey_type."
        )


def _normalize_group_column_inplace(df: pd.DataFrame, profile: SurveyProfile) -> str:
    group_col_name = df.columns[profile.group_col_index]
    df[group_col_name] = (
        df[group_col_name]
        .fillna(profile.unassigned_label)
        .astype(str)
        .str.strip()
    )
    df.loc[df[group_col_name] == "", group_col_name] = profile.unassigned_label
    return group_col_name

# ---------------------------
# AI Helpers (KEEP THIS BLOCK)
# ---------------------------
import os
import re
import json
import textwrap
from typing import Optional, List, Literal

import matplotlib.patches as patches
from pydantic import BaseModel, Field

# If you already have OpenAI imported elsewhere, you can remove this import here.
from openai import OpenAI


class Theme(BaseModel):
    title: str = Field(..., max_length=70)
    frequency: Literal["Very high", "High", "Medium", "Low"]
    criticality: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    whats_being_said: str = Field(..., max_length=320)
    emotional_signals: List[str] = Field(default_factory=list)  # limit in code/prompt


class CommentsInsights(BaseModel):
    top_priorities: List[str] = Field(default_factory=list)  # limit in code/prompt
    themes: List[Theme] = Field(default_factory=list)         # limit in code/prompt
    notes: str = Field(default="", max_length=280)


def _is_meaningful_comment(s: str) -> bool:
    if s is None:
        return False
    t = str(s).strip()
    if not t:
        return False
    low = t.lower()
    junk = {"no", "none", "n/a", "na", "nope", "nothing", "nil"}
    if low in junk:
        return False
    if len(t) < 8:
        return False
    return True


def _infer_comment_col_indices(profile, df) -> List[int]:
    """
    Prefer profile.comment_col_indices if it exists.
    Otherwise infer by header keywords + content shape.
    """
    idxs = getattr(profile, "comment_col_indices", None)
    if idxs:
        return [i for i in idxs if 0 <= int(i) < len(df.columns)]

    keywords = ("comment", "feedback", "suggest", "why", "explain", "anything else", "notes", "improve")
    candidate = []
    for i, col in enumerate(df.columns):
        name = str(col).strip().lower()
        if any(k in name for k in keywords):
            candidate.append(i)

    # fallback heuristic: object-like columns with average string length reasonably large
    if not candidate:
        for i, col in enumerate(df.columns):
            if i in (profile.group_col_index, profile.respondent_name_index):
                continue
            if i in (profile.rating_col_indices or []):
                continue
            if i in (profile.yesno_col_indices or []):
                continue
            if profile.choice_col_index is not None and i == int(profile.choice_col_index):
                continue

            series = df.iloc[:, i].dropna().astype(str).str.strip()
            if series.empty:
                continue
            avg_len = float(series.map(len).mean())
            if avg_len >= 25:
                candidate.append(i)

    return sorted(set(candidate))


def _collect_comments(df, col_indices: List[int]) -> List[str]:
    out = []
    for idx in col_indices:
        if idx < 0 or idx >= len(df.columns):
            continue
        series = df.iloc[:, idx].fillna("").astype(str).map(lambda x: x.strip())
        for val in series.tolist():
            if _is_meaningful_comment(val):
                out.append(val)

    # de-dup preserving order
    seen = set()
    dedup = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


def _try_get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        return OpenAI()
    except Exception:
        return None


def _safe_json_loads(s: str) -> Optional[dict]:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _llm_summarize_comments_structured(
    client,
    comments: List[str],
    model: str = "gpt-5-mini",
    chunk_size: int = 60,
) -> Optional[CommentsInsights]:
    """
    Returns CommentsInsights (structured). This is what enables the pretty PDF cards.
    """
    if not comments:
        return None

    # keep prompts consistent so output stays clean
    scale = (
        "Criticality scale:\n"
        "CRITICAL = threatens to leave / not recommend / strong anger or betrayal\n"
        "HIGH = strong frustration, repeated pain point, clear expectation gap\n"
        "MEDIUM = constructive suggestions, annoyance but not dealbreaker\n"
        "LOW = minor inconvenience\n"
    )

    system_msg = (
        "You analyze survey comments for a sports academy.\n"
        "Return a structured summary with 8-12 themes ranked by frequency.\n"
        "Rules:\n"
        "- Themes must be non-overlapping (dedupe aggressively).\n"
        "- Titles must be short and specific.\n"
        "- whats_being_said: 1-2 sentences max.\n"
        "- emotional_signals: 0-6 short phrases/quotes max (no paragraphs).\n"
        "- top_priorities: 0-6 concise action items.\n"
        "- Never add meta lines like 'If you want I can...'\n"
    )

    def one_pass(input_comments: List[str]) -> Optional[CommentsInsights]:
        user_msg = "SURVEY COMMENTS:\n" + "\n".join([f"- {c}" for c in input_comments])

        # Best case: structured parse
        try:
            resp = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": scale + "\n" + user_msg},
                ],
                text_format=CommentsInsights,
            )
            out = resp.output_parsed

            # hard limits (so rendering never explodes)
            out.top_priorities = (out.top_priorities or [])[:6]
            out.themes = (out.themes or [])[:12]
            for t in out.themes:
                t.emotional_signals = (t.emotional_signals or [])[:6]
            return out

        except Exception:
            # fallback: ask for JSON and validate
            resp = client.responses.create(
                model=model,
                instructions=system_msg,
                input=scale + "\n" + user_msg + "\n\nReturn ONLY valid JSON matching the schema.",
            )
            raw = getattr(resp, "output_text", "") or ""
            data = _safe_json_loads(raw)
            if not data:
                return None
            try:
                out = CommentsInsights.model_validate(data)
                out.top_priorities = (out.top_priorities or [])[:6]
                out.themes = (out.themes or [])[:12]
                for t in out.themes:
                    t.emotional_signals = (t.emotional_signals or [])[:6]
                return out
            except Exception:
                return None

    # one-shot
    if len(comments) <= chunk_size:
        return one_pass(comments)

    # chunk + local merge (simple + predictable)
    partials: List[CommentsInsights] = []
    for start in range(0, len(comments), chunk_size):
        chunk = comments[start:start + chunk_size]
        out = one_pass(chunk)
        if out:
            partials.append(out)

    if not partials:
        return None

    merged = CommentsInsights(top_priorities=[], themes=[], notes="")
    seen_titles = set()

    # priorities
    for p in partials:
        for item in (p.top_priorities or []):
            item = str(item).strip()
            if item and item not in merged.top_priorities and len(merged.top_priorities) < 6:
                merged.top_priorities.append(item)

    # themes (keep order; they are “ranked” within each chunk)
    for p in partials:
        for t in (p.themes or []):
            key = t.title.strip().lower()
            if not key or key in seen_titles:
                continue
            seen_titles.add(key)
            t.emotional_signals = (t.emotional_signals or [])[:6]
            merged.themes.append(t)
            if len(merged.themes) >= 12:
                break
        if len(merged.themes) >= 12:
            break

    merged.notes = (partials[0].notes or "").strip()
    return merged

# ---------------------------
# AI Rendering (KEEP THIS BLOCK)
# ---------------------------
import math

def _crit_style(crit: str):
    crit = (crit or "").upper().strip()
    if crit == "CRITICAL":
        return {"face": "#fde2e2", "edge": "#d14949"}
    if crit == "HIGH":
        return {"face": "#fff1d6", "edge": "#c47f00"}
    if crit == "MEDIUM":
        return {"face": "#e8f1ff", "edge": "#2f6fb0"}
    return {"face": "#eef7ee", "edge": "#2f7a3d"}  # LOW


def _wrap(s: str, width: int) -> str:
    s = (s or "").strip()
    return textwrap.fill(s, width=width) if s else ""


def _add_comments_insights_cards_to_pdf(
    pdf,
    title: str,
    insights: CommentsInsights,
    themes_per_page: int = 6,
) -> None:
    if insights is None or not getattr(insights, "themes", None):
        return

    def _wrap_lines(s: str, width: int) -> List[str]:
        s = (s or "").strip()
        if not s:
            return []
        return textwrap.wrap(s, width=width)

    def _truncate_lines(lines: List[str], max_lines: int) -> List[str]:
        if max_lines <= 0:
            return []
        if len(lines) <= max_lines:
            return lines
        out = lines[:max_lines]
        if out:
            last = out[-1]
            if len(last) >= 3:
                out[-1] = last[:-3] + "..."
            else:
                out[-1] = last + "..."
        return out

    # Conservative line height (the old 1.25 often underestimates and causes collisions)
    def _line_h_axes(fig, fontsize: float, line_spacing: float = 1.55) -> float:
        fig_h_in = float(fig.get_size_inches()[1])
        return (fontsize * line_spacing / 72.0) / fig_h_in

    themes = list(insights.themes or [])
    priorities = list((insights.top_priorities or [])[:6])
    has_priorities = len(priorities) > 0

    # Detect "verbose mode" (families tends to trigger this)
    if themes:
        avg_chars = sum(
            len(str(getattr(t, "whats_being_said", "") or "")) +
            len(" ".join(getattr(t, "emotional_signals", []) or []))
            for t in themes
        ) / max(1, len(themes))
    else:
        avg_chars = 0

    verbose_mode = (avg_chars >= 260) or (has_priorities and len(priorities) >= 5)

    # Layout strategy
    # - If verbose: fewer cards per page (taller boxes)
    # - First page with priorities gets even fewer cards
    #per_page_other = 4 if verbose_mode else max(4, int(themes_per_page))
    #per_page_first = 2 if (verbose_mode and has_priorities) else min(per_page_other, 4)
    # Layout strategy
    # Keep the first page smaller because Top priorities takes space.
    per_page_first = 2 if (verbose_mode and has_priorities) else 4
    
    # Every other page: 6 cards (3 rows x 2 cols)
    per_page_other = 6


    # Split themes into pages
    pages: List[List[Theme]] = []
    if has_priorities and len(themes) > per_page_first:
        pages.append(themes[:per_page_first])
        rest = themes[per_page_first:]
        for i in range(0, len(rest), per_page_other):
            pages.append(rest[i:i + per_page_other])
    else:
        for i in range(0, len(themes), per_page_other):
            pages.append(themes[i:i + per_page_other])

    total_pages = len(pages) if pages else 1

    for page_i, page_themes in enumerate(pages, start=1):
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")

        header = title if total_pages == 1 else f"{title} (Page {page_i}/{total_pages})"
        ax.text(0.02, 0.97, header, ha="left", va="top", fontsize=16, fontweight="bold", transform=ax.transAxes)

        y = 0.92

        # Top priorities (first page only)
        if page_i == 1 and has_priorities:
            ax.text(0.02, y, "Top priorities", ha="left", va="top",
                    fontsize=12, fontweight="bold", transform=ax.transAxes)
            y -= _line_h_axes(fig, 12) + 0.010

            # Full width priorities block
            wrap_w = 140 if verbose_mode else 125
            lh = _line_h_axes(fig, 10)

            for p in priorities:
                lines = _wrap_lines(str(p), width=wrap_w)
                if not lines:
                    continue

                # Draw bullet + hanging indent line-by-line (prevents stray '-' artifacts)
                for li, line in enumerate(lines):
                    prefix = "- " if li == 0 else "  "
                    ax.text(0.02, y, prefix + line, ha="left", va="top", fontsize=10, transform=ax.transAxes)
                    y -= lh + 0.002

                y -= 0.006  # space between bullets

            y -= 0.012  # space before cards

        # Notes area reserved on last page only
        show_notes = (page_i == total_pages and bool((getattr(insights, "notes", "") or "").strip()))
        bottom_margin = 0.12 if show_notes else 0.06

        if not page_themes:
            if show_notes:
                ax.text(0.02, 0.10, "Notes", ha="left", va="top",
                        fontsize=11, fontweight="bold", transform=ax.transAxes)
                note_lines = _truncate_lines(_wrap_lines(insights.notes, width=120), max_lines=3)
                ax.text(0.02, 0.075, "\n".join(note_lines), ha="left", va="top", fontsize=9, transform=ax.transAxes)

            fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            pdf.savefig(fig)
            plt.close(fig)
            continue

        # Cards grid
        n_cards = len(page_themes)
        rows = int(math.ceil(n_cards / 2.0))

        margin_x = 0.02
        col_gap_x = 0.02
        gap_y = 0.024 if verbose_mode else 0.020

        card_w = (1.0 - 2 * margin_x - col_gap_x) / 2.0
        x_left = margin_x
        x_right = x_left + card_w + col_gap_x

        #------
        available_h = max(0.10, y - bottom_margin)
        card_h = (available_h - (rows - 1) * gap_y) / rows
        #card_h = max(card_h, 0.22 if verbose_mode else 0.18)
        # Allow smaller cards so 3 rows can fit (6 per page)
        card_h = max(card_h, 0.16 if verbose_mode else 0.15)

        #-
        #available_h = max(0.10, y - bottom_margin)
        #card_h = (available_h - (rows - 1) * gap_y) / rows
        
        # Minimum height so text has room, but do not let cards grow huge
        #min_card_h = 0.20 if verbose_mode else 0.17
        #max_card_h = 0.24 if verbose_mode else 0.21
        
        #card_h = max(card_h, min_card_h)
        #card_h = min(card_h, max_card_h)
        #-----

        base_idx = sum(len(p) for p in pages[:page_i - 1])

        # Font sizes tuned for verbose blocks
        fs_title = 11
        fs_meta = 9
        fs_body = 8.7 if verbose_mode else 9
        fs_sig = 8.2 if verbose_mode else 8.5

        lh_title = _line_h_axes(fig, fs_title)
        lh_meta = _line_h_axes(fig, fs_meta)
        lh_body = _line_h_axes(fig, fs_body)
        lh_sig = _line_h_axes(fig, fs_sig)

        idx_in_page = 0
        for r in range(rows):
            row_top = y - r * (card_h + gap_y)

            for c in range(2):
                if idx_in_page >= n_cards:
                    break

                t = page_themes[idx_in_page]
                idx_in_page += 1
                idx = base_idx + idx_in_page

                x = x_left if c == 0 else x_right
                style = _crit_style(getattr(t, "criticality", "LOW"))

                rect = patches.FancyBboxPatch(
                    (x, row_top - card_h),
                    card_w, card_h,
                    boxstyle="round,pad=0.008,rounding_size=0.01",
                    linewidth=1.2,
                    edgecolor=style["edge"],
                    facecolor=style["face"],
                    transform=ax.transAxes,
                )
                ax.add_patch(rect)

                pad_x = 0.012
                pad_top = 0.016
                pad_bot = 0.014

                inner_top = row_top - pad_top
                inner_bot = (row_top - card_h) + pad_bot
                cur_y = inner_top

                # Title (max 2 lines)
                title_line = f"{idx}) {str(getattr(t, 'title', '') or '').strip()}"
                title_lines = _truncate_lines(_wrap_lines(title_line, width=54), max_lines=2)
                if title_lines:
                    ax.text(x + pad_x, cur_y, "\n".join(title_lines),
                            ha="left", va="top", fontsize=fs_title, fontweight="bold", transform=ax.transAxes)
                    cur_y -= len(title_lines) * (lh_title + 0.001) + 0.006

                # Meta (1 line)
                meta = f"Frequency: {getattr(t, 'frequency', '')}   |   Criticality: {getattr(t, 'criticality', '')}"
                ax.text(x + pad_x, cur_y, meta, ha="left", va="top", fontsize=fs_meta, transform=ax.transAxes)
                cur_y -= lh_meta + 0.010

                # Reserve at least 1 line for Signals if present
                emos = [str(e).strip() for e in (getattr(t, "emotional_signals", []) or []) if str(e).strip()]
                emos = emos[:4] if verbose_mode else emos[:6]
                reserve_for_signals = (lh_sig * 1.2 + 0.006) if emos else 0.0

                # Body: truncate to fit
                wbs = str(getattr(t, "whats_being_said", "") or "").strip()
                wbs_lines = _wrap_lines(wbs, width=78)
                if wbs_lines and cur_y > inner_bot:
                    room = max(0.0, (cur_y - inner_bot) - reserve_for_signals)
                    max_lines = int(room / (lh_body + 0.001))
                    max_lines = max(2, max_lines)
                    wbs_lines = _truncate_lines(wbs_lines, max_lines=max_lines)

                    ax.text(x + pad_x, cur_y, "\n".join(wbs_lines),
                            ha="left", va="top", fontsize=fs_body, transform=ax.transAxes)
                    cur_y -= len(wbs_lines) * (lh_body + 0.001) + 0.008

                # Signals: only if there is room
                if emos and cur_y > inner_bot + lh_sig:
                    sig_line = "Signals: " + ", ".join(emos)
                    sig_lines = _wrap_lines(sig_line, width=78)
                    # Keep signals short visually
                    sig_lines = _truncate_lines(sig_lines, max_lines=2 if verbose_mode else 3)

                    ax.text(x + pad_x, cur_y, "\n".join(sig_lines),
                            ha="left", va="top", fontsize=fs_sig, transform=ax.transAxes)

        # Notes on last page
        if show_notes:
            ax.text(0.02, 0.10, "Notes", ha="left", va="top",
                    fontsize=11, fontweight="bold", transform=ax.transAxes)
            note_lines = _truncate_lines(_wrap_lines(insights.notes, width=120), max_lines=3)
            ax.text(0.02, 0.075, "\n".join(note_lines),
                    ha="left", va="top", fontsize=9, transform=ax.transAxes)

        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        pdf.savefig(fig)
        plt.close(fig)




#End AI Helpers
#----------------------------


# -----------------------------
# Builders for low ratings / NO answers
# -----------------------------
def build_low_ratings_table(
    df_group: pd.DataFrame,
    rating_indices: List[int],
    respondent_name_index: int,
    max_star: int = 3,
) -> Optional[pd.DataFrame]:
    if respondent_name_index < 0 or respondent_name_index >= len(df_group.columns):
        return None

    cols = list(df_group.columns)
    out_cols: Dict[str, List[str]] = {}
    max_len = 0

    names_all = _clean_series_as_str_keep_len(df_group.iloc[:, respondent_name_index])

    for idx in rating_indices:
        if idx < 0 or idx >= len(cols):
            continue

        q_name = str(cols[idx])
        ratings_all = pd.to_numeric(df_group.iloc[:, idx], errors="coerce")

        entries: List[str] = []
        for n, v in zip(names_all, ratings_all):
            if not n or pd.isna(v):
                continue
            try:
                rv = int(round(float(v)))
            except Exception:
                continue
            if 1 <= rv <= max_star:
                entries.append(f"{n}, ({rv}*)")

        out_cols[q_name] = entries
        max_len = max(max_len, len(entries))

    if not out_cols:
        return None

    if max_len == 0:
        for k in list(out_cols.keys()):
            out_cols[k] = [""]
        return pd.DataFrame(out_cols)

    for k in list(out_cols.keys()):
        vals = out_cols[k]
        out_cols[k] = vals + [""] * (max_len - len(vals))

    return pd.DataFrame(out_cols)


def build_no_answers_table(
    df_group: pd.DataFrame,
    yesno_indices: List[int],
    respondent_name_index: int,
) -> Optional[pd.DataFrame]:
    if respondent_name_index < 0 or respondent_name_index >= len(df_group.columns):
        return None

    cols = list(df_group.columns)
    out_cols: Dict[str, List[str]] = {}
    max_len = 0

    names_all = _clean_series_as_str_keep_len(df_group.iloc[:, respondent_name_index])

    for idx in yesno_indices:
        if idx < 0 or idx >= len(cols):
            continue

        q_name = str(cols[idx])
        series_all = _clean_series_as_str_keep_len(df_group.iloc[:, idx]).str.upper()

        entries: List[str] = []
        for n, v in zip(names_all, series_all):
            if not n:
                continue
            if v in NO_SET:
                entries.append(n)

        out_cols[q_name] = entries
        max_len = max(max_len, len(entries))

    if not out_cols:
        return None

    if max_len == 0:
        for k in list(out_cols.keys()):
            out_cols[k] = [""]
        return pd.DataFrame(out_cols)

    for k in list(out_cols.keys()):
        vals = out_cols[k]
        out_cols[k] = vals + [""] * (max_len - len(vals))

    return pd.DataFrame(out_cols)


def _filter_low_df_by_max_star(low_df: pd.DataFrame, max_star: int = 2) -> pd.DataFrame:
    pattern = re.compile(r"\((\d)\*\)")
    new_cols: Dict[str, List[str]] = {}
    max_len = 0

    for colname in low_df.columns:
        filtered: List[str] = []
        for val in low_df[colname]:
            s = str(val).strip()
            if not s:
                continue
            m = pattern.search(s)
            if m:
                rating = int(m.group(1))
                if rating <= max_star:
                    filtered.append(s)
        new_cols[colname] = filtered
        max_len = max(max_len, len(filtered))

    if max_len == 0:
        for colname in new_cols:
            new_cols[colname] = [""]
        return pd.DataFrame(new_cols)

    for colname, vals in new_cols.items():
        new_cols[colname] = vals + [""] * (max_len - len(vals))

    return pd.DataFrame(new_cols)


# -----------------------------
# Plot metadata
# -----------------------------
def _build_plot_metadata(profile: SurveyProfile, df_group: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = list(df_group.columns)

    rating_indices = [i for i in (profile.rating_col_indices or []) if i < len(cols)]
    yesno_indices = [i for i in (profile.yesno_col_indices or []) if i < len(cols)]
    has_choice = profile.choice_col_index is not None and int(profile.choice_col_index) < len(cols)

    meta: List[Dict[str, Any]] = []
    number = 1

    for idx in rating_indices:
        meta.append({"ptype": "rating", "idx": idx, "col_name": cols[idx], "number": number})
        number += 1

    for idx in yesno_indices:
        meta.append({"ptype": "yesno", "idx": idx, "col_name": cols[idx], "number": number})
        number += 1

    if has_choice:
        cidx = int(profile.choice_col_index)
        meta.append({"ptype": "choice", "idx": cidx, "col_name": cols[cidx], "number": number})

    return meta



# -----------------------------
# Page: charts grid (page 1 per group)
# -----------------------------
def _add_group_charts_page_to_pdf(
    pdf: PdfPages,
    profile: SurveyProfile,
    df_group: pd.DataFrame,
    title_label: str,
    cycle_label: str,
    plots_meta: List[Dict[str, Any]],
) -> None:
    if not plots_meta:
        return

    n_resp = _get_unique_respondent_count(df_group, profile.respondent_name_index)
    noun = profile.respondent_singular if n_resp == 1 else profile.respondent_plural
    n_text = f" ({n_resp} {noun})"

    n_plots = len(plots_meta)

    if n_plots <= 4:
        ncols = 2
    elif n_plots <= 9:
        ncols = 3
    else:
        ncols = 4

    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 8.5))

    axes = np.array(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        if nrows == 1:
            axes = axes.reshape(1, ncols)
        else:
            axes = axes.reshape(nrows, 1)

    axes_flat = axes.flatten()

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    y_label = f"{profile.respondent_singular.capitalize()} Count"

    # Better wrapping for long Excel headers + smaller font
    wrap_width = 32 if ncols == 4 else 50
    title_fs = 6 if ncols == 4 else 7  # small enough to fit long questions

    for ax, meta in zip(axes_flat, plots_meta):
        ptype = meta["ptype"]
        idx = meta["idx"]
        number = meta["number"]

        # Big chart number in corner stays the same
        ax.text(
            0.02, 0.98, str(number),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10, fontweight="bold",
        )

        # THE FIX:
        # Always pull the title from the original Excel header in df_group (NOT profile.chart_labels)
        try:
            display_title = str(df_group.columns[idx])
        except Exception:
            display_title = str(meta.get("col_name", ""))

        wrapped_title = textwrap.fill(display_title, width=wrap_width)

        if ptype == "rating":
            series = pd.to_numeric(df_group.iloc[:, idx], errors="coerce").dropna()
            counts = series.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)

            ax.bar([1, 2, 3, 4, 5], counts.values)

            avg = series.mean() if not series.empty else None
            if avg is None or np.isnan(avg):
                title = wrapped_title
            else:
                title = f"{wrapped_title}\n(Avg = {avg:.2f})"

            ax.set_title(title, fontsize=title_fs)
            ax.set_xlabel("# of Stars", fontsize=8)
            ax.set_ylabel(y_label, fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_ylim(0, max(counts.values.tolist() + [1]) * 1.2)

        elif ptype == "yesno":
            series = _clean_series_as_str_keep_len(df_group.iloc[:, idx]).str.upper()
            yes_count = int(series.isin(YES_SET).sum())
            no_count = int(series.isin(NO_SET).sum())

            data = [yes_count, no_count]
            labels = ["YES", "NO"]

            if yes_count + no_count == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=9)
                ax.axis("off")
            else:
                def make_label(pct, allvals=data):
                    total = sum(allvals)
                    count = int(round(pct * total / 100.0)) if total else 0
                    return f"{pct:.0f}%, {count}"

                ax.pie(data, labels=labels, autopct=make_label, textprops={"fontsize": 8})
                ax.set_title(wrapped_title, fontsize=title_fs)

        elif ptype == "choice":
            series = df_group.iloc[:, idx].dropna().astype(str).str.strip()
            counts = series.value_counts()
            data = counts.values
            labels = counts.index.tolist()

            if len(data) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=9)
                ax.axis("off")
            else:
                def make_label(pct, allvals=data):
                    total = sum(allvals)
                    count = int(round(pct * total / 100.0)) if total else 0
                    return f"{pct:.0f}%, {count}"

                ax.pie(data, labels=labels, autopct=make_label, textprops={"fontsize": 8})
                ax.set_title(wrapped_title, fontsize=title_fs)

    full_title = _compose_group_title(profile, title_label, cycle_label) + n_text
    fig.suptitle(full_title, fontsize=14, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)



# -----------------------------
# Respondents grid
# -----------------------------
def _build_all_respondents_grid(
    df_group: pd.DataFrame,
    respondent_name_index: int,
    max_cols: int = 8,
) -> Optional[pd.DataFrame]:
    """
    Exactly max_cols columns per row (default 8).
    Row 2 exists only if needed.
    IMPORTANT: Uses blank column names so you NEVER see c0/c1/c2 headers.
    """
    if respondent_name_index < 0 or respondent_name_index >= len(df_group.columns):
        return None

    names = _clean_series_as_str_dropna(df_group.iloc[:, respondent_name_index])
    names = names[names != ""]
    if names.empty:
        return None

    names = names.drop_duplicates(keep="first").reset_index(drop=True)

    n = int(len(names))
    ncols = int(max_cols)
    nrows = 1 if n <= ncols else int(np.ceil(n / ncols))

    grid = [["" for _ in range(ncols)] for _ in range(nrows)]
    for i in range(n):
        r = i // ncols
        c = i % ncols
        grid[r][c] = str(names.iloc[i])

    # BLANK COLUMN NAMES (this prevents the c0/c1 header issue permanently)
    out = pd.DataFrame(grid, columns=[""] * ncols)
    out = out[(out != "").any(axis=1)]
    return out


# -----------------------------
# Page: tables (page 2 per group)
# -----------------------------
def _add_group_tables_page_to_pdf(
    pdf: PdfPages,
    profile: SurveyProfile,
    df_group: pd.DataFrame,
    title_label: str,
    cycle_label: str,
    plots_meta: List[Dict[str, Any]],
    is_all_teams: bool,
) -> None:
    n_resp = _get_unique_respondent_count(df_group, profile.respondent_name_index)
    noun = profile.respondent_singular if n_resp == 1 else profile.respondent_plural
    n_text = f" ({n_resp} {noun})"
    base_title = _compose_group_title(profile, title_label, cycle_label) + n_text + " (Details)"

    rating_indices = [m["idx"] for m in plots_meta if m["ptype"] == "rating"]
    yesno_indices = [m["idx"] for m in plots_meta if m["ptype"] == "yesno"]

    rating_number_by_name = {m["col_name"]: m["number"] for m in plots_meta if m["ptype"] == "rating"}
    yesno_number_by_name = {m["col_name"]: m["number"] for m in plots_meta if m["ptype"] == "yesno"}

    # -----------------------------
    # Build low ratings table
    # -----------------------------
    low_df = None
    low_labels = None
    if rating_indices:
        low_df = build_low_ratings_table(
            df_group,
            rating_indices=rating_indices,
            respondent_name_index=profile.respondent_name_index,
            max_star=3,
        )

        # All Teams page uses stricter cutoff (1-2 stars)
        if low_df is not None and is_all_teams:
            low_df = _filter_low_df_by_max_star(low_df, max_star=2)

        if low_df is not None:
            low_labels = []
            for colname in low_df.columns:
                num = rating_number_by_name.get(colname)
                if num is not None and profile.chart_labels and num in profile.chart_labels:
                    low_labels.append(profile.chart_labels[num])
                else:
                    low_labels.append(str(colname))

    # -----------------------------
    # Build NO answers table
    # -----------------------------
    no_df = None
    no_labels = None
    if yesno_indices:
        no_df = build_no_answers_table(
            df_group,
            yesno_indices=yesno_indices,
            respondent_name_index=profile.respondent_name_index,
        )
        if no_df is not None:
            no_labels = []
            for colname in no_df.columns:
                num = yesno_number_by_name.get(colname)
                if num is not None and profile.chart_labels and num in profile.chart_labels:
                    no_labels.append(profile.chart_labels[num])
                else:
                    no_labels.append(str(colname))

    # -----------------------------
    # Completion (All Teams) OR respondents grid (team pages)
    # -----------------------------
    completion_df = None
    respondents_df = None

    if is_all_teams:
        completion_df = pd.DataFrame(
            {"Metric": [f"{profile.respondent_plural.capitalize()} who completed this survey"], "Value": [n_resp]}
        )
    else:
        max_cols = 8 if profile.key.lower() == "families" else 6
        respondents_df = _build_all_respondents_grid(
            df_group,
            respondent_name_index=profile.respondent_name_index,
            max_cols=max_cols,
        )

    if low_df is None and no_df is None and completion_df is None and respondents_df is None:
        return

    # -----------------------------
    # Helper to draw a table
    # -----------------------------
    def _draw_table(ax, df, labels, title, fontsize=8, scale_y=1.35, col_widths=None, wrap=False):
        ax.axis("off")

        if df is None or getattr(df, "empty", False):
            return None

        ncols = int(df.shape[1])
        if col_widths is None:
            col_widths = [1.0 / max(ncols, 1)] * ncols

        hide_header = False
        if labels is None:
            hide_header = True
        else:
            lab_strs = [str(x).strip() for x in labels]
            if all(s == "" for s in lab_strs):
                hide_header = True

        table_kwargs = dict(
            cellText=df.values,
            loc="upper left",
            colWidths=col_widths,
        )
        if not hide_header:
            table_kwargs["colLabels"] = labels

        tbl = ax.table(**table_kwargs)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)
        tbl.scale(1.0, scale_y)

        if title:
            ax.set_title(title, fontsize=10, pad=6)

        if wrap:
            for (r, c), cell in tbl.get_celld().items():
                # if header exists, format it
                if (not hide_header) and r == 0:
                    cell.set_text_props(ha="center", va="center", fontweight="bold")
                    continue

                if c == 0:
                    cell.set_text_props(ha="center", va="top")
                else:
                    txt = cell.get_text()
                    txt.set_wrap(True)
                    txt.set_ha("left")
                    txt.set_va("top")
                    cell.PAD = 0.02

        return tbl

    # -----------------------------
    # Special behavior:
    # Families TEAM pages only (NOT All Teams):
    # combine into ONE page: low(1-8) + low(9-16 with Q16 merged) + respondents
    # -----------------------------
    MAX_COLS_PER_PAGE = 8
    is_families = (profile.key.lower() == "families")

    if (
        is_families
        and (not is_all_teams)
        and (low_df is not None)
        and (len(low_df.columns) > MAX_COLS_PER_PAGE)
    ):
        ncols_total = len(low_df.columns)

        low_1 = low_df.iloc[:, 0:MAX_COLS_PER_PAGE]
        labels_1 = low_labels[0:MAX_COLS_PER_PAGE] if low_labels is not None else list(low_1.columns)

        low_2 = low_df.iloc[:, MAX_COLS_PER_PAGE:ncols_total]
        labels_2 = low_labels[MAX_COLS_PER_PAGE:ncols_total] if low_labels is not None else list(low_2.columns)

        # merge Q16 (NO replies) into last column of the second block
        if (no_df is not None) and (no_labels is not None) and (len(no_df.columns) == 1):
            max_rows = max(len(low_2), len(no_df))
            low_2 = low_2.reindex(range(max_rows)).fillna("")
            no_col = no_df.iloc[:, 0].reindex(range(max_rows)).fillna("").astype(str)

            low_2 = low_2.copy()
            low_2[str(no_df.columns[0])] = no_col.values
            labels_2 = list(labels_2) + [no_labels[0]]

        sections = [
            ("low1", low_1, list(labels_1)),
            ("low2", low_2, list(labels_2)),
        ]
        if respondents_df is not None:
            sections.append(("respondents", respondents_df, list(respondents_df.columns)))

        height_ratios = [1.1, 1.1] + ([0.75] if respondents_df is not None else [])

        fig, axes = plt.subplots(
            nrows=len(sections),
            ncols=1,
            figsize=(11, 8.5),
            gridspec_kw={"height_ratios": height_ratios},
        )
        if len(sections) == 1:
            axes = [axes]

        for ax, (key, df_sec, labels_sec) in zip(axes, sections):
            if key == "low1":
                _draw_table(
                    ax, df_sec, labels_sec,
                    title="1-3 Star Reviews (columns = chart numbers) (1-8)",
                    fontsize=8, scale_y=1.25
                )
            elif key == "low2":
                _draw_table(
                    ax, df_sec, labels_sec,
                    title="1-3 Star Reviews (columns = chart numbers) (9-16)",
                    fontsize=8, scale_y=1.25
                )
            elif key == "respondents":
                _draw_table(
                    ax, df_sec, list(df_sec.columns),
                    title="Families who completed this survey",
                    fontsize=8, scale_y=1.6
                )

        fig.suptitle(base_title, fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.92])
        fig.subplots_adjust(hspace=0.55)
        pdf.savefig(fig)
        plt.close(fig)
        return

    # -----------------------------
    # OLD wide-table split behavior:
    # keeps Families ALL TEAMS split into 2 pages like the old version
    # -----------------------------
    if low_df is not None and len(low_df.columns) > MAX_COLS_PER_PAGE:
        ncols_total = len(low_df.columns)

        for start in range(0, ncols_total, MAX_COLS_PER_PAGE):
            end = min(start + MAX_COLS_PER_PAGE, ncols_total)
            is_last_chunk = (end == ncols_total)

            low_chunk = low_df.iloc[:, start:end]
            low_chunk_labels = low_labels[start:end] if low_labels is not None else list(low_chunk.columns)

            merged_low = low_chunk
            merged_labels = list(low_chunk_labels)

            # merge Q16 NO into last chunk only (Families)
            do_merge_q16 = (
                is_last_chunk
                and (profile.key.lower() == "families")
                and (no_df is not None)
                and (no_labels is not None)
                and (len(no_df.columns) == 1)
            )

            if do_merge_q16:
                max_rows = max(len(merged_low), len(no_df))
                merged_low = merged_low.reindex(range(max_rows)).fillna("")
                no_col = no_df.iloc[:, 0].reindex(range(max_rows)).fillna("").astype(str)

                merged_low = merged_low.copy()
                merged_low[str(no_df.columns[0])] = no_col.values
                merged_labels.append(no_labels[0])

            sections = [("low", merged_low, merged_labels)]

            if is_last_chunk:
                if is_all_teams and completion_df is not None:
                    sections.append(("completion", completion_df, list(completion_df.columns)))
                if (not is_all_teams) and (respondents_df is not None):
                    sections.append(("respondents", respondents_df, list(respondents_df.columns)))

            height_ratios = []
            for key, *_ in sections:
                if key == "low":
                    height_ratios.append(1.35)
                elif key == "completion":
                    height_ratios.append(0.65)
                elif key == "respondents":
                    height_ratios.append(1.0)
                else:
                    height_ratios.append(1.0)

            fig, axes = plt.subplots(
                nrows=len(sections),
                ncols=1,
                figsize=(11, 8.5),
                gridspec_kw={"height_ratios": height_ratios},
            )
            if len(sections) == 1:
                axes = [axes]

            for ax, (key, df_sec, labels_sec) in zip(axes, sections):
                if key == "low":
                    title = ("1-2 Star Reviews (columns = chart numbers)" if is_all_teams
                             else "1-3 Star Reviews (columns = chart numbers)")
                    _draw_table(ax, df_sec, labels_sec, title=title, fontsize=8, scale_y=1.35)

                elif key == "completion":
                    _draw_table(ax, df_sec, labels_sec, title="Survey completion summary", fontsize=11, scale_y=1.35)

                elif key == "respondents":
                    _draw_table(
                        ax, df_sec, labels_sec,
                        title=f"{profile.respondent_plural.capitalize()} who completed this survey",
                        fontsize=8, scale_y=1.6
                    )

            range_str = "1-8" if start == 0 else ("9-16" if do_merge_q16 else f"{start+1}-{end}")

            fig.suptitle(f"{base_title} - Low Ratings ({range_str})", fontsize=12)
            fig.tight_layout(rect=[0, 0.03, 1, 0.92])
            fig.subplots_adjust(hspace=0.55)
            pdf.savefig(fig)
            plt.close(fig)

        return

    # -----------------------------
    # Not wide: single page fallback (no comments)
    # -----------------------------
    sections = []
    if low_df is not None:
        sections.append("low")
    if no_df is not None:
        sections.append("no")

    if is_all_teams:
        if completion_df is not None:
            sections.append("completion")
    else:
        if respondents_df is not None:
            sections.append("respondents")

    height_ratios = []
    for s in sections:
        if s == "low":
            height_ratios.append(1.2)
        elif s == "no":
            height_ratios.append(0.9)
        elif s == "completion":
            height_ratios.append(0.7)
        elif s == "respondents":
            height_ratios.append(1.1)
        else:
            height_ratios.append(1.0)

    fig, axes = plt.subplots(
        nrows=len(sections),
        ncols=1,
        figsize=(11, 8.5),
        gridspec_kw={"height_ratios": height_ratios},
    )
    if len(sections) == 1:
        axes = [axes]

    row_idx = 0

    if low_df is not None:
        _draw_table(
            axes[row_idx],
            low_df,
            low_labels if low_labels is not None else list(low_df.columns),
            title=("1-2 Star Reviews (columns = chart numbers)" if is_all_teams
                   else "1-3 Star Reviews (columns = chart numbers)"),
            fontsize=7,
            scale_y=1.2,
        )
        row_idx += 1

    if no_df is not None:
        _draw_table(
            axes[row_idx],
            no_df,
            no_labels if no_labels is not None else list(no_df.columns),
            title='"NO" Replies (columns = chart numbers)',
            fontsize=8,
            scale_y=1.2,
        )
        row_idx += 1

    if is_all_teams and completion_df is not None:
        _draw_table(
            axes[row_idx],
            completion_df,
            list(completion_df.columns),
            title="Survey completion summary",
            fontsize=11,
            scale_y=1.35,
        )
        row_idx += 1

    if (not is_all_teams) and (respondents_df is not None):
        _draw_table(
            axes[row_idx],
            respondents_df,
            list(respondents_df.columns),
            title=f"{profile.respondent_plural.capitalize()} who completed this survey",
            fontsize=8,
            scale_y=1.6,
        )

    fig.suptitle(base_title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.subplots_adjust(hspace=0.55)
    pdf.savefig(fig)
    plt.close(fig)


# -----------------------------
# Page: cycle summary (page 1)  -  AI Addition
# -----------------------------
def _add_cycle_summary_page(
    pdf: PdfPages,
    profile: SurveyProfile,
    df: pd.DataFrame,
    cycle_label: str,
) -> None:
    cols = list(df.columns)
    if profile.group_col_index >= len(cols):
        raise ValueError("Group column index is outside the available columns.")

    group_col_name = df.columns[profile.group_col_index]

    sample = df[group_col_name].dropna().astype(str).str.strip().head(80)
    frac_between_1_5 = _fraction_numeric_between_1_and_5(sample)
    if frac_between_1_5 > 0.60:
        raise ValueError(
            "BUG: Group column values look like 1-5 ratings, so teams become '5', '4', etc. "
            f"Fix profiles.py: group_col_index currently points to '{group_col_name}'."
        )

    rating_indices = [i for i in (profile.rating_col_indices or []) if i < len(cols)]

    if profile.qq_rating_col_index is not None and int(profile.qq_rating_col_index) < len(cols):
        qq_idx = int(profile.qq_rating_col_index)
    else:
        qq_idx = rating_indices[6] if len(rating_indices) >= 7 else None

    stats_by_team: Dict[str, Tuple[int, float]] = {}

    for team_name, group_df in df.groupby(group_col_name, sort=False):
        if str(team_name).strip() == profile.unassigned_label:
            continue

        n_resp = _get_unique_respondent_count(group_df, profile.respondent_name_index)

        if qq_idx is not None and qq_idx < len(group_df.columns):
            series = pd.to_numeric(group_df.iloc[:, qq_idx], errors="coerce").dropna()
            avg_rating = float(series.mean()) if not series.empty else np.nan
        else:
            avg_rating = np.nan

        stats_by_team[str(team_name).strip()] = (n_resp, avg_rating)

    all_team_names = sorted(stats_by_team.keys())
    if not all_team_names:
        return

    rows: List[Dict[str, Any]] = []
    for team_name in all_team_names:
        coach = (profile.team_coach_map or {}).get(team_name, "?")
        count, avg_rating = stats_by_team.get(team_name, (0, np.nan))
        rows.append({"Team": team_name, "Coach": coach, "Count": count, "Rating": avg_rating})

    summary_df = pd.DataFrame(rows)

    total_responses = int(summary_df["Count"].sum())
    total_str = (
        f"{total_responses} {profile.respondent_singular}"
        if total_responses == 1
        else f"{total_responses} {profile.respondent_plural}"
    )

    summary_df["TeamCoach"] = summary_df["Team"] + " - " + summary_df["Coach"]
    summary_df["RatingStr"] = summary_df["Rating"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    summary_df["CountDisplay"] = [
        _format_count_pct_cell(profile, team, int(c))
        for team, c in zip(summary_df["Team"], summary_df["Count"])
    ]

    qq_vals: List[float] = []
    for team, count, rating in zip(summary_df["Team"], summary_df["Count"], summary_df["Rating"]):
        if pd.isna(rating):
            qq_vals.append(0.0)
            continue

        denom_map = profile.denominator_map
        if denom_map and denom_map.get(team) and denom_map.get(team) > 0:
            frac = float(count) / float(denom_map[team])
        else:
            frac = 1.0

        qq_vals.append(float(rating) * frac)

    summary_df["QQIndex"] = qq_vals
    summary_df = summary_df.sort_values("QQIndex", ascending=False, ignore_index=True)

    fig, (ax_table, ax_bar) = plt.subplots(
        1, 2,
        figsize=(11, 8.5),
        gridspec_kw={"width_ratios": [1.2, 1.8]},
    )

    who_title = profile.respondent_plural.capitalize()
    fig.suptitle(f"{cycle_label} Summary - {who_title}", fontsize=14, fontweight="bold")

    ax_table.axis("off")
    display_df = summary_df[["TeamCoach", "CountDisplay", "RatingStr"]]

    tbl = ax_table.table(
        cellText=display_df.values,
        colLabels=["Team - Coach", who_title, "Rating"],
        loc="center",
        colWidths=[0.72, 0.14, 0.14],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.1, 1.2)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(ha="center", va="center", fontweight="bold")
        else:
            cell.set_text_props(ha=("left" if c == 0 else "center"), va="center")

    ax_bar.set_title(f"{cycle_label} QQ (Quality-Quantity) Index - {total_str}", fontsize=10)
    y_pos = np.arange(len(summary_df))
    ax_bar.barh(y_pos, summary_df["QQIndex"].values.astype(float), height=0.6, label="QQ index")
    ax_bar.set_xlabel("QQ index (rating * completion fraction)")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(summary_df["TeamCoach"], fontsize=6)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 5.1)
    ax_bar.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _wrap(s: str, width: int) -> str:
    s = (s or "").strip()
    return textwrap.fill(s, width=width)

def _crit_rank(c: str) -> int:
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    return order.get((c or "").strip().upper(), 9)

def _add_all_teams_comments_insights_page_to_pdf(
    pdf,                 # PdfPages
    profile,
    df,
    cycle_label: str,
    model: str = "gpt-5-mini",
    chunk_size: int = 60,
) -> None:
    client = _try_get_openai_client()
    if client is None:
        return

    comment_cols = _infer_comment_col_indices(profile, df)
    if not comment_cols:
        return

    comments = _collect_comments(df, comment_cols)
    if not comments:
        return

    insights = _llm_summarize_comments_structured(
        client,
        comments,
        model=model,
        chunk_size=chunk_size,
    )
    if insights is None or not insights.themes:
        return

    title = f"All Teams - {cycle_label} - Comments Insights (CARDS_V1)"
    _add_comments_insights_cards_to_pdf(
        pdf,
        title=title,
        insights=insights,
        themes_per_page=6,
    )

# -----------------------------
# Main entry
# -----------------------------
def create_pdf_report(
    input_path: str,
    cycle_label: str = "Cycle",
    survey_type: str = "players",
    output_path: Optional[str] = None,
) -> str:
    profile = PROFILES.get(survey_type.lower().strip())
    if profile is None:
        raise ValueError(f"Unknown survey_type: {survey_type}. Use one of: {list(PROFILES.keys())}")

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + f"_{profile.key}_report.pdf"

    df = pd.read_excel(input_path, sheet_name=0)

    _assert_profile_and_df_make_sense(profile, df)
    group_col_name = _normalize_group_column_inplace(df, profile)

    cols = list(df.columns)
    rating_indices = [i for i in (profile.rating_col_indices or []) if i < len(cols)]

    if profile.qq_rating_col_index is not None and int(profile.qq_rating_col_index) < len(cols):
        qq_idx = int(profile.qq_rating_col_index)
    else:
        qq_idx = rating_indices[6] if len(rating_indices) >= 7 else None

    stats_rows: List[Dict[str, Any]] = []
    for team, group_df in df.groupby(group_col_name, sort=False):
        team = str(team).strip()
        if team == profile.unassigned_label:
            continue

        count = _get_unique_respondent_count(group_df, profile.respondent_name_index)

        if qq_idx is not None and qq_idx < len(group_df.columns):
            series = pd.to_numeric(group_df.iloc[:, qq_idx], errors="coerce").dropna()
            avg = float(series.mean()) if not series.empty else np.nan
        else:
            avg = np.nan

        stats_rows.append({"Team": team, "Count": count, "Avg": avg})

    if not stats_rows:
        return output_path

    stats_df = pd.DataFrame(stats_rows)

    qq_vals: List[float] = []
    for team, count, avg in zip(stats_df["Team"], stats_df["Count"], stats_df["Avg"]):
        if pd.isna(avg):
            qq_vals.append(0.0)
            continue

        denom_map = profile.denominator_map
        if denom_map and denom_map.get(team) and denom_map.get(team) > 0:
            frac = float(count) / float(denom_map[team])
        else:
            frac = 1.0

        qq_vals.append(float(avg) * frac)

    stats_df["QQIndex"] = qq_vals
    stats_df = stats_df.sort_values("QQIndex", ascending=False, ignore_index=True)
    qq_sorted_teams = list(stats_df["Team"].values)

    grouped: Dict[str, pd.DataFrame] = {
        str(team).strip(): sub_df for team, sub_df in df.groupby(group_col_name, sort=False)
    }

    with PdfPages(output_path) as pdf:
        _add_cycle_summary_page(pdf, profile, df, cycle_label)

        all_meta = _build_plot_metadata(profile, df)
        _add_group_charts_page_to_pdf(pdf, profile, df, "All Teams", cycle_label, all_meta)
        _add_group_tables_page_to_pdf(pdf, profile, df, "All Teams", cycle_label, all_meta, is_all_teams=True)

        # AI page (All Teams only)
        _add_all_teams_comments_insights_page_to_pdf(
            pdf,
            profile,
            df,
            cycle_label=cycle_label,
            model="gpt-5-mini",
            chunk_size=60,
        )



        for team in qq_sorted_teams:
            group_df = grouped.get(team)
            if group_df is None:
                continue

            coach = (profile.team_coach_map or {}).get(team, "?")
            title_label = f"{team} - {coach}"

            meta = _build_plot_metadata(profile, group_df)
            _add_group_charts_page_to_pdf(pdf, profile, group_df, title_label, cycle_label, meta)
            _add_group_tables_page_to_pdf(pdf, profile, group_df, title_label, cycle_label, meta, is_all_teams=False)

    return output_path


def create_pdf_from_original(
    input_path: str,
    cycle_label: str = "Cycle",
    output_path: Optional[str] = None,
) -> str:
    return create_pdf_report(
        input_path=input_path,
        cycle_label=cycle_label,
        survey_type="players",
        output_path=output_path,
    )

