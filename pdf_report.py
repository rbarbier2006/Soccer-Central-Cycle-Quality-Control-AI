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
# (LANDSCAPE so it's same size as the other pages)
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

    # Grid rule:
    # - Up to 4 plots: 2 columns
    # - 5 to 9 plots: 3 columns
    # - 10+ plots: 4 columns (Families 16 -> 4x4)
    if n_plots <= 4:
        ncols = 2
    elif n_plots <= 9:
        ncols = 3
    else:
        ncols = 4

    nrows = int(np.ceil(n_plots / ncols))

    # IMPORTANT: landscape so it matches the other pages' "size"
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 8.5))

    axes = np.array(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        # One row or one column
        if nrows == 1:
            axes = axes.reshape(1, ncols)
        else:
            axes = axes.reshape(nrows, 1)

    axes_flat = axes.flatten()

    # Turn off unused axes
    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    y_label = f"{profile.respondent_singular.capitalize()} Count"

    # Title wrap tighter when there are 4 columns
    wrap_width = 26 if ncols == 4 else 40

    for ax, meta in zip(axes_flat, plots_meta):
        ptype = meta["ptype"]
        idx = meta["idx"]
        col_name = meta["col_name"]
        number = meta["number"]

        # Big chart number in corner
        ax.text(
            0.02, 0.98, str(number),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10, fontweight="bold",
        )

        # Use profile.chart_labels when available
        display_title = None
        if profile.chart_labels and number in profile.chart_labels:
            display_title = str(profile.chart_labels[number])
        else:
            display_title = str(col_name)

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

            ax.set_title(title, fontsize=9)
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
                ax.set_title(wrapped_title, fontsize=9)

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
                ax.set_title(wrapped_title, fontsize=9)

    full_title = _compose_group_title(profile, title_label, cycle_label) + n_text
    fig.suptitle(full_title, fontsize=14, fontweight="bold")

    # Leave space for the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)



# -----------------------------
# Respondents grid + comments
# -----------------------------
def _build_all_respondents_grid(
    df_group: pd.DataFrame,
    respondent_name_index: int,
    max_cols: int = 8,
) -> Optional[pd.DataFrame]:
    """
    Up to 8 names per row.
    Only creates row 2 if row 1 is completely filled (i.e., more than 8 names).
    """
    if respondent_name_index < 0 or respondent_name_index >= len(df_group.columns):
        return None

    names = _clean_series_as_str_dropna(df_group.iloc[:, respondent_name_index])
    names = names[names != ""]

    if names.empty:
        return None

    # Preserve original order, unique
    names = names.drop_duplicates(keep="first").reset_index(drop=True)

    n = int(len(names))
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))

    grid = [["" for _ in range(ncols)] for _ in range(nrows)]

    # Fill row 1 left-to-right, then row 2, etc.
    for i in range(n):
        r = i // ncols
        c = i % ncols
        grid[r][c] = names.iloc[i]

    # IMPORTANT: columns are blanks; header will be auto-hidden anyway by _draw_table
    out = pd.DataFrame(grid, columns=[""] * ncols)

    # Drop fully empty rows (shouldn't happen, but safe)
    out = out[(out != "").any(axis=1)]
    return out




def _build_comments_table(
    df_group: pd.DataFrame,
    respondent_name_index: int,
) -> Optional[pd.DataFrame]:
    if respondent_name_index < 0 or respondent_name_index >= len(df_group.columns):
        return None

    cols = list(df_group.columns)
    comment_indices: List[int] = []
    for i, name in enumerate(cols):
        nl = str(name).lower()
        if "comment" in nl or "suggest" in nl:
            comment_indices.append(i)

    if not comment_indices:
        return None

    rows: List[List[str]] = []
    who_all = _clean_series_as_str_keep_len(df_group.iloc[:, respondent_name_index])

    for row_i in range(len(df_group)):
        who = str(who_all.iloc[row_i]).strip()
        if not who:
            continue

        for idx in comment_indices:
            val = df_group.iloc[row_i, idx]
            if pd.isna(val):
                continue
            txt = str(val).strip()
            if not txt:
                continue

            col_label = str(cols[idx])
            text_final = f"[{col_label}] {txt}" if len(comment_indices) > 1 else txt
            rows.append([who, text_final])

    if not rows:
        return None

    return pd.DataFrame(rows, columns=["Respondent", "Comment / Suggestion"])


# -----------------------------
# Page: tables (page 2 per group)
# Includes "two blocks" behavior for Families:
# Low Ratings split into 1-8 and 9-16, where 16 is NO replies merged in.
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
    # Build low ratings table (ratings only)
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
    # Build NO answers table (yes/no only)
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
    # Completion/respondents/comments
    # -----------------------------
    completion_df = None
    respondents_df = None
    comments_df = None

    if is_all_teams:
        completion_df = pd.DataFrame(
            {"Metric": [f"{profile.respondent_plural.capitalize()} who completed this survey"], "Value": [n_resp]}
        )
    else:
        respondents_df = _build_all_respondents_grid(
            df_group,
            respondent_name_index=profile.respondent_name_index,
            max_cols=6,
        )
        comments_df = _build_comments_table(df_group, respondent_name_index=profile.respondent_name_index)

    if (
        low_df is None and
        no_df is None and
        completion_df is None and
        respondents_df is None and
        comments_df is None
    ):
        return

    # -----------------------------
    # Helper to draw a table
    # -----------------------------

    def _draw_table(ax, df, labels, title, fontsize=8, scale_y=1.35, col_widths=None, wrap=False):
        """
        Draw a matplotlib table.
    
        Fixes:
        - Automatically removes the header row if labels look like "Respondents 1", "Respondents 2", etc.
        - You can also remove headers by passing labels=None.
        """
        ax.axis("off")
    
        if df is None or getattr(df, "empty", False):
            return None
    
        ncols = int(df.shape[1])
        if col_widths is None:
            col_widths = [1.0 / max(ncols, 1)] * ncols
    
        # Auto-hide header if labels are the "Respondents #" placeholders
        hide_header = False
        if labels is None:
            hide_header = True
        else:
            lab_strs = [str(x).strip() for x in labels]
            if all(re.fullmatch(r"Respondents\s*\d+", s) for s in lab_strs):
                hide_header = True
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
                # If header exists, format it
                if (not hide_header) and r == 0:
                    cell.set_text_props(ha="center", va="center", fontweight="bold")
                    continue
    
                # Body formatting
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
    # WIDE low_df: split into chunks of 8 columns
    # Families special rule:
    #   On the LAST chunk page, append Q16 (NO replies) as the LAST COLUMN of that SAME table.
    # -----------------------------
    MAX_COLS_PER_PAGE = 8
    if low_df is not None and len(low_df.columns) > MAX_COLS_PER_PAGE:
        ncols_total = len(low_df.columns)

        for start in range(0, ncols_total, MAX_COLS_PER_PAGE):
            end = min(start + MAX_COLS_PER_PAGE, ncols_total)
            is_last_chunk = (end == ncols_total)

            low_chunk = low_df.iloc[:, start:end]
            low_chunk_labels = low_labels[start:end] if low_labels is not None else list(low_chunk.columns)

            # ---- Families-only merge: add Q16 as last column of the low table on the LAST chunk page
            merged_low = low_chunk
            merged_labels = list(low_chunk_labels)

            do_merge_q16 = (
                is_last_chunk and
                (profile.key.lower() == "families") and
                (no_df is not None) and
                (no_labels is not None) and
                (len(no_df.columns) == 1)
            )

            if do_merge_q16:
                # ensure same row count by reindexing both to max rows
                max_rows = max(len(merged_low), len(no_df))
                merged_low = merged_low.reindex(range(max_rows)).fillna("")
                no_col = no_df.iloc[:, 0].reindex(range(max_rows)).fillna("").astype(str)

                # append as last column
                merged_low = merged_low.copy()
                merged_low[str(no_df.columns[0])] = no_col.values
                merged_labels.append(no_labels[0])  # should be (16)Q16

            # Build sections for this page
            sections = [("low", merged_low, merged_labels)]

            # On last chunk page, also add completion/respondents/comments
            if is_last_chunk:
                if is_all_teams and completion_df is not None:
                    sections.append(("completion", completion_df, list(completion_df.columns)))
                if (not is_all_teams) and (respondents_df is not None):
                    sections.append(("respondents", respondents_df, list(respondents_df.columns)))
                if (not is_all_teams) and (comments_df is not None):
                    sections.append(("comments", comments_df, list(comments_df.columns)))

            height_ratios = []
            for key, *_ in sections:
                if key == "low":
                    height_ratios.append(1.35)
                elif key == "completion":
                    height_ratios.append(0.65)
                elif key == "respondents":
                    height_ratios.append(1.0)
                elif key == "comments":
                    height_ratios.append(1.6)
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
                    # If we merged Q16 in, the title is still fine (it’s the same numbered grid)
                    _draw_table(ax, df_sec, labels_sec, title=title, fontsize=8, scale_y=1.35)

                elif key == "completion":
                    _draw_table(ax, df_sec, labels_sec, title="Survey completion summary", fontsize=11, scale_y=1.35)

                elif key == "respondents":
                    _draw_table(
                        ax, df_sec, labels_sec,
                        title=f"{profile.respondent_plural.capitalize()} who completed this survey",
                        fontsize=8, scale_y=1.6
                    )

                elif key == "comments":
                    _draw_table(
                        ax, df_sec, labels_sec,
                        title="Comments and Suggestions",
                        fontsize=8, scale_y=2.2,
                        col_widths=[0.12, 0.88],
                        wrap=True
                    )

            # Page range label
            if start == 0:
                range_str = "1-8"
            else:
                # If we merged Q16, make the label show 9-16
                range_str = "9-16" if do_merge_q16 else f"{start+1}-{end}"

            fig.suptitle(f"{base_title} - Low Ratings ({range_str})", fontsize=12)
            fig.tight_layout(rect=[0, 0.03, 1, 0.92])
            fig.subplots_adjust(hspace=0.55)
            pdf.savefig(fig)
            plt.close(fig)

        # IMPORTANT: In the wide-table path we handled everything already.
        return

    # -----------------------------
    # Not wide: single page behavior (keep original sections)
    # -----------------------------
    sections: List[str] = []
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
        if comments_df is not None:
            sections.append("comments")

    height_ratios: List[float] = []
    for s in sections:
        if s == "low":
            height_ratios.append(1.2)
        elif s == "no":
            height_ratios.append(0.9)
        elif s == "completion":
            height_ratios.append(0.7)
        elif s == "respondents":
            height_ratios.append(1.1)
        elif s == "comments":
            height_ratios.append(1.6)

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
        row_idx += 1

    if (not is_all_teams) and (comments_df is not None):
        _draw_table(
            axes[row_idx],
            comments_df,
            list(comments_df.columns),
            title="Comments and Suggestions",
            fontsize=8,
            scale_y=2.2,
            col_widths=[0.12, 0.88],
            wrap=True
        )

    fig.suptitle(base_title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.subplots_adjust(hspace=0.55)
    pdf.savefig(fig)
    plt.close(fig)



# -----------------------------
# Page: cycle summary (page 1)
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

    # Rank by QQIndex (bars + table)
    summary_df = summary_df.sort_values("QQIndex", ascending=False, ignore_index=True)

    fig, (ax_table, ax_bar) = plt.subplots(
        1, 2,
        figsize=(11, 8.5),
        gridspec_kw={"width_ratios": [1.2, 1.8]},
    )

    # Title: Cycle X Summary - Players/Families
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
