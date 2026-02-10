
# profiles.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


def col(letter: str) -> int:
    """
    Convert Excel column letter(s) to 0-based index:
    A -> 0, B -> 1, ..., Z -> 25, AA -> 26, etc.
    """
    s = letter.strip().upper()
    n = 0
    for ch in s:
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Invalid column letter: {letter}")
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


@dataclass(frozen=True)
class SurveyProfile:
    # Required (no defaults first)
    key: str
    title: str

    respondent_singular: str
    respondent_plural: str

    respondent_name_index: int
    group_col_index: int

    rating_col_indices: Tuple[int, ...]
    yesno_col_indices: Tuple[int, ...]

    # Optional (defaults after)
    choice_col_index: Optional[int] = None
    qq_rating_col_index: Optional[int] = None

    unassigned_label: str = "UNASSIGNED"

    team_coach_map: Optional[Dict[str, str]] = None
    denominator_map: Optional[Dict[str, int]] = None
    chart_labels: Optional[Dict[int, str]] = None


# -----------------------------
# Shared team metadata
# -----------------------------

TEAM_COACH_MAP: Dict[str, str] = {
    "MLS HG U19": "Michael", #previous Jorge
    "MLS HG U17": "Chris",
    "MLS HG U16": "David K",
    "MLS HG U15": "Michael A", #previous Jorge
    "MLS HG U14": "David K",
    "MLS HG U13": "Chris M",
    "MLS AD U19": "TBD", #previous Michael
    "MLS AD U17": "TBD", #previous Michael
    "MLS AD U16": "Miguel",
    "MLS AD U15": "Miguel",
    "MLS AD U14": "Junro",
    "MLS AD U13": "Miguel",
    "TX2 U19": "Jesus",
    "TX2 U17": "Fernando",
    "TX2 U16": "Jesus",
    "TX2 U15": "Claudia",
    "TX2 U14": "Rene/Claudia",
    "TX2 U13": "Claudia/Rene",
    "TX2 U12": "Armando",
    "TX2 U11": "Armando",
    "Athenians U16": "Rumen",
    "Athenians U13": "Keeley",
    "Athenians WDDOA U12": "Keeley",
    "Athenians WDDOA U11": "TBD", #previous Robert
    "Athenians PDF U10": "TBD", #previous Robert
    "Athenians PDF U9": "TBD", #previous Robert
    "WDDOA U12": "Adam",
    "WDDOA U11": "Adam",
    "PDF U10 White": "Steven",
    "PDF U9 White": "Steven",
    "PDF U10 Red": "Pablo",
    "PDF U9 Red": "Pablo",
}

TEAM_ROSTER_SIZE: Dict[str, int] = {
    "MLS HG U19": 19,
    "MLS HG U17": 19,
    "MLS HG U16": 13,
    "MLS HG U15": 12,
    "MLS HG U14": 15,
    "MLS HG U13": 17,
    "MLS AD U19": 19,
    "MLS AD U17": 17,
    "MLS AD U16": 19,
    "MLS AD U15": 18,
    "MLS AD U14": 19,
    "MLS AD U13": 15,
    "TX2 U19": 14,
    "TX2 U17": 19,
    "TX2 U16": 22,
    "TX2 U15": 22,
    "TX2 U14": 17,
    "TX2 U13": 15,
    "TX2 U12": 13,
    "TX2 U11": 11,
    "Athenians U16": 15,
    "Athenians U13": 14,
    "Athenians WDDOA U12": 8,
    "Athenians WDDOA U11": 11,
    "Athenians PDF U10": 11,
    "Athenians PDF U9": 5,
    "WDDOA U12": 10,
    "WDDOA U11": 14,
    "PDF U10 White": 8,
    "PDF U9 White": 11,
    "PDF U10 Red": 9,
    "PDF U9 Red": 8,
}


# -----------------------------
# Players profile
# -----------------------------
# Players:
# F = Player Name
# G = Player Team
# H-L = ratings (5)
# M-N = yes/no (2)
# O = multiple choice (1)
# P-Q = ratings (2)
# R = comments

PLAYERS_RATING_COLS = tuple([col("H"), col("I"), col("J"), col("K"), col("L"), col("P"), col("Q")])
PLAYERS_YESNO_COLS = tuple([col("M"), col("N")])
PLAYERS_CHOICE_COL = col("O")

PLAYERS_CHART_LABELS: Dict[int, str] = {
    1: "(1)Safety and Support",
    2: "(2)Improvement",
    3: "(3)Instructions and Feedback",
    4: "(4)Coaches Listening",
    5: "(5)Effort and Discipline",
    6: "(6)SC Value Alignment",
    7: "(7)Overall Experience",
}

PLAYERS_PROFILE = SurveyProfile(
    key="players",
    title="Players Survey",
    respondent_singular="player",
    respondent_plural="players",
    respondent_name_index=col("F"),
    group_col_index=col("G"),
    rating_col_indices=PLAYERS_RATING_COLS,
    yesno_col_indices=PLAYERS_YESNO_COLS,
    choice_col_index=PLAYERS_CHOICE_COL,
    qq_rating_col_index=col("Q"),
    team_coach_map=TEAM_COACH_MAP,
    denominator_map=TEAM_ROSTER_SIZE,
    chart_labels=PLAYERS_CHART_LABELS,
)


# -----------------------------
# Families profile
# -----------------------------
# Families:
# G = Player Name (respondent label)
# H = Player Team (grouping)
# I-J = ratings (2)
# K = yes/no (1)   -> this becomes Q16 in your report numbering
# L-X = ratings (13)
# Y = comments

FAMILIES_RATING_COLS = tuple([col("I"), col("J")] + list(range(col("L"), col("X") + 1)))
FAMILIES_YESNO_COLS = tuple([col("K")])

FAMILIES_CHART_LABELS: Dict[int, str] = {
    1:  "(1)Coaching Qual.",
    2:  "(2)Comm. clarity",
    3:  "(3)Value/cost",
    4:  "(4)Safe/clean",
    5:  "(5)Comm. matches",
    6:  "(6)Scheduling",
    7:  "(7)Events Exp.",
    8:  "(8)Satisfaction",
    9:  "(9)Discipline",
    10: "(10)Wellbeing",
    11: "(11)Resilience",
    12: "(12)Growth Mind.",
    13: "(13)Teamwork",
    14: "(14)Values",
    15: "(15)Recommend",
    16: "(16)24h reply",
}

FAMILIES_PROFILE = SurveyProfile(
    key="families",
    title="Families Survey",
    respondent_singular="family",
    respondent_plural="families",
    respondent_name_index=col("G"),
    group_col_index=col("H"),  # change if your Team column is different
    rating_col_indices=FAMILIES_RATING_COLS,
    yesno_col_indices=FAMILIES_YESNO_COLS,
    choice_col_index=None,
    qq_rating_col_index=col("Q"),
    team_coach_map=TEAM_COACH_MAP,
    denominator_map=TEAM_ROSTER_SIZE,
    chart_labels=FAMILIES_CHART_LABELS,
)


# -----------------------------
# Coaches profile (NEW)
# -----------------------------

COACHES_RATING_COLS = tuple([
    col("G"), col("H"), col("I"),   # first set of 1–5 rating questions
    col("K"), col("L"), col("M"), col("N"), col("O"), col("P"), col("Q"), col("R")  # second set
])

COACHES_YESNO_COLS = tuple([col("J")])

COACHES_CHART_LABELS: Dict[int, str] = {
    1: "(1) Support",
    2: "(2) Communication",
    3: "(3) Collaboration",
    4: "(4) Facility Condition",
    5: "(5) Scheduling",
    6: "(6) Applying Values",
    7: "(7) Seeing Values",
    8: "(8) Growth",
    9: "(9) Wellbeing",
    10: "(10) Overall Values",
    11: "(11) Overall Experience",
    12: "(12) On-time Payroll",
}

COACHES_PROFILE = SurveyProfile(
    key="coaches",
    title="Coaches Survey",
    respondent_singular="coach",
    respondent_plural="coaches",
    respondent_name_index=col("F"),     # Coach name column
    group_col_index=col("F"),           # Placeholder; overridden in pdf_report.py
    rating_col_indices=COACHES_RATING_COLS,
    yesno_col_indices=COACHES_YESNO_COLS,
    choice_col_index=None,
    qq_rating_col_index=col("R"),       # last rating = overall satisfaction (example)
    team_coach_map=None,
    denominator_map=None,
    chart_labels=COACHES_CHART_LABELS,
)


PROFILES: Dict[str, SurveyProfile] = {
    "players": PLAYERS_PROFILE,
    "families": FAMILIES_PROFILE,
    "coaches": COACHES_PROFILE,  # NEW
}
