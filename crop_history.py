# crop_history.py
#
# Looks up REAL historical crop production data for a district + season
# (Kharif/Rabi) from a downloaded government dataset, and ranks crops by
# how much of that district's recorded production/area they actually
# account for.
#
# This does NOT use the trained ML model (best_model.pkl / scaler.pkl /
# label_encoder.pkl) at all — it's a statistics lookup against real
# government records, not a soil/climate prediction. Deliberately kept
# in its own module so that distinction stays obvious in the code, not
# just in a comment somewhere in app.py.
#
# ----------------------------------------------------------------------
# SETUP REQUIRED BEFORE THIS WILL WORK:
#
# 1. Download the dataset as a CSV. Two options:
#
#    a) Kaggle (easier — direct download button, free account):
#       https://www.kaggle.com/datasets/abhinand05/crop-production-in-india
#       This is a mirror of the same government data (originally from
#       data.gov.in's "District-wise, season-wise crop production
#       statistics", 1997-2015), republished with columns:
#       State_Name, District_Name, Crop_Year, Season, Crop, Area, Production.
#
#    b) data.gov.in directly (official source, but slower/less reliable
#       to download):
#       https://www.data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997
#
# 2. Save it as: dataset/district_season_crop_production.csv
#    (same folder as your existing Crop_recommendation.csv)
#
# No API key needed for this version — it reads the file straight off
# disk, which also sidesteps data.gov.in's live API being slow/timing
# out, and means the exact column names in your file are all that
# matters (this code inspects them at runtime rather than assuming).
# ----------------------------------------------------------------------

import pandas as pd
import streamlit as st

LOCAL_DATASET_PATH = "dataset/district_season_crop_production.csv"


def _find_column(columns, *substrings: str, exclude: tuple = ()):
    """
    Find a column name that contains any of the given substrings
    (case-insensitive), skipping any column that contains an excluded
    substring. Government CSVs don't use fully consistent naming, so
    we match loosely instead of hardcoding one exact spelling.
    """
    for col in columns:
        col_lower = str(col).lower()
        if any(ex in col_lower for ex in exclude):
            continue
        if any(s in col_lower for s in substrings):
            return col
    return None


@st.cache_data
def load_local_dataset(path: str = LOCAL_DATASET_PATH) -> pd.DataFrame:
    """
    Load the downloaded crop production CSV once per session. Raises
    FileNotFoundError with a clear message if the file hasn't been
    downloaded and placed yet.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Couldn't find '{path}'. Download the dataset from "
            f"data.gov.in and save it at this exact path — see the "
            f"setup instructions at the top of crop_history.py."
        )


def get_districts_for_state(state: str) -> list:
    """
    Return the exact district names as spelled in the real government
    dataset, for the given state — so the UI picker always matches
    something real instead of relying on the user guessing the exact
    government spelling.

    Returns an empty list if the dataset file isn't downloaded yet, or
    the state column/district column can't be identified — callers
    should treat an empty list as "fall back to manual text entry",
    not as an error.
    """
    try:
        df = load_local_dataset()
    except FileNotFoundError:
        return []

    columns = df.columns.tolist()
    state_col = _find_column(columns, "state")
    district_col = _find_column(columns, "district")

    if district_col is None:
        return []

    if state_col:
        mask = df[state_col].astype(str).str.strip().str.lower() == state.strip().lower()
        subset = df[mask]
    else:
        subset = df

    if subset.empty:
        return []

    districts = subset[district_col].astype(str).str.strip()
    return sorted(districts.unique().tolist())


def get_historical_crops(state: str, district: str, season: str) -> list:
    """
    Filter the local dataset to a district + season, and rank crops by
    total recorded production (or area, if production isn't a column)
    summed across every year present in the file.

    Returns: list of dicts, sorted by total value descending:
        {"crop": str, "total": float, "unit": str, "years": int}

    Raises RuntimeError for any failure — file missing, no matching
    rows, or a shape we don't recognise — with enough detail to debug it.
    """
    try:
        df = load_local_dataset()
    except FileNotFoundError as e:
        raise RuntimeError(str(e))

    columns = df.columns.tolist()

    state_col = _find_column(columns, "state")
    district_col = _find_column(columns, "district")
    season_col = _find_column(columns, "season")
    crop_col = _find_column(columns, "crop", exclude=("year",))
    production_col = _find_column(columns, "production")
    area_col = _find_column(columns, "area")
    year_col = _find_column(columns, "year")

    if district_col is None or crop_col is None:
        raise RuntimeError(
            f"Couldn't find expected 'district' and 'crop' columns. "
            f"Columns found in the file: {columns}"
        )

    value_col = production_col or area_col
    if value_col is None:
        raise RuntimeError(
            f"Couldn't find a production/area column. "
            f"Columns found in the file: {columns}"
        )
    unit = "tonnes" if production_col else "hectares"

    # ------------------------------------------------------------
    # Filter to district (+ state if that column exists) + season
    # ------------------------------------------------------------
    mask = df[district_col].astype(str).str.strip().str.lower() == district.strip().lower()

    if state_col:
        mask &= df[state_col].astype(str).str.strip().str.lower() == state.strip().lower()

    if season_col:
        mask &= df[season_col].astype(str).str.strip().str.lower().str.contains(season.lower())

    filtered = df[mask]

    if filtered.empty:
        raise RuntimeError(
            f"No rows found for district '{district}'"
            + (f" in state '{state}'" if state_col else "")
            + (f" and season '{season}'" if season_col else "")
            + ". The district name may not exactly match the government "
            "dataset's spelling (e.g. 'Bardhaman' vs 'Barddhaman')."
        )

    # ------------------------------------------------------------
    # Sum by crop
    # ------------------------------------------------------------
    filtered = filtered.copy()
    filtered[value_col] = pd.to_numeric(filtered[value_col], errors="coerce")
    filtered = filtered.dropna(subset=[value_col])

    if filtered.empty:
        raise RuntimeError(
            f"Found rows for {district} but the '{value_col}' column had "
            f"no valid numeric values in them."
        )

    grouped = filtered.groupby(crop_col)[value_col].sum()

    years_by_crop = {}
    if year_col:
        years_by_crop = filtered.groupby(crop_col)[year_col].nunique().to_dict()

    ranked = grouped.sort_values(ascending=False)

    return [
        {
            "crop": str(crop),
            "total": float(total),
            "unit": unit,
            "years": int(years_by_crop.get(crop, 1)),
        }
        for crop, total in ranked.items()
    ]
