# crop_care.py
#
# Simple per-crop care guide: for a chosen crop, shows the ACTUAL
# average N, P, K, and temperature that crop was grown under in the
# training dataset (real numbers, not invented — same CSV the ML model
# learned from), a fertilizer suggestion derived from those numbers,
# and a general step-by-step farming process.
#
# HONEST LIMITATIONS:
# - N/P/K/temperature are the dataset's per-crop averages. Real fields
#   vary — this is a reference point, not a soil test result.
# - Fertilizer guidance is a general N/P/K -> common-fertilizer mapping
#   (Urea, DAP, MOP), not a precise blended dose. A soil test from your
#   local Krishi Vigyan Kendra (KVK) will always be more accurate.
# - Farming process steps are general, widely-practiced steps — actual
#   practice varies by region, variety, and local conditions.
# - No pesticide/chemical dosage guidance is included (see project notes
#   on why that's scoped out for now).

import pandas as pd
import streamlit as st

DATASET_PATH = "dataset/Crop_recommendation.csv"


@st.cache_data
def get_crop_requirements() -> pd.DataFrame:
    """
    Real per-crop average N, P, K, temperature, humidity, pH, and
    rainfall, computed directly from the training dataset — not a
    separate/invented table.
    """
    df = pd.read_csv(DATASET_PATH)
    return df.groupby("label")[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].mean()


def _classify_level(value: float, all_values: pd.Series) -> str:
    """
    Classify a crop's N/P/K need as Low/Medium/High relative to the
    OTHER 21 crops in the dataset (using the 33rd/66th percentile of
    all crop-level averages as cutoffs) — not an arbitrary fixed
    number, so "High" actually means high compared to real alternatives.
    """
    low_cut = all_values.quantile(0.33)
    high_cut = all_values.quantile(0.66)

    if value <= low_cut:
        return "Low"
    elif value <= high_cut:
        return "Medium"
    else:
        return "High"


def get_fertilizer_guidance(n_level: str, p_level: str, k_level: str) -> list:
    """
    Translate N/P/K levels into commonly available Indian fertilizers.
    General mapping (Urea = N source, DAP/SSP = P source, MOP = K
    source), not a precise blended-dose calculation.
    """
    n_text = {
        "High": "High Nitrogen need — Urea or Ammonium Sulphate as the main N source, likely in split doses.",
        "Medium": "Moderate Nitrogen need — a standard Urea top-dressing schedule should be enough.",
        "Low": "Low Nitrogen need — go easy on Urea; too much can hurt yield/quality for this crop.",
    }
    p_text = {
        "High": "High Phosphorus need — DAP or Single Super Phosphate (SSP) as a generous basal dose.",
        "Medium": "Moderate Phosphorus need — a standard SSP/DAP basal dose is typical.",
        "Low": "Low Phosphorus need — a light basal P dose is usually sufficient.",
    }
    k_text = {
        "High": "High Potassium need — Muriate of Potash (MOP) as the main K source.",
        "Medium": "Moderate Potassium need — a standard MOP dose is typical.",
        "Low": "Low Potassium need — minimal extra K is usually required.",
    }

    return [n_text[n_level], p_text[p_level], k_text[k_level]]


# ----------------------------------------------------------------------
# General farming process per crop — widely-practiced standard steps.
# Kept intentionally concise; actual practice varies by region/variety.
# ----------------------------------------------------------------------

CROP_PROCESS = {
    "rice": [
        "Prepare a fine, leveled seedbed and puddle the main field.",
        "Raise nursery seedlings for 25-30 days, then transplant into the puddled, flooded field.",
        "Maintain shallow standing water through most of the growth cycle.",
        "Apply basal fertilizer before transplanting; top-dress at tillering and panicle initiation.",
        "Control weeds early (first 40 days); watch for stem borer and blast disease.",
        "Drain the field before harvest; harvest when grains turn golden-yellow.",
    ],
    "maize": [
        "Plough and level the land; sow seeds 2-3 cm deep in rows about 60-75 cm apart.",
        "Thin seedlings to proper spacing after emergence.",
        "Apply basal N-P-K at sowing; top-dress Nitrogen at knee-high and tasseling stages.",
        "Irrigate at critical stages: knee-high, tasseling, silking, and grain-filling.",
        "Control weeds in the first 30-40 days.",
        "Harvest when husks dry and kernels harden.",
    ],
    "chickpea": [
        "Sow in well-drained soil after monsoon (Rabi season), rows about 30 cm apart.",
        "Being a legume, it needs minimal Nitrogen; a light basal P and K dose helps.",
        "Irrigate sparingly — chickpea is fairly drought-tolerant; 1-2 irrigations if needed.",
        "Weed once around 20-25 days after sowing.",
        "Watch for pod borer.",
        "Harvest when pods dry and rattle.",
    ],
    "kidneybeans": [
        "Sow in rows 40-45 cm apart in well-drained soil.",
        "Apply a basal fertilizer dose; being a legume, Nitrogen need is lower than cereals.",
        "Irrigate at flowering and pod-development stages.",
        "Weed early in the season.",
        "Watch for bean beetle and aphids.",
        "Harvest when pods dry.",
    ],
    "pigeonpeas": [
        "Sow at the onset of monsoon with wide row spacing (60-75 cm) — it's a longer-duration crop.",
        "Legume — moderate basal P/K, minimal Nitrogen needed.",
        "Fairly drought-tolerant; irrigate during flowering/pod-filling only if there's a dry spell.",
        "Often intercropped with cereals.",
        "Watch for pod borer.",
        "Harvest in stages as pods mature.",
    ],
    "mothbeans": [
        "Sown in arid/semi-arid regions at monsoon onset.",
        "A minimal-input, drought-tolerant crop — light basal fertilizer is enough.",
        "Needs very little irrigation.",
        "Weed once early in the season.",
        "Harvest when pods dry.",
    ],
    "mungbean": [
        "Sow in rows about 30 cm apart; can be grown in both Kharif and summer season.",
        "Legume — light basal P/K, minimal Nitrogen.",
        "Irrigate at flowering and pod-filling if there's no rain.",
        "Weed around 20-25 days after sowing.",
        "Harvest in 2-3 pickings, since pods mature at different times.",
    ],
    "blackgram": [
        "Sow at monsoon onset, rows about 30 cm apart.",
        "Legume — a light basal fertilizer dose is usually enough.",
        "Moderate water need; avoid waterlogging.",
        "Weed once early.",
        "Harvest when pods turn black and dry.",
    ],
    "lentil": [
        "A Rabi (winter) crop — sow in well-prepared, well-drained soil.",
        "Legume — minimal Nitrogen, light basal P/K.",
        "1-2 light irrigations; avoid waterlogging.",
        "Weed once around 25-30 days.",
        "Harvest when plants turn yellowish-brown.",
    ],
    "pomegranate": [
        "Plant grafted saplings in pits enriched with farmyard manure, spacing about 4-5 m.",
        "Apply organic manure plus NPK in split doses through the year.",
        "Drip irrigation preferred — regular water, but avoid waterlogging.",
        "Prune to shape and remove suckers regularly.",
        "Watch for fruit borer and bacterial blight.",
        "Harvest when fruit develops full color and a slightly metallic sound when tapped.",
    ],
    "banana": [
        "Plant tissue-culture plants or suckers in manure-enriched pits, spacing about 1.8-2 m.",
        "A heavy feeder — apply NPK regularly in split doses.",
        "Needs frequent irrigation (drip preferred) — very water-sensitive.",
        "Remove excess suckers, keeping one follower per plant.",
        "Support plants against wind once bunches form.",
        "Harvest bunches when fingers fill out and turn light/yellow-green.",
    ],
    "mango": [
        "Plant grafted saplings, spacing about 8-10 m, in well-drained soil.",
        "Apply organic manure plus NPK annually, more once fruiting begins.",
        "Irrigate young trees regularly; reduce irrigation before flowering to help induce it.",
        "Prune dead or criss-crossing branches after harvest.",
        "Watch for hopper pest and anthracnose during flowering.",
        "Harvest once fruit reaches mature size and color change begins.",
    ],
    "grapes": [
        "Plant on a trellis/pandal system in well-drained soil, spacing per training system used.",
        "Prune heavily on a regular cycle (main pruning + fruit pruning) — critical for yield.",
        "Drip irrigation is standard; manage water carefully around flowering and fruit-set.",
        "Apply regular NPK plus micronutrients.",
        "Watch for downy mildew and powdery mildew.",
        "Harvest when berries reach full color and the desired sweetness.",
    ],
    "watermelon": [
        "Sow directly or transplant, with wide spacing (2-2.5 m between vines).",
        "Apply basal fertilizer at planting; top-dress during vine growth and flowering.",
        "Irrigate regularly, especially during fruit development; reduce near ripening for sweetness.",
        "Mulching helps retain moisture and suppress weeds.",
        "Watch for fruit fly and powdery mildew.",
        "Harvest when the ground spot turns yellow and the nearby tendril dries.",
    ],
    "muskmelon": [
        "Similar to watermelon — a warm-season vine crop needing wide spacing.",
        "Apply basal fertilizer, then top-dress during vine growth.",
        "Irrigate regularly; reduce watering as fruit nears ripening.",
        "Mulching is recommended.",
        "Watch for fruit fly.",
        "Harvest when the fruit develops full aroma/color and slips easily from the vine.",
    ],
    "apple": [
        "Needs a region with real winter chilling hours; plant grafted saplings at the spacing suited to the rootstock.",
        "Apply organic manure plus NPK annually.",
        "Moderate, regular irrigation, especially during fruit development.",
        "Prune annually during dormancy for shape and light penetration.",
        "Watch for scab and aphids.",
        "Harvest when fruit firmness and color match the variety's maturity signs.",
    ],
    "orange": [
        "Plant grafted saplings, spacing about 5-6 m, in well-drained soil.",
        "Apply organic manure plus NPK in split doses.",
        "Irrigate regularly; avoid waterlogging — citrus is sensitive to wet roots.",
        "Prune to remove dead wood and water shoots.",
        "Watch for citrus canker and leaf miner.",
        "Harvest once color and sweetness reach maturity — citrus fruit doesn't ripen further after picking.",
    ],
    "papaya": [
        "Plant seedlings in manure-enriched pits, spacing about 2 m.",
        "A fast grower — apply NPK regularly in split doses.",
        "Frequent, light irrigation; avoid waterlogging (root rot risk).",
        "Once flowering shows plant sex, remove most male plants, keeping a few for pollination.",
        "Watch for papaya ring spot virus.",
        "Harvest when the fruit shows a slight green-to-yellow color break.",
    ],
    "coconut": [
        "Plant seedlings in large, manure-enriched pits, spacing about 7-8 m.",
        "Apply regular manure/NPK, especially in the early years.",
        "Needs regular irrigation, especially in dry periods (young palms particularly).",
        "A basin or mulch around the base helps retain moisture.",
        "Watch for rhinoceros beetle and red palm weevil.",
        "Harvest mature nuts (~12 months after flowering for copra; earlier for tender coconut).",
    ],
    "cotton": [
        "Sow at monsoon onset, rows about 60-90 cm apart depending on variety.",
        "Apply basal fertilizer; top-dress Nitrogen at squaring and flowering stages.",
        "Moderate irrigation; avoid water stress during flowering and boll formation.",
        "Regular monitoring for bollworm is critical.",
        "Weed in the first 40-45 days.",
        "Harvest in multiple pickings as bolls open.",
    ],
    "jute": [
        "Sow at the start of monsoon, closely spaced, for tall, fiber-rich stems.",
        "Light basal fertilizer; top-dress Nitrogen once early in growth.",
        "Needs good rainfall/moisture; avoid waterlogging early, though it tolerates standing water later.",
        "Weed once early.",
        "Harvest at flowering stage for the best fiber quality, then ret (soak) the stems to extract fiber.",
    ],
    "coffee": [
        "Plant seedlings under shade trees, spacing about 2x2 m (Arabica) or wider (Robusta).",
        "Apply organic manure plus NPK in split doses through the year.",
        "Regular irrigation, especially before flowering ('blossom showers') and during dry spells.",
        "Prune to maintain shape and remove unproductive wood.",
        "Watch for berry borer and leaf rust.",
        "Harvest when cherries turn deep red — by selective picking, or strip picking at season's end.",
    ],
}


def get_crop_care(crop: str) -> dict:
    """
    Main entry point: real N/P/K/temperature averages for the crop,
    their relative High/Medium/Low levels, derived fertilizer guidance,
    and the general farming process.

    Raises RuntimeError if the crop isn't in the dataset.
    """
    requirements = get_crop_requirements()

    crop_key = crop.lower()
    if crop_key not in requirements.index:
        raise RuntimeError(f"'{crop}' isn't in the training dataset.")

    row = requirements.loc[crop_key]

    n_level = _classify_level(row["N"], requirements["N"])
    p_level = _classify_level(row["P"], requirements["P"])
    k_level = _classify_level(row["K"], requirements["K"])

    fertilizer_guidance = get_fertilizer_guidance(n_level, p_level, k_level)
    process = CROP_PROCESS.get(crop_key, [
        "Detailed farming process for this crop isn't written up yet — "
        "N/P/K/temperature and fertilizer guidance above are still real."
    ])

    return {
        "crop": crop_key,
        "N": float(row["N"]),
        "P": float(row["P"]),
        "K": float(row["K"]),
        "temperature": float(row["temperature"]),
        "n_level": n_level,
        "p_level": p_level,
        "k_level": k_level,
        "fertilizer_guidance": fertilizer_guidance,
        "process": process,
    }