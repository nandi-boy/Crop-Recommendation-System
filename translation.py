# translation.py
#
# Translates Crop Care Guide text into major Indian languages, using
# the free (no API key) Google Translate wrapper via deep-translator.
#
# HONEST LIMITATIONS:
#
# - This calls Google Translate's public web frontend unofficially
#   (via the deep-translator library) — not an Anthropic or government
#   API. It needs internet access at runtime, and could break if
#   Google changes that frontend (deep-translator is actively
#   maintained to track such changes, but there's no guarantee).
#
# - Machine translation of farming instructions can occasionally be
#   imprecise, especially for technical terms. If something looks off
#   in your language, cross-check against the English version.
#
# - Scoped to the 12 most widely-spoken Indian languages by number of
#   speakers, not all 22 scheduled languages — smaller-language
#   translation quality through this method is much less reliable.
#   More languages can be added to LANGUAGES below if needed.
#
# - Untested against the live Google Translate service from the
#   environment this was built in (no network access to it there) —
#   confirmed to import and run correctly, but the actual translated
#   output needs a real run to verify.

from deep_translator import GoogleTranslator
import streamlit as st

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Odia": "or",
    "Urdu": "ur",
    "Assamese": "as",
}


@st.cache_data(show_spinner=False)
def translate_text(text: str, lang_code: str) -> str:
    """
    Translate one piece of text into the target language. Returns the
    original text unchanged if the target is English, the text is
    empty, or translation fails for any reason — so the app degrades
    to English instead of crashing.
    """
    if lang_code == "en" or not text:
        return text

    try:
        return GoogleTranslator(source="en", target=lang_code).translate(text)
    except Exception:
        return text


def translate_list(items: list, lang_code: str) -> list:
    """Translate each string in a list, same fallback behavior as translate_text."""
    return [translate_text(item, lang_code) for item in items]
