"""Tests for transcript_cleaner_service.clean_transcript().

Pure Python, no API calls, no fixtures needed.
"""
from __future__ import annotations

from app.services.transcript_cleaner_service import clean_transcript


def test_removes_seconds_timestamp():
    raw = "0:099 seconds यह एक वाक्य है।"
    result = clean_transcript(raw)
    assert "seconds" not in result
    assert "0:09" not in result
    assert "यह एक वाक्य है।" in result


def test_removes_minutes_timestamp():
    raw = "4:004 minutes यह मामला बहुत गंभीर था।"
    result = clean_transcript(raw)
    assert "minutes" not in result
    assert "यह मामला बहुत गंभीर था।" in result


def test_removes_combined_minutes_seconds_timestamp():
    raw = "1:091 minute, 9 seconds Devika Rathi का घर"
    result = clean_transcript(raw)
    assert "minute" not in result
    assert "Devika Rathi का घर" in result


def test_removes_crime_beat_tv_season_footer():
    raw = "पुलिस ने गिरफ्तार किया।\nCrime Beat TV - Season 3\nयह सच्ची घटना है।"
    result = clean_transcript(raw)
    assert "Crime Beat TV" not in result
    assert "पुलिस ने गिरफ्तार किया।" in result
    assert "यह सच्ची घटना है।" in result


def test_removes_crime_beat_tv_standalone():
    raw = "Crime Beat TV सुनवाई शुरू हुई।"
    result = clean_transcript(raw)
    assert "Crime Beat TV" not in result
    assert "सुनवाई शुरू हुई।" in result


def test_removes_sync_to_video_time():
    raw = "Sync to video time न्यायालय ने निर्णय दिया।"
    result = clean_transcript(raw)
    assert "Sync to video time" not in result
    assert "न्यायालय ने निर्णय दिया।" in result


def test_removes_playlist_position():
    raw = "1 / 13 Prakash Soni को दोषी पाया गया।"
    result = clean_transcript(raw)
    assert "1 / 13" not in result
    assert "Prakash Soni को दोषी पाया गया।" in result


def test_removes_standalone_season_line():
    raw = "यह घटना नागपुर में हुई।\nSeason 4\nपीड़िता 28 वर्ष की थी।"
    result = clean_transcript(raw)
    assert "Season 4" not in result
    assert "यह घटना नागपुर में हुई।" in result
    assert "पीड़िता 28 वर्ष की थी।" in result


def test_preserves_hindi_names_and_dates():
    raw = "2021-06-10 को Devika Rathi का शव मिला। Justice Arvind Nair ने फैसला सुनाया।"
    result = clean_transcript(raw)
    assert "Devika Rathi" in result
    assert "Justice Arvind Nair" in result
    assert "2021-06-10" in result


def test_collapses_excessive_blank_lines():
    raw = "पहली पंक्ति।\n\n\n\nदूसरी पंक्ति।"
    result = clean_transcript(raw)
    assert "\n\n\n" not in result
    assert "पहली पंक्ति।" in result
    assert "दूसरी पंक्ति।" in result


def test_clean_passthrough_returns_stripped():
    raw = "  यह पहले से साफ़ है।  "
    result = clean_transcript(raw)
    assert result == "यह पहले से साफ़ है।"


def test_empty_string_returns_empty():
    assert clean_transcript("") == ""
