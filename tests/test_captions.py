"""Tests de core/captions.py (Fase 5) - segmentacion y captions puras.

Cubre:
- normalizacion de texto
- CaptionSegment modelo (timestamps, words, validacion)
- captions_from_segment (word-level y segment-level)
- captions_for_window (offset, clamp, determinismo)
- _chunk_words (limite de palabras)
- esc_ass (texto hostil)
- wrap_words (lineas)
- build_ass (determinismo, estructura ASS)
- Font resolver
- SubtitleRenderResult modelo
"""

from __future__ import annotations

import math
import pytest

import config
from core.captions import (
    captions_for_window,
    captions_from_segment,
    normalize_text,
    words_from_segment,
    _chunk_words,
)
from core.subtitles import (
    build_ass,
    esc_ass,
    resolve_font_family,
    wrap_words,
    _ass_time,
    _bgr,
)
from core.errors import SubtitleDataError, SubtitleFontError
from models.schemas import CaptionSegment, SubtitleRenderResult, TranscriptSegment


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_collapse_whitespace(self):
        assert normalize_text("  hola   mundo  ") == "hola mundo"

    def test_newlines_to_space(self):
        assert normalize_text("hola\nmundo\n\nsegunda") == "hola mundo segunda"

    def test_tabs_to_space(self):
        assert normalize_text("hola\tmundo") == "hola mundo"

    def test_cr_lf_to_space(self):
        assert normalize_text("hola\r\nmundo") == "hola mundo"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_none_returns_empty(self):
        assert normalize_text(None) == ""

    def test_only_whitespace(self):
        assert normalize_text("   \t\n\r  ") == ""

    def test_unicode_preserved(self):
        s = "hola " + chr(252) + "ndo"
        assert normalize_text(s) == s

    def test_control_chars_collapsed(self):
        assert normalize_text("hola\x00mundo") == "hola mundo"

    def test_nul_char(self):
        assert normalize_text("a\x00b") == "a b"


# ---------------------------------------------------------------------------
# CaptionSegment modelo
# ---------------------------------------------------------------------------

class TestCaptionSegmentModel:
    def test_valid_minimal(self):
        c = CaptionSegment(text="hola", start=0.0, end=1.0)
        assert c.text == "hola"
        assert c.start == 0.0
        assert c.end == 1.0
        assert c.highlight is True
        assert c.style == "dynamic"

    def test_valid_with_words(self):
        c = CaptionSegment(
            text="hola mundo",
            start=0.0,
            end=2.0,
            words=[
                {"start": 0.0, "end": 1.0, "word": "hola"},
                {"start": 1.0, "end": 2.0, "word": "mundo"},
            ],
        )
        assert len(c.words) == 2

    def test_rejects_empty_text(self):
        with pytest.raises(Exception):
            CaptionSegment(text="", start=0.0, end=1.0)

    def test_rejects_end_le_start(self):
        with pytest.raises(Exception):
            CaptionSegment(text="x", start=1.0, end=1.0)
        with pytest.raises(Exception):
            CaptionSegment(text="x", start=2.0, end=1.0)

    def test_rejects_nan(self):
        with pytest.raises(Exception):
            CaptionSegment(text="x", start=float("nan"), end=1.0)
        with pytest.raises(Exception):
            CaptionSegment(text="x", start=0.0, end=float("nan"))

    def test_rejects_inf(self):
        with pytest.raises(Exception):
            CaptionSegment(text="x", start=0.0, end=float("inf"))
        with pytest.raises(Exception):
            CaptionSegment(text="x", start=float("-inf"), end=1.0)

    def test_rejects_negative_start(self):
        with pytest.raises(Exception):
            CaptionSegment(text="x", start=-0.5, end=1.0)

    def test_rejects_word_out_of_range(self):
        with pytest.raises(Exception):
            CaptionSegment(
                text="x",
                start=0.0,
                end=1.0,
                words=[{"start": 0.0, "end": 2.0, "word": "x"}],
            )

    def test_rejects_word_end_before_start(self):
        with pytest.raises(Exception):
            CaptionSegment(
                text="x",
                start=0.0,
                end=1.0,
                words=[{"start": 0.5, "end": 0.3, "word": "x"}],
            )

    def test_rejects_word_missing_keys(self):
        with pytest.raises(Exception):
            CaptionSegment(
                text="x",
                start=0.0,
                end=1.0,
                words=[{"word": "x"}],
            )

    def test_rejects_word_empty_text(self):
        with pytest.raises(Exception):
            CaptionSegment(
                text="x",
                start=0.0,
                end=1.0,
                words=[{"start": 0.0, "end": 0.5, "word": ""}],
            )

    def test_rejects_word_nan(self):
        with pytest.raises(Exception):
            CaptionSegment(
                text="x",
                start=0.0,
                end=1.0,
                words=[{"start": 0.0, "end": float("nan"), "word": "x"}],
            )


# ---------------------------------------------------------------------------
# SubtitleRenderResult modelo
# ---------------------------------------------------------------------------

class TestSubtitleRenderResultModel:
    def test_valid(self):
        r = SubtitleRenderResult(
            clip_id="clip_000",
            source_clip="v.mp4",
            output_clip="c.mp4",
            duration=10.0,
            caption_count=5,
        )
        assert r.qc_status == "pending"
        assert r.subtitle_source == "transcript"

    def test_rejects_invalid_subtitle_source(self):
        with pytest.raises(Exception):
            SubtitleRenderResult(
                clip_id="c",
                source_clip="s",
                output_clip="o",
                duration=1.0,
                caption_count=0,
                subtitle_source="invalid",
            )

    def test_rejects_zero_duration(self):
        with pytest.raises(Exception):
            SubtitleRenderResult(
                clip_id="c",
                source_clip="s",
                output_clip="o",
                duration=0.0,
                caption_count=0,
            )


# ---------------------------------------------------------------------------
# words_from_segment
# ---------------------------------------------------------------------------

class TestWordsFromSegment:
    def test_valid_words(self):
        seg = TranscriptSegment(
            start=0.0, end=2.0, text="hola mundo",
            words=[
                {"start": 0.0, "end": 1.0, "word": "hola"},
                {"start": 1.0, "end": 2.0, "word": "mundo"},
            ],
        )
        words = words_from_segment(seg)
        assert words is not None
        assert len(words) == 2
        assert words[0]["word"] == "hola"

    def test_no_words_returns_none(self):
        seg = TranscriptSegment(start=0.0, end=2.0, text="hola")
        assert words_from_segment(seg) is None

    def test_empty_words_returns_none(self):
        seg = TranscriptSegment(start=0.0, end=2.0, text="hola", words=[])
        assert words_from_segment(seg) is None

    def test_rejects_word_no_start(self):
        seg = TranscriptSegment(
            start=0.0, end=2.0, text="x",
            words=[{"word": "x", "end": 1.0}],
        )
        with pytest.raises(SubtitleDataError):
            words_from_segment(seg)

    def test_rejects_word_nan_start(self):
        seg = TranscriptSegment(
            start=0.0, end=2.0, text="x",
            words=[{"start": float("nan"), "end": 1.0, "word": "x"}],
        )
        with pytest.raises(SubtitleDataError):
            words_from_segment(seg)

    def test_rejects_word_end_before_start(self):
        seg = TranscriptSegment(
            start=0.0, end=2.0, text="x",
            words=[{"start": 1.0, "end": 0.5, "word": "x"}],
        )
        with pytest.raises(SubtitleDataError):
            words_from_segment(seg)


# ---------------------------------------------------------------------------
# _chunk_words
# ---------------------------------------------------------------------------

class TestChunkWords:
    def test_chunks_by_max(self):
        words = [{"word": f"w{i}", "start": float(i), "end": float(i + 1)}
                 for i in range(8)]
        chunks = _chunk_words(words, max_words=3)
        assert len(chunks) == 3
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 2

    def test_merges_trailing_single(self):
        words = [{"word": f"w{i}", "start": float(i), "end": float(i + 1)}
                 for i in range(5)]
        chunks = _chunk_words(words, max_words=3)
        assert len(chunks) == 2
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 2

    def test_single_word(self):
        words = [{"word": "w0", "start": 0.0, "end": 1.0}]
        chunks = _chunk_words(words, max_words=6)
        assert len(chunks) == 1
        assert len(chunks[0]) == 1

    def test_rejects_max_words_lt_1(self):
        with pytest.raises(SubtitleDataError):
            _chunk_words([], max_words=0)


# ---------------------------------------------------------------------------
# captions_from_segment
# ---------------------------------------------------------------------------

class TestCaptionsFromSegment:
    def test_word_level(self):
        seg = TranscriptSegment(
            start=0.0, end=4.0, text="hola mundo cruel",
            words=[
                {"start": 0.0, "end": 1.0, "word": "hola"},
                {"start": 1.0, "end": 2.0, "word": "mundo"},
                {"start": 2.0, "end": 4.0, "word": "cruel"},
            ],
        )
        caps = captions_from_segment(seg, max_words=6)
        assert len(caps) == 1
        assert caps[0].text == "hola mundo cruel"
        assert len(caps[0].words) == 3

    def test_word_level_chunked(self):
        seg = TranscriptSegment(
            start=0.0, end=8.0, text="a b c d e f",
            words=[
                {"start": 0.0, "end": 1.0, "word": "a"},
                {"start": 1.0, "end": 2.0, "word": "b"},
                {"start": 2.0, "end": 3.0, "word": "c"},
                {"start": 3.0, "end": 4.0, "word": "d"},
                {"start": 4.0, "end": 5.0, "word": "e"},
                {"start": 5.0, "end": 8.0, "word": "f"},
            ],
        )
        caps = captions_from_segment(seg, max_words=3)
        assert len(caps) == 2
        assert caps[0].text == "a b c"
        assert caps[1].text == "d e f"

    def test_segment_level_no_words(self):
        seg = TranscriptSegment(start=0.0, end=3.0, text="hola mundo cruel")
        caps = captions_from_segment(seg, max_words=6)
        assert len(caps) == 1
        assert caps[0].text == "hola mundo cruel"
        assert caps[0].words == []

    def test_empty_text_no_words(self):
        seg = TranscriptSegment(start=0.0, end=3.0, text="  ")
        caps = captions_from_segment(seg, max_words=6)
        assert caps == []

    def test_rejects_degenerate_segment(self):
        seg = TranscriptSegment(start=2.0, end=1.0, text="x")
        with pytest.raises(SubtitleDataError):
            captions_from_segment(seg, max_words=6)


# ---------------------------------------------------------------------------
# captions_for_window (offset + determinism)
# ---------------------------------------------------------------------------

class TestCaptionsForWindow:
    def test_offset_shifts_to_clip_timeline(self):
        segs = [TranscriptSegment(
            start=75.0, end=80.0, text="hello world",
            words=[
                {"start": 75.0, "end": 76.0, "word": "hello"},
                {"start": 77.0, "end": 80.0, "word": "world"},
            ],
        )]
        caps = captions_for_window(segs, 60.0, 100.0)
        assert len(caps) == 1
        assert caps[0].start == pytest.approx(15.0)
        assert caps[0].end == pytest.approx(20.0)

    def test_segments_outside_window_excluded(self):
        segs = [
            TranscriptSegment(start=0.0, end=5.0, text="before"),
            TranscriptSegment(start=60.0, end=65.0, text="inside"),
            TranscriptSegment(start=100.0, end=105.0, text="after"),
        ]
        caps = captions_for_window(segs, 50.0, 80.0)
        assert len(caps) == 1
        assert caps[0].text == "inside"

    def test_clamp_to_window(self):
        segs = [TranscriptSegment(start=55.0, end=85.0, text="straddling")]
        caps = captions_for_window(segs, 60.0, 80.0)
        assert len(caps) == 1
        assert caps[0].start == 0.0
        assert caps[0].end == pytest.approx(20.0)

    def test_deterministic(self):
        segs = [
            TranscriptSegment(start=10.0, end=12.0, text="a",
                words=[{"start": 10.0, "end": 11.0, "word": "a"}]),
            TranscriptSegment(start=13.0, end=15.0, text="b",
                words=[{"start": 13.0, "end": 15.0, "word": "b"}]),
        ]
        r1 = captions_for_window(segs, 5.0, 20.0)
        r2 = captions_for_window(segs, 5.0, 20.0)
        assert [c.text for c in r1] == [c.text for c in r2]
        assert [c.start for c in r1] == [c.start for c in r2]

    def test_rejects_degenerate_window(self):
        with pytest.raises(SubtitleDataError):
            captions_for_window([], 10.0, 5.0)

    def test_rejects_nan_window(self):
        with pytest.raises(SubtitleDataError):
            captions_for_window([], float("nan"), 10.0)


# ---------------------------------------------------------------------------
# esc_ass (texto hostil)
# ---------------------------------------------------------------------------

class TestEscAss:
    def test_curly_braces(self):
        assert esc_ass("{test}") == "\\{test\\}"
        assert esc_ass("}") == "\\}"
        assert esc_ass("{") == "\\{"

    def test_backslash(self):
        assert esc_ass("\\N") == "\\\\N"
        assert esc_ass("\\n") == "\\\\n"
        assert esc_ass("\\\\") == "\\\\\\\\"

    def test_shell_injection(self):
        assert esc_ass("; rm -rf /") == "; rm -rf /"
        assert esc_ass("$(whoami)") == "$(whoami)"

    def test_shell_backticks(self):
        assert esc_ass("`whoami`") == "`whoami`"

    def test_percent_format(self):
        assert esc_ass("%s") == "%s"

    def test_colon(self):
        assert esc_ass(":") == ":"

    def test_quotes(self):
        assert esc_ass("'") == "'"
        assert esc_ass('"') == '"'

    def test_html_entities(self):
        assert esc_ass("&amp;") == "&amp;"
        assert esc_ass("|") == "|"

    def test_brackets(self):
        assert esc_ass("[test]") == "[test]"

    def test_unicode(self):
        s = "hola " + chr(233) + "l" + chr(232)
        result = esc_ass(s)
        assert chr(233) in result

    def test_newlines_collapsed(self):
        result = esc_ass("line1\nline2")
        assert "\\N" not in result
        assert "\n" not in result

    def test_deterministic(self):
        s = "{test} \\\\ special"
        assert esc_ass(s) == esc_ass(s)

    def test_empty(self):
        assert esc_ass("") == ""

    def test_crlf(self):
        result = esc_ass("a\r\nb")
        assert "\r" not in result
        assert "\n" not in result


# ---------------------------------------------------------------------------
# wrap_words
# ---------------------------------------------------------------------------

class TestWrapWords:
    def test_single_line(self):
        words = ["hello", "world"]
        lines = wrap_words(words, max_chars=20)
        assert len(lines) == 1
        assert lines[0] == ["hello", "world"]

    def test_wraps_to_two_lines(self):
        words = ["this", "is", "a", "very", "long", "caption"]
        lines = wrap_words(words, max_chars=12)
        assert len(lines) == 2

    def test_long_word_cut(self):
        words = ["supercalifragilisticexpialidocious"]
        lines = wrap_words(words, max_chars=5)
        assert len(lines) == 2
        assert lines[0][0] == "super"

    def test_max_lines_enforced(self):
        words = ["a", "b", "c", "d", "e", "f"]
        lines = wrap_words(words, max_chars=1, max_lines=1)
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# _ass_time
# ---------------------------------------------------------------------------

class TestAssTime:
    def test_zero(self):
        assert _ass_time(0.0) == "0:00:00.01"

    def test_seconds(self):
        assert _ass_time(12.345) == "0:00:12.34"

    def test_minutes(self):
        assert _ass_time(65.0) == "0:01:05.00"

    def test_hours(self):
        assert _ass_time(3661.5) == "1:01:01.50"


# ---------------------------------------------------------------------------
# _bgr
# ---------------------------------------------------------------------------

class TestBgr:
    def test_white(self):
        assert _bgr("FFFFFF") == "&H00FFFFFF"

    def test_orange(self):
        result = _bgr("FF6600")
        assert result == "&H000066FF"

    def test_invalid_length(self):
        with pytest.raises(SubtitleDataError):
            _bgr("FFF")

    def test_invalid_hex(self):
        with pytest.raises(SubtitleDataError):
            _bgr("ZZZZZZ")


# ---------------------------------------------------------------------------
# build_ass (determinismo + estructura)
# ---------------------------------------------------------------------------

class TestBuildAss:
    def _simple_caps(self):
        return [
            CaptionSegment(
                text="hello world",
                start=0.0,
                end=2.0,
                words=[
                    {"start": 0.0, "end": 1.0, "word": "hello"},
                    {"start": 1.0, "end": 2.0, "word": "world"},
                ],
            ),
        ]

    def test_structure_has_all_sections(self):
        ass = build_ass(self._simple_caps(), 506, 900, font="Arial")
        assert "[Script Info]" in ass
        assert "[V4+ Styles]" in ass
        assert "[Events]" in ass
        assert "Dialogue:" in ass

    def test_playres_matches_dims(self):
        ass = build_ass(self._simple_caps(), 506, 900, font="Arial")
        assert "PlayResX: 506" in ass
        assert "PlayResY: 900" in ass

    def test_playres_1080x1920(self):
        ass = build_ass(self._simple_caps(), 1080, 1920, font="Arial")
        assert "PlayResX: 1080" in ass
        assert "PlayResY: 1920" in ass

    def test_deterministic(self):
        caps = self._simple_caps()
        a1 = build_ass(caps, 506, 900, font="Arial")
        a2 = build_ass(caps, 506, 900, font="Arial")
        assert a1 == a2

    def test_empty_captions(self):
        ass = build_ass([], 506, 900, font="Arial")
        assert "Dialogue:" not in ass
        assert "[Events]" in ass

    def test_highlight_dialogues(self):
        caps = self._simple_caps()
        ass = build_ass(caps, 506, 900, font="Arial", highlight_enabled=True)
        dialogue_count = ass.count("Dialogue:")
        assert dialogue_count >= 3  # base + 2 words

    def test_no_highlight(self):
        caps = self._simple_caps()
        ass = build_ass(caps, 506, 900, font="Arial", highlight_enabled=False)
        assert ass.count("Dialogue:") == 1

    def test_rejects_zero_dims(self):
        with pytest.raises(SubtitleDataError):
            build_ass(self._simple_caps(), 0, 0, font="Arial")

    def test_ass_file_valid_utf8(self):
        ass = build_ass(self._simple_caps(), 506, 900, font="Arial")
        encoded = ass.encode("utf-8")
        decoded = encoded.decode("utf-8")
        assert decoded == ass

    def test_scale_font_by_width(self):
        ass64 = build_ass(self._simple_caps(), 506, 900, font="Arial", font_size=64)
        ass128 = build_ass(self._simple_caps(), 1080, 1920, font="Arial", font_size=64)
        fs1 = int(ass64.split("Style: Default,")[1].split(",")[1])
        fs2 = int(ass128.split("Style: Default,")[1].split(",")[1])
        assert fs1 < fs2


# ---------------------------------------------------------------------------
# resolve_font_family
# ---------------------------------------------------------------------------

class TestResolveFontFamily:
    def test_auto_detects(self):
        font = resolve_font_family()
        assert isinstance(font, str)
        assert len(font) > 0

    def test_explicit_valid(self):
        font = resolve_font_family("Arial")
        assert font == "Arial"

    def test_explicit_invalid_raises(self):
        with pytest.raises(SubtitleFontError):
            resolve_font_family("NonExistentFont123")

    def test_empty_auto(self):
        font = resolve_font_family("")
        assert len(font) > 0

    def test_custom_fonts_dir(self, tmp_path):
        # Create a fake font file
        fake = tmp_path / "myfont.ttf"
        fake.write_bytes(b"fake")
        font = resolve_font_family("Myfont", fonts_dir=tmp_path)
        assert font == "Myfont"

    def test_custom_fonts_dir_not_found(self, tmp_path):
        with pytest.raises(SubtitleFontError):
            resolve_font_family("NoFont", fonts_dir=tmp_path)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(SubtitleFontError):
            resolve_font_family("", fonts_dir=tmp_path)


# ---------------------------------------------------------------------------
# Multi-line caption
# ---------------------------------------------------------------------------

class TestMultiLineCaption:
    def test_two_lines(self):
        seg = TranscriptSegment(
            start=0.0, end=4.0, text="a b c d e f g h",
            words=[
                {"start": 0.0, "end": 0.5, "word": f"word{i}"}
                for i, w in enumerate(["a", "b", "c", "d", "e", "f", "g", "h"])
            ],
        )
        seg = TranscriptSegment(
            start=0.0, end=4.0, text="a b c d e f g h",
            words=[{"start": float(i) * 0.5, "end": float(i) * 0.5 + 0.4, "word": w}
                   for i, w in enumerate(["a", "b", "c", "d", "e", "f", "g", "h"])],
        )
        caps = captions_from_segment(seg, max_words=8)
        assert len(caps) == 1
        ass = build_ass(caps, 506, 900, font="Arial", max_chars=12)
        assert "\\N" in ass
