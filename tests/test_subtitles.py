"""Tests E2E de core/subtitles.py (Fase 5) - render real con FFmpeg.

Cubre:
- render_captioned: FFmpeg real + ffprobe
- render_captioned_clips: manifest
- resolve_font_family: fuente real del sistema
- E2E completo: vertical -> captioned -> qc
- E2E con texto hostil
- Preservacion de resolucion/audio/duracion
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
import core.clipper as clipper
import core.vertical as vertical
from core.captions import captions_for_window
from core.errors import (
    ClipError,
    ClipFilesystemError,
    SubtitleFontError,
    SubtitleRenderError,
)
from core.subtitles import (
    build_ass,
    render_captioned,
    render_captioned_clips,
    resolve_font_family,
    write_captioned_manifest,
)
from models.schemas import (
    CaptionSegment,
    HookCandidate,
    SubtitleRenderResult,
    TranscriptSegment,
)
from tests.fixtures import video_synthetic


def _hook(start: float = 2.0, end: float = 10.0, score: float = 0.8) -> HookCandidate:
    return HookCandidate(start=start, end=end, score=score, title="Hook", reason="test")


def _transcript_with_words():
    return [
        TranscriptSegment(
            start=0.0, end=3.0, text="hello world",
            words=[
                {"start": 0.0, "end": 1.5, "word": "hello"},
                {"start": 1.5, "end": 3.0, "word": "world"},
            ],
        ),
        TranscriptSegment(
            start=3.0, end=6.0, text="this is a test",
            words=[
                {"start": 3.0, "end": 3.5, "word": "this"},
                {"start": 3.5, "end": 4.0, "word": "is"},
                {"start": 4.0, "end": 4.5, "word": "a"},
                {"start": 4.5, "end": 6.0, "word": "test"},
            ],
        ),
    ]


def _transcript_hostile():
    return [
        TranscriptSegment(
            start=0.0, end=3.0, text="{test} special chars",
            words=[
                {"start": 0.0, "end": 1.0, "word": "{test}"},
                {"start": 1.0, "end": 3.0, "word": "special chars"},
            ],
        ),
        TranscriptSegment(
            start=3.0, end=6.0, text="; rm -rf /",
            words=[
                {"start": 3.0, "end": 6.0, "word": "; rm -rf /"},
            ],
        ),
    ]


def _build_vertical_job(src, job_id):
    """Build horizontal clips + vertical from fixture."""
    entries = clipper.clip_hooks(
        str(src), "direct_file", [_hook(2.0, 10.0)], job_id,
        raw_path=src,
        media_duration=clipper.probe_duration(str(src)),
    )
    assert len(entries) == 1
    v = vertical.render_vertical(entries[0], job_id)
    vertical.write_vertical_manifest(job_id, [v])
    return entries[0], v


# ---------------------------------------------------------------------------
# resolve_font_family (real system)
# ---------------------------------------------------------------------------

class TestResolveFontReal:
    def test_resolves_real_font(self):
        font = resolve_font_family()
        assert len(font) > 0

    def test_explicit_invalid_raises(self):
        with pytest.raises(SubtitleFontError):
            resolve_font_family("NonexistentFontXYZ123")


# ---------------------------------------------------------------------------
# render_captioned (real FFmpeg)
# ---------------------------------------------------------------------------

class TestRenderCaptionedReal:
    def test_captioned_resolution_preserved(self):
        src = video_synthetic()
        assert src.exists()
        horiz, vert = _build_vertical_job(src, "cap_res")
        expected_w, expected_h = vert.output_width, vert.output_height
        assert (expected_w, expected_h) == (506, 900)

        segs = _transcript_with_words()
        caps = captions_for_window(segs, horiz.source_start, horiz.source_end)

        result = render_captioned(vert, caps, "cap_res")

        assert result.qc_status == "passed"
        assert result.caption_count == len(caps)
        out = config.output_dir() / result.output_clip
        assert out.exists()
        assert out.stat().st_size > 0

        w, h = clipper.probe_dimensions(out)
        assert (w, h) == (expected_w, expected_h)

    def test_audio_preserved(self):
        src = video_synthetic()
        horiz, vert = _build_vertical_job(src, "cap_audio")
        segs = _transcript_with_words()
        caps = captions_for_window(segs, horiz.source_start, horiz.source_end)
        result = render_captioned(vert, caps, "cap_audio")

        out = config.output_dir() / result.output_clip
        assert clipper._probe_has_audio(out)

    def test_duration_preserved(self):
        src = video_synthetic()
        horiz, vert = _build_vertical_job(src, "cap_dur")
        segs = _transcript_with_words()
        caps = captions_for_window(segs, horiz.source_start, horiz.source_end)
        result = render_captioned(vert, caps, "cap_dur")

        out = config.output_dir() / result.output_clip
        dur = clipper.probe_duration(out)
        assert dur == pytest.approx(vert.duration, abs=1.0)

    def test_hostile_text_renders(self):
        src = video_synthetic()
        horiz, vert = _build_vertical_job(src, "cap_hostile")
        segs = _transcript_hostile()
        caps = captions_for_window(segs, horiz.source_start, horiz.source_end)
        assert len(caps) > 0
        result = render_captioned(vert, caps, "cap_hostile")
        assert result.qc_status == "passed"
        assert result.caption_count > 0

    def test_no_captions_skips_render(self):
        src = video_synthetic()
        horiz, vert = _build_vertical_job(src, "cap_empty")
        result = render_captioned(vert, [], "cap_empty")
        assert result.caption_count == 0

    def test_missing_vertical_raises(self):
        fake = SubtitleRenderResult.__fields_set__
        from models.schemas import VerticalClipEntry, CropGeometry
        geom = CropGeometry(
            source_width=640, source_height=360,
            crop_width=202, crop_height=360,
            crop_x=219, crop_y=0,
            target_width=506, target_height=900,
        )
        v = VerticalClipEntry(
            clip_id="clip_nonexist",
            source_clip="nope.mp4",
            source_width=640, source_height=360,
            output_width=506, output_height=900,
            crop=geom, duration=10.0,
            path="nope.mp4",
        )
        with pytest.raises(ClipFilesystemError):
            render_captioned(v, [CaptionSegment(text="x", start=0.0, end=1.0)], "nope")


# ---------------------------------------------------------------------------
# render_captioned_clips + manifest
# ---------------------------------------------------------------------------

class TestRenderCaptionedClipsReal:
    def test_manifest_written(self):
        src = video_synthetic()
        horiz, vert = _build_vertical_job(src, "cap_man")
        segs = _transcript_with_words()
        caps = captions_for_window(segs, horiz.source_start, horiz.source_end)
        by_clip = {vert.clip_id: caps}
        results = render_captioned_clips([vert], by_clip, "cap_man")
        assert len(results) == 1

        manifest = config.output_dir() / "cap_man" / "clips_captioned" / "manifest.json"
        assert manifest.exists()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert len(payload["subtitles"]) == 1
        entry = payload["subtitles"][0]
        assert entry["clip_id"] == "clip_000"
        assert entry["qc_status"] == "passed"
        assert entry["subtitle_source"] == "transcript"
        assert entry["caption_count"] == len(caps)

    def test_empty_verticals(self):
        results = render_captioned_clips([], {}, "cap_empty_v")
        assert results == []


# ---------------------------------------------------------------------------
# write_captioned_manifest (standalone)
# ---------------------------------------------------------------------------

class TestWriteCaptionedManifest:
    def test_writes_json(self):
        r = SubtitleRenderResult(
            clip_id="clip_000", source_clip="v.mp4", output_clip="c.mp4",
            duration=10.0, caption_count=3, qc_status="passed",
        )
        p = write_captioned_manifest("job_test", [r], source_url="test.mp4")
        assert p.exists()
        payload = json.loads(p.read_text(encoding="utf-8"))
        assert payload["source"] == "test.mp4"
        assert len(payload["subtitles"]) == 1
