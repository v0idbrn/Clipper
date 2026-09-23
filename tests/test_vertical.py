# Fase 4 - composicion vertical 9:16 (core/vertical.py).
#
# Sincronizado con el contrato REAL del modulo (verificado por probe directo),
# no con versiones anteriores del diseño:
#
#   - vertical.calculate_vertical_crop(width, height, *, target_width=None,
#     target_height=None, max_upscale=None, mode="center_crop")
#       -> CropGeometry   (funcion pura, sin I/O; salida SIEMPRE par y 9:16).
#
#   - Geometria center_crop real verificada sobre el modulo:
#       1920x1080 -> crop 608x1080 x=656 y=0        target 1080x1920
#       1280x720  -> crop 406x720  x=437 y=0        target 1016x1800
#       1440x1080 -> crop 608x1080 x=416 y=0        target 1080x1920
#       1080x1080 -> crop 608x1080 x=236 y=0        target 1080x1920
#       640x360   -> crop 202x360  x=219 y=0 (cap) target 506x900
#       1080x1920 -> identidad                      target 1080x1920
#       3840x2160 -> crop 1216x2160 x=1312 y=0      target 1080x1920
#       720x720   -> crop 406x720  x=157  y=0       target 1016x1800
#       200x400   -> crop 200x356  x=0   y=22       target 500x890
#       740x740   -> crop 416x740  x=162  y=0       target 1040x1850
#       640x480   -> crop 270x480  x=185  y=0       target 676x1200
#
#   - render_vertical(entry, job_id) -> VerticalClipEntry   (un solo entry).
#   - core.clipper.video_synthetic: fixture local offline (mp4 con ffmpeg,
#     sin red ni YouTube).

from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.clipper as clipper
import core.vertical as vertical
import config
from models.schemas import CropGeometry, HookCandidate, VerticalClipEntry
from tests.fixtures import video_synthetic

_9_16 = 9 / 16
_MAX_UPSCALE = vertical.MAX_UPSCALE


def _hook(start: float = 2.0, end: float = 10.0, score: float = 0.8) -> HookCandidate:
    return HookCandidate(start=start, end=end, score=score, title='Hook', reason='test')


def _geom(src_w: int, src_h: int) -> CropGeometry:
    return vertical.calculate_vertical_crop(src_w, src_h)


# ---------------------------------------------------------------------------
# Geometria pura espejada con el modulo real (probe -> valores esperados).
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_1920x1080(self):
        g = vertical.calculate_vertical_crop(1920, 1080)
        assert (g.crop_width, g.crop_height) == (608, 1080)
        assert g.crop_x == 656 and g.crop_y == 0
        assert (g.target_width, g.target_height) == (1080, 1920)

    def test_1280x720(self):
        g = vertical.calculate_vertical_crop(1280, 720)
        assert (g.crop_width, g.crop_height) == (406, 720)
        assert g.crop_x == 437 and g.crop_y == 0
        assert (g.target_width, g.target_height) == (1016, 1800)

    def test_1440x1080(self):
        g = vertical.calculate_vertical_crop(1440, 1080)
        assert (g.crop_width, g.crop_height) == (608, 1080)
        assert g.crop_x == 416 and g.crop_y == 0
        assert (g.target_width, g.target_height) == (1080, 1920)

    def test_1080x1080(self):
        g = vertical.calculate_vertical_crop(1080, 1080)
        assert (g.crop_width, g.crop_height) == (608, 1080)
        assert g.crop_x == 236 and g.crop_y == 0
        assert (g.target_width, g.target_height) == (1080, 1920)

    def test_640x360_upscale_capped(self):
        g = vertical.calculate_vertical_crop(640, 360)
        assert (g.crop_width, g.crop_height) == (202, 360)
        assert g.crop_x == 219 and g.crop_y == 0
        assert g.crop_width % 2 == 0 and g.crop_height % 2 == 0
        assert g.target_width % 2 == 0 and g.target_height % 2 == 0
        assert g.target_width / g.target_height == pytest.approx(_9_16, abs=0.02)
        assert (g.target_width, g.target_height) == (506, 900)

    def test_1080x1920_identity(self):
        g = vertical.calculate_vertical_crop(1080, 1920)
        assert (g.crop_width, g.crop_height) == (1080, 1920)
        assert g.crop_x == 0 and g.crop_y == 0
        assert (g.target_width, g.target_height) == (1080, 1920)

    def test_3840x2160(self):
        g = vertical.calculate_vertical_crop(3840, 2160)
        assert (g.crop_width, g.crop_height) == (1216, 2160)
        assert g.crop_x == 1312 and g.crop_y == 0
        assert (g.target_width, g.target_height) == (1080, 1920)

    def test_720x720(self):
        g = vertical.calculate_vertical_crop(720, 720)
        assert (g.crop_width, g.crop_height) == (406, 720)
        assert g.crop_x == 157 and g.crop_y == 0
        assert (g.target_width, g.target_height) == (1016, 1800)

    def test_200x400(self):
        g = vertical.calculate_vertical_crop(200, 400)
        assert (g.crop_width, g.crop_height) == (200, 356)
        assert g.crop_x == 0 and g.crop_y == 22
        assert (g.target_width, g.target_height) == (500, 890)

    def test_740x740(self):
        g = vertical.calculate_vertical_crop(740, 740)
        assert (g.crop_width, g.crop_height) == (416, 740)
        assert g.crop_x == 162 and g.crop_y == 0
        assert (g.target_width, g.target_height) == (1040, 1850)

    def test_640x480(self):
        g = vertical.calculate_vertical_crop(640, 480)
        assert (g.crop_width, g.crop_height) == (270, 480)
        assert g.crop_x == 185 and g.crop_y == 0
        assert (g.target_width, g.target_height) == (676, 1200)


class TestGeometryBounds:
    @pytest.mark.parametrize(
        "w, h",
        [
            (1920, 1080), (1280, 720), (1440, 1080), (1080, 1080),
            (640, 360), (1080, 1920), (3840, 2160), (720, 720),
            (200, 400), (740, 740), (640, 480),
        ],
    )
    def test_crop_inside_frame(self, w, h):
        g = vertical.calculate_vertical_crop(w, h)
        assert g.crop_x >= 0 and g.crop_y >= 0
        assert g.crop_x + g.crop_width <= w
        assert g.crop_y + g.crop_height <= h
        assert g.crop_width > 0 and g.crop_height > 0
        assert g.crop_width % 2 == 0 and g.crop_height % 2 == 0
        assert g.target_width % 2 == 0 and g.target_height % 2 == 0

    def test_output_ratio_and_upscale_limit(self):
        for w, h in [
            (1920, 1080), (1280, 720), (1440, 1080), (1080, 1080),
            (640, 360), (1080, 1920), (3840, 2160), (720, 720),
        ]:
            g = vertical.calculate_vertical_crop(w, h)
            assert g.target_width / g.target_height == pytest.approx(_9_16, abs=0.02)
            assert g.target_width <= w * _MAX_UPSCALE + 2
            assert g.target_height <= h * _MAX_UPSCALE + 2


class TestGeometryDefaultsAndInvalid:
    def test_none_uses_default(self):
        g = vertical.calculate_vertical_crop(1920, 1080)
        assert (g.target_width, g.target_height) == (1080, 1920)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(width=0, height=1080),
            dict(width=1920, height=0),
            dict(width=-5, height=1080),
            dict(width=1920, height=-5),
            dict(width=float('nan'), height=1080),
            dict(width=1920, height=float('inf')),
            dict(width=float('-inf'), height=1080),
            dict(width=1.5, height=1080),
            dict(width=1920, height=1080, target_width=0),
            dict(width=1920, height=1080, target_height=0),
            dict(width=1920, height=1080, target_width=-1),
            dict(width=1920, height=1080, max_upscale=0.5),
            dict(width=1920, height=1080, max_upscale=float('nan')),
            dict(width=1920, height=1080, mode='fancy_ai_crop'),
        ],
    )
    def test_raises_value_error(self, kwargs):
        with pytest.raises(ValueError):
            vertical.calculate_vertical_crop(**kwargs)


# ---------------------------------------------------------------------------
# Modelo VerticalClipEntry (contrato REAL: source_* + crop obligatorios).
# ---------------------------------------------------------------------------


class TestVerticalClipEntryModel:
    def test_model_defaults_and_ratio(self):
        g = _geom(1920, 1080)
        e = VerticalClipEntry(
            clip_id='clip_000',
            source_clip='job_x/clips_raw/clip_000.mp4',
            source_width=g.source_width,
            source_height=g.source_height,
            output_width=g.target_width,
            output_height=g.target_height,
            crop=g,
            duration=40.0,
            path='job_x/clips_vertical/clip_000.mp4',
        )
        assert e.aspect_ratio == '9:16'
        assert e.composition_mode == 'center_crop'
        assert e.qc_status == 'pending'
        assert e.duration > 0

    def test_model_rejects_non_9x16_output(self):
        g = _geom(1920, 1080)
        with pytest.raises(Exception) as exc:
            VerticalClipEntry(
                clip_id='clip_000',
                source_clip='job_x/clips_raw/clip_000.mp4',
                source_width=g.source_width,
                source_height=g.source_height,
                output_width=1080,
                output_height=1080,
                crop=g,
                duration=40.0,
                path='job_x/clips_vertical/clip_000.mp4',
            )
        assert '9:16' in str(exc.value) or 'aspect' in str(exc.value)

    def test_model_rejects_zero_output_dimensions(self):
        g = _geom(1920, 1080)
        with pytest.raises(Exception):
            VerticalClipEntry(
                clip_id='clip_000',
                source_clip='job_x/clips_raw/clip_000.mp4',
                source_width=g.source_width,
                source_height=g.source_height,
                output_width=0,
                output_height=1920,
                crop=g,
                duration=40.0,
                path='job_x/clips_vertical/clip_000.mp4',
            )

    def test_model_requires_source_dims_and_crop(self):
        with pytest.raises(Exception):
            VerticalClipEntry(
                clip_id='clip_000',
                source_clip='job_x/clips_raw/clip_000.mp4',
                output_width=1080,
                output_height=1920,
                duration=40.0,
                path='job_x/clips_vertical/clip_000.mp4',
            )


# ---------------------------------------------------------------------------
# Render FFmpeg REAL (offline, ffprobe real, sin red; un solo entry por
# pasada como dicta render_vertical actual).
# ---------------------------------------------------------------------------


class TestRenderVerticalFFmpegReal:
    def test_real_offline_render(self):
        src = video_synthetic()
        assert src.exists()

        entries = clipper.clip_hooks(
            str(src), 'direct_file', [_hook(2.0, 10.0)], 'job_vert_real',
            raw_path=src,
            media_duration=clipper.probe_duration(str(src)),
        )
        assert len(entries) == 1
        horiz = entries[0]

        v = vertical.render_vertical(horiz, 'job_vert_real')

        assert v.clip_id == 'clip_000'
        assert v.qc_status == 'passed'
        assert v.aspect_ratio == '9:16'
        assert v.crop is not None
        out = config.output_dir() / v.path
        assert out.exists() and out.stat().st_size > 0

        w, h = clipper.probe_dimensions(out)
        assert (w, h) == (v.output_width, v.output_height)
        assert clipper._probe_has_video(out)
        assert clipper._probe_has_audio(out)
        duration = clipper.probe_duration(out)
        assert duration == pytest.approx(v.duration, abs=0.5)
        assert 15.0 <= duration <= 90.0

    def test_manifest_vertical_traced(self):
        src = video_synthetic()
        assert src.exists()

        entries = clipper.clip_hooks(
            str(src), 'direct_file', [_hook(2.0, 10.0)], 'job_vert_manifest',
            raw_path=src,
            media_duration=clipper.probe_duration(str(src)),
        )
        v = vertical.render_vertical(entries[0], 'job_vert_manifest')
        vertical.write_vertical_manifest('job_vert_manifest', [v])

        manifest = config.output_dir() / 'job_vert_manifest' / 'clips_vertical' / 'manifest.json'
        assert manifest.exists()
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        assert len(payload['clips']) == 1
        clip = payload['clips'][0]
        assert clip['clip_id'] == 'clip_000'
        assert clip['aspect_ratio'] == '9:16'
        assert clip['composition_mode'] == 'center_crop'
        assert clip['qc_status'] == 'passed'
        assert clip['crop']['crop_width'] > 0
        geom = vertical.calculate_vertical_crop(
            clip['source_width'], clip['source_height']
        )
        assert clip['output_width'] == geom.target_width
        assert clip['output_height'] == geom.target_height
        out = config.output_dir() / clip['path']
        assert out.exists() and out.stat().st_size > 0