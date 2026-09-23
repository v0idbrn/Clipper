"""Fixtures locales de test, cacheados en el repo.

Regla: los tests automatizados JAMÁS dependen de red (ni YouTube ni ningún
host). Todo lo que un test necesita para audio/extracción está acá, generado
una sola vez con herramientas locales:

- audio_tone.{mp3,wav}: tono sintético 440Hz 2s (ffmpeg lavfi sine), para
  tests de probe_duration / extractor / ffmpeg que solo quieren un archivo.
- audio_speech.{mp3,wav}: frase hablada corta sintetizada con TTS de Windows
  (System.Speech), sin copyright, para tests de transcripción real (Whisper
  escucha voz, no un tono).
- video_synthetic.mp4: video sintético de 20s (ffmpeg lavfi testsrc + sine),
  sin copyright, para tests de recorte (Fase 3) que cortan con ffmpeg local.

Regeneración (no debe hacer falta): `docs/testing_offline.md` documenta los
comandos exactos.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def audio_mp3() -> Path:
    return FIXTURES_DIR / "audio_speech.mp3"


def audio_wav() -> Path:
    return FIXTURES_DIR / "audio_speech.wav"


def tone_mp3() -> Path:
    return FIXTURES_DIR / "audio_tone.mp3"


def tone_wav() -> Path:
    return FIXTURES_DIR / "audio_tone.wav"


def video_synthetic() -> Path:
    return FIXTURES_DIR / "video_synthetic.mp4"


def synthetic_video(
    path: Path, *, duration: float = 120.0, size: str = "320x180", rate: int = 8
) -> Path:
    """Genera (si falta) un video sintético de `duration` s con ffmpeg local.

    Offline y determinista; pensado para tests de ventanas narrativas largas
    (Fase 3), donde el fixture de 20 s es demasiado corto para producir dos
    ventanas distintas.
    """
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={size}:rate={rate}",
        "-f", "lavfi", "-i", f"sine=frequency=330:duration={duration}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path