# Testing offline — fixtures de audio

Los tests automatizados de AutoClipper **nunca dependen de red**. Verificado
durante la verificación real de la Fase 1: YouTube hace rate-limit (429 /
"Sign in to confirm you're not a bot") tras varias corridas, y pegarle a la red
en cada `pytest` convierte el suite en frágil. Por eso los fixtures viven en el
repo, en `tests/fixtures/`.

## Archivos

| Archivo              | Qué es                                                              | Para qué sirve                                   |
| -------------------- | ------------------------------------------------------------------- | ------------------------------------------------ |
| `audio_tone.wav/mp3` | Tono sintético 440 Hz, 2 s, mono 16 kHz (ffmpeg `lavfi` sine)       | `probe_duration`, extractor ffmpeg, E2E direct_file |
| `audio_speech.wav/mp3` | Frase hablada corta, ~6 s, TTS local de Windows (`System.Speech`) | transcripción real (Whisper escucha voz)         |
| `video_synthetic.mp4` | Video sintético 20 s (ffmpeg `lavfi` testsrc + sine, 640x360)       | recorte quirúrgico (Fase 3) con ffmpeg local     |

Ambos son deterministas y sin contenido con copyright (sintético / TTS local).

## Regeneración (no debería hacer falta)

```powershell
# Tono
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 -ac 1 tests/fixtures/audio_tone.wav
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 -ac 1 tests/fixtures/audio_tone.mp3

# Habla (Windows)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile("tests/fixtures/audio_speech.wav")
$s.Speak("Welcome to AutoClipper. This video demonstrates how long form content becomes vertical shorts.")
ffmpeg -y -i tests/fixtures/audio_speech.wav -codec:a libmp3lame -q:a 5 tests/fixtures/audio_speech.mp3

# Video sintético (sin copyright)
ffmpeg -y -f lavfi -i "testsrc=duration=20:size=640x360:rate=12" -f lavfi -i "sine=frequency=440:duration=20" -c:v libx264 -preset veryfast -crf 28 -c:a aac -shortest tests/fixtures/video_synthetic.mp4
```

## Uso

```python
from tests.fixtures import audio_mp3, tone_wav

path = audio_mp3()   # Path a tests/fixtures/audio_speech.mp3
```

Si algún test del futuro necesita un archivo distinto (ej. un video corto
con movimiento para la Fase de crop), agregarlo acá con el mismo criterio:
generar una vez, commitear, y que el test lo lea de `tests/fixtures.py`.