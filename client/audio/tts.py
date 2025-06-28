import os
import tempfile
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from playsound import playsound

load_dotenv()

API_KEY = os.getenv("TTS_API_KEY")
if not API_KEY:
    raise RuntimeError("TTS_API_KEY tidak ditemukan di .env")

client = ElevenLabs(api_key=API_KEY)

def speak_text(text: str):
    # 1) Generate audio chunks (generator)
    chunks = client.text_to_speech.convert(
        text=text,
        voice_id="EXAVITQu4vr4xnSDxMaL",
        model_id="eleven_monolingual_v1"
    )

    # 2) Gabungkan semua chunk jadi bytes
    audio_bytes = b"".join(chunks)

    # 3) Tulis ke temporary file yang akan ditutup otomatis
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # 4) Putar audio dari temporary file
        playsound(tmp_path)
    finally:
        # 5) Hapus temporary file meski error saat pemutaran
        try:
            os.remove(tmp_path)
        except OSError:
            pass