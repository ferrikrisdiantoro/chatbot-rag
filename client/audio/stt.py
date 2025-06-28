import speech_recognition as sr

def recognize_speech_from_mic():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Silakan bicara...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language="id-ID")
        print(f"📝 Teks: {text}")
        return text
    except sr.UnknownValueError:
        return "Maaf, tidak bisa mengenali suara."
    except sr.RequestError as e:
        return f"Error dengan STT: {e}"
