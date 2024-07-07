from google.cloud import texttospeech
from pydub import AudioSegment
from pydub.playback import play
import os
import re
from openai import OpenAI
import speech_recognition as sr
from dotenv import load_dotenv


load_dotenv()
speed_talk = 1.1

openai_api_key = os.getenv('OPENAI_API_KEY')
google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path

client = OpenAI(api_key=openai_api_key)

# Initializing the Google Cloud Text-to-Speech client
tts_client = texttospeech.TextToSpeechClient()

# Function for pre-processing text
def preprocess_text(text):
    return text

# Function to play text by voice using Google Cloud Text-to-Speech
def speak(text, speed=speed_talk, voice_name="ru-RU-Standard-C"):
    text = preprocess_text(text)

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ru-RU",
        name=voice_name,
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speed
    )

    voice_response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    filename = "output.mp3"
    with open(filename, "wb") as out:
        out.write(voice_response.audio_content)
        print(f'Audio content written to file "{filename}"')

    sound = AudioSegment.from_file(filename)
    play(sound)
    os.remove(filename)

# Function for listening and speech recognition
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Слушаю...")
        speak("Слушаю...", speed=speed_talk)
        audio = recognizer.listen(source)

    try:
        voice_command = recognizer.recognize_google(audio, language='ru-RU')
        print(f"Вы сказали: {voice_command}")
        return voice_command
    except sr.UnknownValueError:
        print("Не удалось распознать речь")
        speak("Не удалось распознать речь", speed=speed_talk)
        return ""
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания речи; {e}")
        speak(f"Ошибка сервиса распознавания речи; {e}", speed=speed_talk)
        return ""

# Function for communicating with GPT-4
def chat_with_gpt(prompt):
    try:
        gpt_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return gpt_response.choices[0].message.content
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return "Произошла ошибка. Попробуйте позже."

if __name__ == "__main__":
    while True:
        command = listen()
        if command.lower() == "стоп":
            speak("До свидания!", speed=speed_talk)
            break
        response = chat_with_gpt(command)
        print(f"GPT-4: {response}")
        speak(response, speed=speed_talk)
