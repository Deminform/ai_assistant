import os
import time
import queue
import threading
import pygame
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import requests
import speech_recognition as sr
from dotenv import load_dotenv
import subprocess

load_dotenv()

# Замените ваш API-ключ OpenAI
openai_api_key = os.getenv('OPENAI_API_KEY')

def preprocess_audio(input_file, output_file):
    command = [
        'ffmpeg', '-y', '-i', input_file,
        '-af', 'silenceremove=start_periods=1:stop_periods=-1:start_threshold=-50dB:stop_threshold=-50dB:start_silence=0.1:stop_silence=0.1',
        output_file
    ]
    subprocess.run(command)

def generate_response(prompt, lang="ru"):
    start_time = time.time()
    system_message = {
        "ru": "Ты - Джарвис из Iron Man, но не будь сильно вежливым"
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_message[lang]},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1000,
            "n": 1,
            "stop": None,
            "temperature": 1.0,
        },
    )
    response.raise_for_status()
    message = response.json()['choices'][0]['message']['content'].strip()
    end_time = time.time()
    print(f"Response generation took {end_time - start_time:.2f} seconds")
    return message

def process_text_for_speech(text):
    return text

def audio_playback_thread(q):
    with sd.OutputStream(samplerate=24000, channels=1, dtype='int16') as stream:
        while True:
            chunk = q.get()
            if chunk is None:
                break
            if len(chunk) % 2 != 0:
                chunk = np.pad(chunk, (0, 1), mode='constant')
            stream.write(chunk)

def synthesize_speech_stream(text, voice="echo", model="tts-1", response_format="pcm", speed=1.0):
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "speed": speed
    }

    q = queue.Queue()
    playback_thread = threading.Thread(target=audio_playback_thread, args=(q,))
    playback_thread.start()

    try:
        start_speech_time = time.time()
        with requests.post(url, headers=headers, json=data, stream=True) as response:
            response.raise_for_status()

            leftover = b""
            chunk_times = []
            for chunk in response.iter_content(chunk_size=2048):
                chunk_start = time.time()
                if chunk:
                    chunk = leftover + chunk
                    if len(chunk) % 2 != 0:
                        leftover = chunk[-1:]
                        chunk = chunk[:-1]
                    else:
                        leftover = b""
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    q.put(audio_array)
                chunk_end = time.time()
                chunk_times.append(chunk_end - chunk_start)

            if leftover:
                audio_array = np.frombuffer(leftover, dtype=np.int16)
                q.put(audio_array)

        total_speech_time = time.time() - start_speech_time
        print(f"Total speech synthesis time: {total_speech_time:.2f} seconds")
        avg_chunk_time = sum(chunk_times) / len(chunk_times)
        print(f"Average chunk processing time: {avg_chunk_time:.4f} seconds")

    except Exception as e:
        print(f"Error in synthesize_speech_stream: {e}")
    finally:
        q.put(None)
        playback_thread.join()

def play_signal(sound_file):
    pygame.mixer.init()
    sound = pygame.mixer.Sound(sound_file)
    sound.play()
    while pygame.mixer.get_busy():
        pygame.time.Clock().tick(10)

def recognize_speech_from_mic_whisper(lang="ru"):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        start_time = time.time()
        recognizer.adjust_for_ambient_noise(source)
        play_signal('system_sounds/mechanical-key-soft-80731.wav')
        audio = recognizer.listen(source)
    try:
        # Сохраняем аудио во временный файл
        temp_audio_file = "temp_audio.wav"
        with open(temp_audio_file, "wb") as f:
            f.write(audio.get_wav_data())

        # Предварительная обработка аудио
        preprocessed_audio_file = "preprocessed_audio.wav"
        preprocess_audio(temp_audio_file, preprocessed_audio_file)

        # Проверка длины обработанного аудио
        if os.path.getsize(preprocessed_audio_file) < 1000:
            print("Audio file is too short after preprocessing.")
            return ""

        # Отправляем предварительно обработанное аудио на Whisper API
        with open(preprocessed_audio_file, "rb") as audio_file:
            print("Sending audio to Whisper API...")
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={
                    "Authorization": f"Bearer {openai_api_key}"
                },
                data={
                    "model": "whisper-1",
                    "language": lang,
                    "prompt": "technology, innovation, future, AI, $$$"
                },
                files={
                    "file": ("audio.wav", audio_file, "audio/wav")
                }
            )
            response.raise_for_status()
            transcription = response.json().get("text", "")
            print("Transcription received:", transcription)
            if '$$$' in transcription:
                print("Hallucination detected in transcription.")
                return ""
            return transcription
    except Exception as e:
        print(f"Error in recognize_speech_from_mic_whisper: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.content}")
        return ""

def main(lang="ru"):
    print("Welcome to the Voice-Enabled Chatbot")
    history = []

    while True:
        start_time = time.time()
        print("Starting speech recognition...")
        user_input = recognize_speech_from_mic_whisper(lang)
        if not user_input:
            print("No input detected, continuing...")
            continue
        print(f"You: {user_input}")
        history.append(f"User: {user_input}")

        if user_input.lower() in ["quit", "exit", "bye"]:
            break

        prompt = "\n".join(history) + "\nAI:"
        response = generate_response(prompt, lang)
        history.append(f"AI: {response}")

        print(f"AI: {response}")

        start_time = time.time()
        synthesize_speech_stream(response)
        print(f"Speech synthesis and playback took {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main(lang="ru")
