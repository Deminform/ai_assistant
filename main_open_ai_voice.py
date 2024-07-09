import os
import time
import re
import queue
import threading
import pygame
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import requests
import speech_recognition as sr
from dotenv import load_dotenv
import subprocess

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


def preprocess_audio(input_file, output_file):
    command = [
        'ffmpeg', '-y', '-i', input_file,
        '-af', 'silenceremove=start_periods=1:stop_periods=-1:start_threshold=-50dB:stop_threshold=-50dB:start_silence=0.1:stop_silence=0.1',
        output_file
    ]
    subprocess.run(command)


def is_significant_audio(file_path, threshold=500):
    sample_rate, audio_data = wavfile.read(file_path)
    audio_data = np.abs(audio_data)
    max_amplitude = np.max(audio_data)
    print(f"Max amplitude in audio: {max_amplitude}")
    return max_amplitude > threshold


def generate_response(prompt, lang="ru"):
    start_time = time.time()
    system_message = {
        "ru": "Ты - Лучший на свете програмист, ты все знаешь про Python и готов помочь. Твоя задача работать с кодом, улучшать то о чем тебя просят в коде"
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
            "max_tokens": 3000,
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

    audio_queue = queue.Queue()

    def play_signal_and_listen(audio_queue):
        play_signal('system_sounds/soft-notice-146623.mp3')
        with microphone as source:
            audio = recognizer.listen(source)
        audio_queue.put(audio)

    play_thread = threading.Thread(target=play_signal_and_listen, args=(audio_queue,))
    play_thread.start()
    play_thread.join()

    try:
        audio = audio_queue.get()

        temp_audio_file = "temp_audio.wav"
        with open(temp_audio_file, "wb") as f:
            f.write(audio.get_wav_data())

        preprocessed_audio_file = "preprocessed_audio.wav"
        preprocess_audio(temp_audio_file, preprocessed_audio_file)

        if not is_significant_audio(preprocessed_audio_file):
            print("No significant audio detected.")
            return ""

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


def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def request_file_path():
    synthesize_speech_stream("Пожалуйста, укажите путь к файлу.")
    file_path = input("Введите путь к файлу: ")
    return file_path


def request_file_content():
    synthesize_speech_stream("Что вы хотите записать в файл?")
    return recognize_speech_from_mic_whisper()


def write_to_file(file_path, content):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:  # 'w' для перезаписи
            file.write(content + "\n")
        return True
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")
        return False


def is_code(content):
    code_indicators = ['def ', 'class ', '{', '}', ';', 'import ', '#include']
    return any(indicator in content for indicator in code_indicators)


def clean_answer_text(answer_text):
    punctuation_pattern = re.compile(r'[^\w\s]')
    return re.sub(punctuation_pattern, '', answer_text).lower()


def discuss_file_content(content):
    while True:
        synthesize_speech_stream("Что вы хотите изменить в содержимом файла?")
        user_input = recognize_speech_from_mic_whisper()

        if clean_answer_text(user_input) in ['отмена', 'не надо', 'стой', 'остановись', 'не нужно', 'стоп', 'стоп не надо', 'не надо стоп']:
            synthesize_speech_stream("Операция отменена.")
            return None

        prompt = f"Сгенерируй текст на тему: {user_input}"
        content = generate_response(prompt)
        print(f"Новое содержимое для записи: {content}")

        synthesize_speech_stream(f"Вот предложение по изменению: {content}\nХотите записать это изменение в файл?")
        user_confirmation = recognize_speech_from_mic_whisper()
        if clean_answer_text(user_confirmation) in ['да', 'записывай', 'подтверждаю', 'да записывай']:
            return content
        else:
            synthesize_speech_stream("Операция отменена.")
            return None


def main(lang="ru"):
    print("Welcome to the Voice-Enabled Chatbot")
    history = []

    while True:
        print("Starting speech recognition...")
        user_input = recognize_speech_from_mic_whisper(lang)
        if not user_input:
            print("No input detected, continuing...")
            continue
        print(f"You: {user_input}")
        history.append(f"User: {user_input}")

        if clean_answer_text(user_input) in ["все пока", "мы закончили", "давай до свидания"]:
            break

        if clean_answer_text(user_input) in ['запиши в файл', 'запиши файл', 'файл запиши', 'добавь код', 'добавь текст', 'добавь код в файл', 'добавь текст в файл', 'добавь код файл']:
            file_path = request_file_path()
            if not file_path:
                synthesize_speech_stream("Не удалось распознать путь к файлу.")
                continue

            content = discuss_file_content(file_path)
            if content:
                success = write_to_file(file_path, content)
                if success:
                    synthesize_speech_stream("Запись выполнена успешно.")
                else:
                    synthesize_speech_stream("Ошибка при записи в файл.")
            continue

        if clean_answer_text(user_input) in ['открой файл', 'прочитай файл', 'проверь файл', 'посмотри файл', 'посмотри код', 'проверь код', 'посмотри на мой код', 'взгляни на мой код']:
            file_path = request_file_path()
            if not file_path:
                synthesize_speech_stream("Не удалось распознать путь к файлу.")
                continue

            content = read_file_content(file_path)
            history.append(content)
            if content:
                print(f"Содержимое файла: {content}")
                if is_code(content):
                    synthesize_speech_stream("Файл содержит код. Проверьте его на экране.")
                elif len(content) > 500:
                    synthesize_speech_stream("Файл слишком большой для чтения вслух. Проверьте его на экране.")
                else:
                    synthesize_speech_stream(f"Содержимое файла: {content}")
            else:
                synthesize_speech_stream("Не удалось прочитать файл.")
            continue

        prompt = "\n".join(history) + "\nAI:"
        response = generate_response(prompt, lang)
        history.append(f"AI: {response}")

        print(f"AI: {response}")

        start_time = time.time()
        synthesize_speech_stream(response)
        print(f"Speech synthesis and playback took {time.time() - start_time:.2f} seconds")

        print("Starting speech recognition immediately after playback...")


if __name__ == "__main__":
    main(lang="ru")
