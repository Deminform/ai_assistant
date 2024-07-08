import os
import speech_recognition as sr
import requests
import pygame
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Замените ваш API-ключ OpenAI
openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_api_key)


def generate_response(prompt, lang="ru"):
    system_message = {
        "ru": "Ты - остроумный ассистент. Ты можешь предложить полезные советы по профессиональным вопросам. "
              "Твои ответы должны быть короткими, легкими и понятными. "
              "Когда перечисляешь элементы, делай большую паузу после каждого слова. "
              "Если ты перечисляешь элементы - ставь точку с запятой после каждого слова. "

    }
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message[lang]},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=1.0,
    )
    message = response.choices[0].message.content.strip()
    return message


def process_text_for_speech(text):
    # Удаляем нумерацию в начале строки (например, "1. ", "2. " и т.д.)

    return text


def synthesize_speech(text, voice="echo", model="tts-1", response_format="mp3", speed=1.0):
    # Обрабатываем текст для синтеза речи
    text = process_text_for_speech(text)

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
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        audio_content = response.content
        return audio_content
    else:
        raise Exception(f"Failed to synthesize speech: {response.text}")


def play_audio(audio_content):
    pygame.init()
    pygame.mixer.init()

    with NamedTemporaryFile(delete=False) as audio_file:
        audio_file.write(audio_content)
        temp_filename = audio_file.name  # Сохраняем имя файла

    # Теперь загружаем и проигрываем звук после закрытия файла
    sound = pygame.mixer.Sound(temp_filename)
    sound.play()
    while pygame.mixer.get_busy():
        pygame.time.Clock().tick(10)

    os.remove(temp_filename)  # Удаляем файл вручную после воспроизведения


def play_signal(sound_file):
    pygame.mixer.init()
    sound = pygame.mixer.Sound(sound_file)
    sound.play()
    while pygame.mixer.get_busy():
        pygame.time.Clock().tick(10)


def recognize_speech_from_mic(recognizer, microphone, lang="ru-RU"):
    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Playing start signal...")
        play_signal("mechanical-key-soft-80731.mp3")  # Воспроизведение сигнала начала
        print("Listening for your voice...")
        audio = recognizer.listen(source)
        print("Playing stop signal...")
        play_signal("mechanical-key-soft-80731.mp3")  # Воспроизведение сигнала окончания

    try:
        print("Recognizing your speech...")
        return recognizer.recognize_google(audio, language=lang)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")


def main(lang="ru"):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Welcome to the Voice-Enabled Chatbot")
    history = []

    while True:
        user_input = recognize_speech_from_mic(recognizer, microphone, lang=f"{lang}-{lang.upper()}")
        if user_input is None:
            continue

        print(f"You: {user_input}")
        history.append(f"User: {user_input}")

        if user_input.lower() in ["quit", "exit", "bye"]:
            break

        prompt = "\n".join(history) + "\nAI:"
        response = generate_response(prompt, lang)
        history.append(f"AI: {response}")

        print(f"AI: {response}")

        audio_content = synthesize_speech(response)
        play_audio(audio_content)


if __name__ == "__main__":
    main(lang="ru")
