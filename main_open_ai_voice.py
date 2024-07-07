import os
from openai import OpenAI
import gtts
import speech_recognition as sr
import pygame
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

load_dotenv()

# Замените ваш API-ключ OpenAI
openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_api_key)


def generate_response(prompt, lang="ru"):
    system_message = {
        "ru": "Ты - крутой друг пользователя, всегда поддерживаешь и даешь советы. Твой голос мужской, уверенный и дружелюбный. Твои ответы должны быть расслабленными и позитивными, как у настоящего друга.",
        "uk": "Ти - крутий друг користувача, завжди підтримуєш і даєш поради. Твій голос чоловічий, впевнений і дружелюбний. Твої відповіді мають бути розслабленими і позитивними, як у справжнього друга."
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
        temperature=0.7,
    )
    message = response.choices[0].message.content.strip()
    return message



def online_tts(text, lang="ru", speed=1.2):
    output_folder = os.path.expanduser("~/JarvisOutput")
    os.makedirs(output_folder, exist_ok=True)

    with NamedTemporaryFile(delete=False) as output_file:
        tts = gtts.gTTS(text, lang=lang, slow=False)
        tts.save(output_file.name)
        output_file.seek(0)

    pygame.init()
    pygame.mixer.init()

    sound = pygame.mixer.Sound(output_file.name)
    sound.set_volume(1.0)
    channel = sound.play()

    # Устанавливаем скорость воспроизведения
    channel.set_volume(speed)

    if channel is not None:
        channel.set_endevent(pygame.USEREVENT)
        is_playing = True
        while is_playing:
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT:
                    is_playing = False
                    break
            pygame.time.Clock().tick(10)

    pygame.mixer.quit()
    pygame.time.wait(500)
    os.remove(output_file.name)



def recognize_speech_from_mic(recognizer, microphone, lang="ru-RU"):
    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for your voice...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing your speech...")
        return recognizer.recognize_google(audio, language=lang)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")



def main(lang="ru", speed=1.2):
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

        online_tts(response, lang, speed)

if __name__ == "__main__":
    # Установка языка и скорости. Пример для русского языка и скорости 1.2
    main(lang="ru", speed=1.2)

