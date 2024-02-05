from openai import OpenAI
import speech_recognition as sr
import tempfile
import wave
import os
import time

client = OpenAI(api_key='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')

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.translations.create(
            model="whisper-1",
            file=audio_file,
        )
    return transcript.text

def capture_and_transcribe(duration=10, interval=10):
    recognizer = sr.Recognizer()

    while True:
        with sr.Microphone() as source:
            print(f"Say something... (capturing for {duration} seconds)")
            audio_data = recognizer.listen(source, phrase_time_limit=duration)

        if not audio_data:
            print("No audio data received.")
            time.sleep(interval)
            continue

        # Save the captured audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
            channels = 1  # Assuming mono audio input, adjust if needed
            with wave.open(temp_wav_path, 'wb') as wave_file:
                wave_file.setnchannels(channels)
                wave_file.setsampwidth(audio_data.sample_width)
                wave_file.setframerate(audio_data.sample_rate)
                wave_file.writeframes(audio_data.frame_data)
            print(f"Audio saved to: {temp_wav_path}")

        try:
            text = transcribe_audio(temp_wav_path)
            print(f"{text}")
        except Exception as e:
            print(f"Error during transcription: {e}")
        finally:
            os.remove(temp_wav_path)

if __name__ == "__main__":
    capture_and_transcribe()

