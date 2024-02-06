from ultralytics import YOLO
import cv2
import cvzone
import math
from openai import OpenAI
import speech_recognition as sr
import tempfile
import wave
import os
from dotenv import load_dotenv
from threading import Thread
import sys

# Load environment variables from .env file
load_dotenv()

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("API_KEY"))

# Flag to indicate if the audio capture thread should continue running
capture_audio = True

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.translations.create(
            model="whisper-1",
            file=audio_file,
        )
    return transcript.text

def perform_object_detection(frame):
    result = model(frame, classes=0, verbose=False)[0]

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h))

        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = box.cls[0]

        cvzone.putTextRect(frame, f'{conf}', (max(0, x1), max(35, y1)), scale=0.5)

    cv2.imshow('Camera Feed', frame)

def video_capture():
    cap = cv2.VideoCapture(0)
    while True:
        flag, frame = cap.read()
        if not flag:
            exit(1)
        perform_object_detection(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def audio_capture_and_transcribe(duration=3, interval=10):
    recognizer = sr.Recognizer()

    global capture_audio  # Declare capture_audio as global inside the function

    while capture_audio:
        try:
            with sr.Microphone() as source:
                # print(f"Say something... (capturing for {duration} seconds)")
                audio_data = recognizer.listen(source, phrase_time_limit=duration)

            if not audio_data:
                print("No audio data received.")
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
                channels = 1
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

        except KeyboardInterrupt:
            # Handle Ctrl+C, set capture_audio flag to False and break the loop
            capture_audio = False
            break

if __name__ == "__main__":
    # Create and start separate threads for video and audio
    video_thread = Thread(target=video_capture)
    audio_thread = Thread(target=audio_capture_and_transcribe)

    video_thread.start()
    audio_thread.start()

    try:
        # Wait for both threads to finish
        video_thread.join()
        audio_thread.join()
    except KeyboardInterrupt:
        # Handle Ctrl+C, set capture_audio flag to False and wait for the audio thread to finish
        capture_audio = False
        audio_thread.join()

    sys.exit(0)
