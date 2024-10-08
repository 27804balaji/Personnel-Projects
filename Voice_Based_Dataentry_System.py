import speech_recognition as sr
from langdetect import detect
from transformers import pipeline

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Load the warning detection model
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

def transcribe_audio(file_path):
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"

def detect_language(text):
    try:
        return detect(text)
    except:
        return "Unknown"

def filter_warnings(text):
    sentences = text.split('.')
    filtered_sentences = []
    
    for sentence in sentences:
        result = classifier(sentence)
        # Assuming warnings/scolding will be categorized as 'anger' or 'disgust'
        if any(label['label'] in ['anger', 'disgust'] and label['score'] > 0.5 for label in result[0]):
            continue
        filtered_sentences.append(sentence)
    
    return '. '.join(filtered_sentences) + '.'

def process_lecture(file_path):
    # Step 1: Transcribe audio
    transcript = transcribe_audio(file_path)
    
    # Step 2: Detect language
    language = detect_language(transcript)
    
    # Step 3: Filter warnings
    filtered_transcript = filter_warnings(transcript)
    
    # Save to file
    with open('lecture_transcript.txt', 'w', encoding='utf-8') as file:
        file.write(f"Language: {language}\n\n")
        file.write(filtered_transcript)
    
    print("Processing complete. Transcript saved to 'lecture_transcript.txt'.")

# Provide the path to the audio file
audio_file_path = 'path_to_your_audio_file.wav'
process_lecture(audio_file_path)
