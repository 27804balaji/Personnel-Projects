import speech_recognition as sr
import pyttsx3
import pywhatkit
import time
import datetime
import wikipedia
import cv2
from pyfiglet import figlet_format

heading = "Tony StarK"
head = figlet_format(heading,font='slant',justify="center")
print(head)

machine = pyttsx3.init()
machine.setProperty('rate', 150)
machine.setProperty('volume', 1.0)

def talk(txt):
    machine.say(txt)
    machine.runAndWait()

def input_instruction(listener):
    instruction = ""
    try:
        with sr.Microphone() as origin:
            listener.adjust_for_ambient_noise(origin)
            print("Listening...")
            speech = listener.listen(origin, timeout=5, phrase_time_limit=7)
            instruction = listener.recognize_google(speech)
            instruction = instruction.lower()
            if "Tony Stark" in instruction or 'Tony' in instruction or 'Stark' in instruction:
                instruction = instruction.replace("Tony Stark", "")
            print(instruction)
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"Error during listening: {e}")

    return instruction
   
   
def play_TonyStark():
    recognizer = sr.Recognizer() 
    cap = cv2.VideoCapture(0)  

    while True:
        instruction = input_instruction(recognizer)

        if "play" in instruction:
            song = instruction.replace("play", '')
            talk("Playing" + song)
            pywhatkit.playonyt(song)
            talk("Enjoyyy !")
            print("Enjoyy !")
            break       
        
        elif "search wikipedia for" in instruction or "search for" in instruction or "search" in instruction:
            search_query = instruction.replace("search wikipedia for", "").replace("search for", "").replace("search", "")
            search_query = search_query.strip()  

            if search_query:
                result = wikipedia.search(search_query)
                if result:
                    for search in result:
                        page = wikipedia.page(search)
                        print(page.summary)
                        talk(page.summary)
                    print(result)
                    talk(result)
                else:
                    print("No results found on Wikipedia.")
                    talk("No results found on Wikipedia.")
            else:
                print("Search query is empty. Please provide a valid query.")
                talk("Search query is empty. Please provide a valid query.")
        
        elif "time" in instruction:
            Time = time.time.now().strftime('%I:%M%p')
            talk("Current Time" + Time)
        
        elif "good morning" in instruction:
            talk("good morning")
            print("good morning")

        elif "good afternoon" in instruction:
            talk("good afternoon")
            print("good afternoon")

        elif "good evening" in instruction:
            talk("good evening")
            print("good evening")

        elif "good night" in instruction:
            talk("good night")
            print("good night")
            
        elif "tony" in instruction or "tony stark" in instruction:
            talk("Good to see You")        
            print("Good to see You")        

        elif "date" in instruction:
            Date = datetime.datetime.now().strftime('%d/%m/%Y')
            talk("Today's Date" + Date)
            print("Today's Date" + Date)
        
        elif "how are you" in instruction:
            talk("Thanks! I'm fine..., How about you")
            print("Thanks! I'm fine..., How about you")
        
        elif "what is your name" in instruction or "what's your name" in instruction or "who are you" in instruction or "hi tony" in instruction or "hi buddy" in instruction:
            talk("hey! , I'm Tony Stark..., What can I do for you")
            print("hey! , I'm Tony Stark..., What can I do for you")
        
        elif "stop" in instruction or "exit" in instruction or "quit" in instruction:
            print("Assistant is stopping. Goodbye!")
            talk("Assistant is stopping. Goodbye!")
            break
        
        elif "camera" in instruction or "open camera" in instruction:
            while True:
                ret, frame = cap.read()
                cv2.imshow('Camera Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if "close camera" in input_instruction(recognizer):
                    break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()
            cap.release()
            print("Camera feed closed.")
            talk("Camera feed closed.")
        
        else:
            talk('Can you please kindly repeat...')

    cap.release() 

machine.setProperty('rate', 200)
machine.setProperty('volume', 0.8)

play_TonyStark()
