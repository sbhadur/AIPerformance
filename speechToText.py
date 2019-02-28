import speech_recognition as sr

def speechToText():
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=0)
    with mic as source:
        audio = r.listen(source)
        print(r.recognize_google(audio,show_all = True))

speechToText()
