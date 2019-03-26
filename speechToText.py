import speech_recognition as sr
import time
import laughtrain
import laughpred as lp

def speechToText():
    #Initialize Model
    model = laughtrain.fit()

    #Initialize the recognizer
    r = sr.Recognizer()

    #Use Sample Rate of 16000 Hz as recommended, device index refers to input for the mic"
    mic = sr.Microphone(device_index=0, sample_rate = 16000, chunk_size = 2048)

    with mic as source:
        #Adjust energy threshold
        r.adjust_for_ambient_noise(source, duration=1)

    while(1):
        with mic as source:
            print("Listening: \n")
            audio = r.listen(source)
            print("Analyzing: \n")
            try:
                message = r.recognize_google(audio)
                print(lp.pred(model, message))
            except sr.UnknownValueError:
                print("Unrecognizable")

speechToText()
