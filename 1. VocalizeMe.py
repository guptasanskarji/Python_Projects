

from tkinter import *
import pyttsx3

text2speech =Tk()
text2speech.title('VocalizeMe')
text2speech.geometry("400x200")


# Text to Speech function
def convertor():
    engine = pyttsx3.init()
    engine.say(userEntry.get())
    engine.runAndWait()
    userEntry.delete(0, END)


# User String
userEntry = Entry(text2speech, font=("Arial", 15))
userEntry.pack(pady=20, padx=80)

# Speck button
speckButton = Button(text2speech, text="Convert", command=convertor)
speckButton.pack(pady=20)



text2speech.mainloop()
