import pyautogui
import time
time.sleep(3)
count=0
while count<=10:
    pyautogui.typewrite("Hey bro! What's up! ")
    pyautogui.press("enter")
    count=count+1
