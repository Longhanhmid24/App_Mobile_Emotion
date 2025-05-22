from multiprocessing import Process
import subprocess
import sys

def run_api_image():
    subprocess.Popen([
        sys.executable, 
        "C:/Users/DELL/Desktop/model/App_Mobile_Emotion/training_model/App_Mobile/Back_End/API_Mobile_Images.py"
    ])

def run_api_webcam():
    subprocess.Popen([
        sys.executable,
        "C:/Users/DELL/Desktop/model/App_Mobile_Emotion/training_model/App_Mobile/Back_End/API_Moble_Webcam.py"
    ])

if __name__ == '__main__':
    p1 = Process(target=run_api_image)
    p2 = Process(target=run_api_webcam)

    p1.start()
    p2.start()

    p1.join()
    p2.join()