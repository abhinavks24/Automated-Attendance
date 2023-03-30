import tkinter as tk
from tkinter import Message , Text
import cv2 , os
import csv
import shutil
import numpy as np
from PIL import Image , ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font


# Creating the attendance window
window = tk.Tk()
window.title("Face Recognizer")
window.geometry('1280x720')
dialog_title = 'QUIT'
dialog_text = 'Are You sure ?'
window.configure(background='blue')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
#window.mainloop()

# Header for the window
message = tk.Label(window, text = "Facial Recognition Based Attendance System", bg = "Green", fg="White", width = 50, height = 3, font=('times',30,'italic bold underline'))
message.place(x=100,y=20)


# defining functions for the buttons

def clear():
    txt.delete(0,"end")
    res = ""
    message.configure(text=res)

def clear2():
    txt2.delete(0,"end")
    res = ""
    message.configure(text=res)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except(TypeError , ValueError):
        pass

    return False
    
def TakeImage():                        #function to rake multiple images of the user
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam=cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret,img = cam.read()
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray,1.3,5)
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)

                sampleNum = sampleNum+1       # incrementing the sample number for collecting more samples

                cv2.imwrite("Training Images\ "+ name+"." + Id + '.' +str(sampleNum) + ".jpg", gray[y:y+h,x:x+h])   # saving captured faces in folder Training Images

                cv2.imshow("Fame",img)     # display the frame
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                            break       # wait for 100 milli seconds or for the peress of button "q"  
                      
            elif sampleNum>60 :
                break                   # break if samples collected more than 60  
        cam.release()
        cv2.destroyAllWindows()
        res = "Images saved for ID : " + Id + " Name : " + name
        row = [Id,name]
        with open("Student Details\studentdetails.csv", 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if(name.isalpha()):
            res="Enter Numeric ID"
            message.configure(text=res)


def TrainImage():
    recognizer = cv2.face_LBPHFaceRecognizer.crete()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    face,Id = getImageAndLabel('Training Images')
    recognizer.train(faces,np.array(Id))
    recognizer.save("TrainingImageLabel\Trainer.yml")
    res = "Image Trained"    #+ ",".join(str(f) for f in Id)
    message.configure(text=res)


def getImageAndLabel(path):
    imagePaths = [os.path.join(path,f)for f in os.listir(path)]
    faces=[]
    Ids=[]
    for imagePath in imgePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage,'unit6')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNP)
        Ids.append(Id)
    return faces,Ids


def TrackImage():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read('TrainingImageLabel\Trainer.yml')
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascae = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("Student Details\studentdetails.csv")
    cam = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id','Name','Date','Time']
    attendance = pd.Dataframe(Columns = col_names)
    while(True):
        ret,img = cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if (conf<50):
                ts= time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp= datetime.datetime.fromtimestamp(ts).strftime('%H:%M,%S')
                aa = df.loc[df['Id'] ==Id]['Name'].values
                tt = str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [id,aa,date,timeStamp]
            else:
                Id='Unknown'
                tt=str(Id)
                if(conf>75):
                    noOfFile = len(os.listdir("ImagesUnknown")) +1
                    cv2.imwrite("ImagesUnknown\Image"+str(noOfFile)+".jpg", im[y:y+h,x:x+w])
                cv2.putText(im,str(tt),(x,y+h), font ,1,(255,255,255),2)
        attendance = attendancce.drop_duplicates(subset=['Id'],keep ='first')
        cv2.imshow('im',im)
        if(cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp= datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName = "Attendance\Attendance_" +date+"_"+Hour+"_"+Minute+"_"+Second+".csv"
    attendance.to_csv(sileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    res=attendance
    message2.configure(text=res)
    
        

#labels and buttons for the attendance window
lbl = tk.Label(window, text="Enter ID", width=20, height=2, fg="red",bg="yellow", font =("times",15,"bold"))
lbl.place(x=200,y=200)
txt = tk.Entry(window, width=20, bg="yellow", fg="red", font=("times",24,'bold'))
txt.place(x=550, y=210)


lbl2 = tk.Label(window, text="Enter Name", width=20, height=2, fg="red",bg="yellow", font =("times",15,"bold"))
lbl2.place(x=200,y=300)
txt2 = tk.Entry(window, width=20, bg="yellow", fg="red", font=("times",24,'bold'))
txt2.place(x=550, y=310)


lbl3 = tk.Label(window, text="Notification :", width=20, height=2, fg="red",bg="yellow", font =("times",15,"bold underline"))
lbl3.place(x=200,y=400)
message = tk.Label(window, text="", width=30, height=2,activebackground='yellow', fg="red",bg="yellow", font =("times",15,"bold"))
message.place(x=550,y=400)


lbl4 = tk.Label(window, text="Attendance :", width=20, height=2, fg="red",bg="yellow", font =("times",15,"bold underline"))
lbl4.place(x=200,y=620)
message2 = tk.Label(window, text="", width=30, height=2,activebackground='yellow', fg="red",bg="yellow", font =("times",15,"bold"))
message2.place(x=550,y=620)


clearbutton = tk.Button(window,text="Clear",command=clear , width=30, height=2,activebackground='red', fg="red",bg="yellow", font =("times",15,"bold"))
clearbutton.place(x=950,y=210)
clearbutton2 = tk.Button(window,text="Clear",command=clear2 , width=30, height=2,activebackground='red', fg="red",bg="yellow", font =("times",15,"bold"))
clearbutton2.place(x=950,y=310)


takeImg = tk.Button(window,text="Take Image",command=TakeImage , width=20, height=3,activebackground='red', fg="red",bg="yellow", font =("times",15,"bold"))
takeImg.place(x=90,y=500)

trainImg = tk.Button(window,text="Train Image",command=TrainImage , width=20, height=3,activebackground='red', fg="red",bg="yellow", font =("times",15,"bold"))
trainImg.place(x=390,y=500)

trackImg = tk.Button(window,text="Track Image",command=TrackImage , width=20, height=3,activebackground='red', fg="red",bg="yellow", font =("times",15,"bold"))
trackImg.place(x=690,y=500)

quitWindow = tk.Button(window,text="Quit",command=window.destroy, width=20, height=3,activebackground='red', fg="red",bg="yellow", font =("times",15,"bold"))
quitWindow.place(x=990,y=500)



        
    
