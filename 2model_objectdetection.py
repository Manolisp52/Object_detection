#εισαγωγή απαραίτητων βιβλιοθηκών
import numpy as np
import argparse
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk,Image
from tkinter import messagebox
import os
count1=0
count2=0
def object_det_MobileNetSSD():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", default=input_image)
	ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt.txt")
	ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel")
	ap.add_argument("-c", "--confidence", type=float, default=0.2)
	args = vars(ap.parse_args())

	# initialize the list of class labels MobileNet SSD was trained to
	# detect, then generate a set of bounding box colors for each class
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	# load our serialized model from disk
	print("[INFO] loading model...")
	model = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	# (note: normalization is done via the authors of the MobileNet SSD
	# implementation)
	image = cv2.imread(args["image"])
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	print("[INFO] computing object detections...")
	model.setInput(blob)
	detections = model.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the `detections`,
			# then compute the (x, y)-coordinates of the bounding box for
			# the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# display the prediction
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			print("[INFO] {}".format(label))
			cv2.rectangle(image, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(image, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# cv2.imwrite('image_detected.jpg',image)
	global imageSSD1
	global imageSSD2
	imageSSD = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	width, height = imageSSD.size
	if int(width)<int(height): imageSSD1 = imageSSD.resize((400, 552), Image.ANTIALIAS)
	else: imageSSD1 = imageSSD.resize((900, 555), Image.ANTIALIAS)
	if int(width)<int(height): imageSSD2 = imageSSD.resize((350, 450), Image.ANTIALIAS)
	else: imageSSD2 = imageSSD.resize((485, 310), Image.ANTIALIAS)
	return imageSSD1,imageSSD2
	
    
def object_det_Yolo():
    config_file="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    frozen_model="frozen_inference_graph.pb"
    model=cv2.dnn_DetectionModel(frozen_model,config_file)
    Labels=[]
    file_name="labels.txt"
    with open(file_name,"rt") as fpt:
        Labels= fpt.read().rstrip("\n").split("\n") ## "διαβάζει" την λίστα Labels με τα ονόματα των αντικειμένων που θα αναγνωρίζονται

    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5,127.5,127.5))
    model.setInputSwapRB(True)

    def image_detection():     ## "διαβάζει" μια εικόνα και την επιστρέφει έχοντας κάνει αναγνώριση
        img=cv2.imread(input_image) #"διαβάζει" την εικόνα
        ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
        for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
            cv2.rectangle(img,boxes,color=(0,255,0),thickness=2)
            cv2.putText(img,Labels[ClassInd-1].upper(),(boxes[0]+10,boxes[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,100,0),2)
        # cv2.imwrite('image_detected.jpg',img) #αποθηκέυει την εικόνα με την αναγνώριση
        global imageYolo1
        global imageYolo2
        imageYolo = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        width, height = imageYolo.size
        if int(width)<int(height): imageYolo1 = imageYolo.resize((400, 552), Image.ANTIALIAS)
        else: imageYolo1 = imageYolo.resize((900, 555), Image.ANTIALIAS)
        if int(width)<int(height): imageYolo2 = imageYolo.resize((350, 450), Image.ANTIALIAS)
        else: imageYolo2 = imageYolo.resize((485, 310), Image.ANTIALIAS)
        return imageYolo1,imageYolo2
        
    image_detection()

def model_compare():
    object_det_MobileNetSSD()
    object_det_Yolo()
    img1 = ImageTk.PhotoImage(imageYolo2)
    panel1.config(image=img1)
    panel1.image=img1
    img2=ImageTk.PhotoImage(imageSSD2)
    panel2.config(image=img2)
    panel2.image=img2
    panel.config(image="")
    panel.image=""



def check_checkbox1() :
    global val1
    val1=is_checked1.get()
    if val1== 1:
        is_checked2.set(0)
        is_checked3.set(0)
    return val1

def check_checkbox2():
    global val2
    val2=is_checked2.get()
    if val2 == 1:
        is_checked1.set(0)
        is_checked3.set(0)
    return val2


def check_checkbox3():
    global val3
    val3=is_checked3.get()
    if val3==1:
            is_checked1.set(0)
            is_checked2.set(0)
    return val3	


def model_choice():
    check_checkbox1()
    check_checkbox2()
    check_checkbox3()
    if val1==1: 
        object_det_MobileNetSSD()
        img1 = ImageTk.PhotoImage(imageSSD1)
        panel.config(image=img1)
        panel.image=img1
        panel1.config(image="")
        panel1.image=""
        panel2.config(image="")
        panel2.image=""
        panel2.place(x=994,y=150)
        

    elif val2==1: 
        object_det_Yolo()
        img2 = ImageTk.PhotoImage(imageYolo1)
        panel.config(image=img2)
        panel.image=img2
        panel1.config(image="")
        panel1.image=""
        panel2.config(image="")
        panel2.image=""
        panel2.place(x=994,y=150)
    elif val3==1: 
        model_compare()
        panel2.place(x=500,y=150)
    else:
        messagebox.showerror("Error", "Δεν έχεις επιλέξει model.")

def showinstructions():
        messagebox.showinfo(title="Τρόποι χρήσης",message='''
======HOW TO USE THIS PROGRAM=======

1)Στο menu 'Detection Models' επιλέγετε το model το οποίο θα πραγματοποίησει την ανίχνευση της εικόνας που θα επιλέξετε.

Έπειτα πατήστε το κουμπί: 'Επιλογή Εικόνας' και επιλέξτε την εικόνα που θέλετε για αναγνώριση.

2)Στο menu 'Commands' υπάρχουν 2 εντολές.

  Με την εντολή 'Σύγκριση' παρουσιάζονται στο παράθυρο 
  τα αποτελέσματα και των 2 models για την ίδια εικόνα.

  Με την εντολή 'Reset' επανέρχεται το παράθυρο 
  στην αρχική του κατάσταση.

''')

def showhelp():
    messagebox.showinfo(title="Help",message='''
    ==========MODELS EXPLANATION==========

    1) YOLO: Μεγάλη ταχύτητα αναγνώρισης που φτάνει 
       τα 40-90 FPS.
       Ωστόσο, η ακρίβεια στην αναγνώριση συχνά μπορεί 
       να μην είναι ικανοποητική

    2) MobileNetSSD: Ταχύτητα αναγνώρισης που φτάνει 
       μεχρι 22-46 FPS.
       Συνήθως η ακρίβεια είναι καλύτερη από τα YOLO models.
    ''')

def reset():
    panel.config(image="")
    panel.image=""
    panel1.config(image="")
    panel1.image=""
    panel2.config(image="")
    panel2.image=""


def callback():
    global input_image
    input_image= filedialog.askopenfilename()
    model_choice()
    
root=tk.Tk()
root.title("Object Detection-Ομάδα 12")
root.resizable(False, False)
is_checked1 = tk.IntVar()
is_checked2 = tk.IntVar()
is_checked3 = tk.IntVar()
window_height = 670
window_width = 1000
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))
root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate-45, y_cordinate-45))
root.configure(background='grey')
menubar=tk.Menu(root)
selection=tk.Menu(menubar,tearoff=0)
commands=tk.Menu(menubar,tearoff=0)
menubar.add_cascade(label="Detection Models",menu=selection)
selection.add_checkbutton(label="MobileNetSSD",onvalue = 1, offvalue = 0, variable = is_checked1, command = check_checkbox1)
selection.add_checkbutton(label="YOLO",onvalue = 1, offvalue = 0, variable = is_checked2, command = check_checkbox2)
selection.add_separator()
selection.add_command(label="Help",command=showhelp)
menubar.add_cascade(label="Commands",menu=commands)
commands.add_checkbutton(label="Σύγκριση",onvalue = 1, offvalue = 0, variable = is_checked3, command = check_checkbox3)
commands.add_command(label="Reset",command=reset)
menubar.add_cascade(label="Οδηγίες",command=showinstructions)
root.config(menu=menubar)
w1=tk.Label(root,text="Αναγνώριση Αντικειμένων",font = "Impact 36", bg ='lightgray', width = 900, borderwidth=4, relief="solid")
w1.pack(fill="x")
panel = tk.Label(root,bg="grey")
panel.pack()
panel1=tk.Label(root,bg="grey")
panel1.pack()
panel1.place(x=10,y=150)
panel2=tk.Label(root,bg="grey")
panel2.pack(anchor="w")
panel2.place(x=500,y=150)
button=tk.Button(root,text="Επιλογή Εικόνας",font = "Impact 15", fg = "lightgray", highlightbackground="lightgray", bg ="black",command=callback)
button.pack()
button.place(x=390,y=625)
root.mainloop()

