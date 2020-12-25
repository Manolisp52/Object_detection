#εισαγωγή απαραίτητων βιβλιοθηκών
import numpy as np
import argparse
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk,Image
from tkinter import messagebox
import os

def object_det_MobileNetSSD():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", default=img1)
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
	image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	width, height = image.size
	if int(width)<int(height): img = image.resize((400, 555), Image.ANTIALIAS)
	else: img = image.resize((900, 555), Image.ANTIALIAS)
	img = ImageTk.PhotoImage(img)
	panel.config(image=img)
	panel.image=img
    
def object_det_Yolo():
    config_file="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    frozen_model="frozen_inference_graph.pb"
    model=cv2.dnn_DetectionModel(frozen_model,config_file)
    Labels=[]
    file_name="labels.txt"
    with open(file_name,"rt") as fpt:
        Labels= fpt.read().rstrip("\n").split("\n") #δημιουργείται η λίστα Labels με τα ονόματα των αντικειμένων που θα αναγνωρίζονται

    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5,127.5,127.5))
    model.setInputSwapRB(True)

    def image_detection():     ## "διαβάζει" μια εικόνα και την επιστρέφει έχοντας κάνει αναγνώριση
        img=cv2.imread(img1) #"διαβάζει" την εικόνα
        ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
        for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
            cv2.rectangle(img,boxes,color=(0,255,0),thickness=2)
            cv2.putText(img,Labels[ClassInd-1].upper(),(boxes[0]+10,boxes[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,100,0),2)
        # cv2.imwrite('image_detected.jpg',img) #αποθηκέυει την εικόνα με την αναγνώριση
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        width, height = img.size
        if int(width)<int(height): img = img.resize((400, 555), Image.ANTIALIAS)
        else: img = img.resize((900, 555), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image=img
    image_detection()

def check_checkbox1() :
    global val1
    val1=is_checked1.get()
    if val1== 1:
        is_checked2.set(0)
    return val1

def check_checkbox2():
    global val2
    val2=is_checked2.get()
    if val2 == 1:
        is_checked1.set(0)
    return val2

def model_choice():
    check_checkbox1()
    check_checkbox2()
    if val1==1: object_det_MobileNetSSD()
    elif val2==1: object_det_Yolo()
    else:
        messagebox.showerror("Error", "Δεν έχεις επιλέξει model.")


def callback():
    global img1
    img1= filedialog.askopenfilename()
    model_choice()
    
root=tk.Tk()
root.title("Object Detection-Ομάδα 12")
root.resizable(False, False)
is_checked1 = tk.IntVar()
is_checked2 = tk.IntVar()
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
menubar.add_cascade(label="Detection Models",menu=selection)
selection.add_checkbutton(label="MobileNet",onvalue = 1, offvalue = 0, variable = is_checked1, command = check_checkbox1)
selection.add_checkbutton(label="YOLO",onvalue = 1, offvalue = 0, variable = is_checked2, command = check_checkbox2)
root.config(menu=menubar)
w1=tk.Label(root,text="Αναγνώριση Αντικειμένων",font="arial 40",bg="light blue")
w1.pack(fill="x")
panel = tk.Label(root,bg="grey")
panel.pack()
button=tk.Button(root,text="Επιλογή Εικόνας",font="arial 17",bg="grey",command=callback)
button.pack()
button.place(x=390,y=625)
root.mainloop()

