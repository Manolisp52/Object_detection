#εισαγωγή απαραίτητων βιβλιοθηκών
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk,Image
from tkinter import messagebox
import filetype

def object_det_MobileNetSSD(): 

    modelConfiguration = "MobileNetSSD_deploy.prototxt.txt"
    modelWeights = "MobileNetSSD_deploy.caffemodel"

    #Δημιουργία λίστας με τα ονόματα των αντικειμένων που μπορεί να ανανγωρίσει το model
    #και δημιουργία μιας ομάδας χρωμάτων για κάθε "κουτί" αντικειμένου
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    #Φόρτωση του model
    print("[INFO] loading model...")
    model = cv2.dnn.readNetFromCaffe(modelConfiguration, modelWeights)

    #φόρτωση της εικόνας του χρήστη και δημιουργία ενός blob(δες τέλος κώδικα) για την εικόνα
    image = cv2.imread(input_image)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    
    #περνάει το blob με την εικόνα στο model και εξάγονται οι αναγνωρίσεις και οι προβλέψεις
    print("[INFO] computing object detections...")
    model.setInput(blob)
    detections = model.forward()
    
    for i in np.arange(0, detections.shape[2]):
        #εξαγωγή του confidence(πιθανότητα εγκυρότητας της αναγνώρισης) για κάθε πρόβλεψη
        confidence = detections[0, 0, i, 2]
    
        #παράβλεψη αδύναμων προβλέψεων κάνοντας σίγουρο ότι το confidence είναι
        #μεγαλύτερο από το ελάχιστο
        if confidence > 0.2:
            #εξαγωγή της θέσης του αντικειμένου(index) στην λίστα CLASSES από το "detections"
            #και υπολογισμός των συντεταγμένων των κουτιών που θα περιβάλλουν τα
            #αναγνωρισμένα αντικείμενα
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #εμφάνιση της πρόβλεψης
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),#δημιουργία πλαισίου γύρω από το αντικείμενο
				COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), #εισαγωγή του ονόματος του αντικειμένου στο πλαίσιο
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 1)
    
    global imageSSD1#η εικόνα που χρησιμοποιείται όταν επιλέγεται το MobileNetSSD
    global imageSSD2#η εικόνα που χρησιμοποιείται όταν επιλέγεται η Σύγκριση
    global imageSSD#η εικόνα που αποθηκεύεται όταν επιλέγεται το MobileNetSSD
    imageSSD = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))#μετατροπή από numpy.ndarrey σε PIL.Image.Image ώστε να μπορεί να επεξεργαστεί
    width, height = imageSSD.size
    #αλλαγή μεγέθους των αρχικών εικόνων ανάλογα με τις αρχικές τους διαστάσεις 
    #ώστε να μπορούν να εμφανιστούν στο παράθυρο
    if int(width)<int(height): imageSSD1 = imageSSD.resize((400, 552), Image.ANTIALIAS)
    else: imageSSD1 = imageSSD.resize((900, 552), Image.ANTIALIAS)
    if int(width)<int(height): imageSSD2 = imageSSD.resize((350, 450), Image.ANTIALIAS)
    else: imageSSD2 = imageSSD.resize((485, 310), Image.ANTIALIAS)
    return imageSSD1,imageSSD2
	
    
def object_det_Yolo():
    whT = 320
    confThreshold =0.5 #Προβλέψεις με πιθανότητα εγκυρότητας κάτω απο 50% θα παραβλέπονται
    nmsThreshold= 0.2
    classesFile = "coco.names"
    classNames = []
    #Δημιουργείται λίστα με τα κλάσεις των αντικειμένων από το "coco.names"
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
       
    modelConfiguration = "yolov3.cfg"
    modelWeights = "yolov3.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
         
    def findObjects(outputs,img):
        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                #παράβλεψη αδύναμων προβλέψεων κάνοντας σίγουρο ότι το confidence είναι
                #μεγαλύτερο από το ελάχιστο(0.5)
                if confidence > confThreshold:
                    w,h = int(det[2]*wT) , int(det[3]*hT)
                    x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                    bbox.append([x,y,w,h])
                    classIds.append(classId)
                    confs.append(float(confidence))
         
        indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
         
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)#δημιουργία πλαισίου γύρω από το αντικείμενο
            cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', #εισαγωγή του ονόματος του αντικειμένου στο πλαίσιο
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 0)

    img = cv2.imread(input_image) #"διαβάζει την εικόνα του χρήστη"      
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False) #δημιουργία ενός blob(δες τέλος κώδικα) για την εικόνα
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img) #κλήση της συνάρτησης που κάνει την αναγνώριση των αντικειμένων
    
    global imageYolo1#η εικόνα που χρησιμοποιείται όταν επιλέγεται το Yolo
    global imageYolo2#η εικόνα που χρησιμοποιείται όταν επιλέγεται η Σύγκριση
    global imageYolo#η εικόνα που αποθηκεύεται όταν επιλέγεται το Yolo
    imageYolo = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))#μετατροπή από numpy.ndarrey σε PIL.Image.Image ώστε να μπορεί να επεξεργαστεί
    width, height = imageYolo.size
    #αλλαγή μεγέθους των αρχικών εικόνων ανάλογα με τις αρχικές τους διαστάσεις 
    #ώστε να μπορούν να εμφανιστούν στο παράθυρο
    if int(width)<int(height): imageYolo1 = imageYolo.resize((400, 552), Image.ANTIALIAS)
    else: imageYolo1 = imageYolo.resize((900, 552), Image.ANTIALIAS)
    if int(width)<int(height): imageYolo2 = imageYolo.resize((350, 450), Image.ANTIALIAS)
    else: imageYolo2 = imageYolo.resize((485, 310), Image.ANTIALIAS)
    return imageYolo1,imageYolo2

def model_compare(): #Σύγκριση των 2 models
    object_det_MobileNetSSD()
    object_det_Yolo()
    img1 = ImageTk.PhotoImage(imageYolo2)
    panel1.config(image=img1)
    panel1.image=img1
    img2=ImageTk.PhotoImage(imageSSD2)
    panel2.config(image=img2)
    panel2.image=img2
    #θέτουμε την εικόνα του label, στο οποίο εμφανίζεται η αναγνωρισμένη εικόνα ενός model 
    #όταν επιλέγεται μόνο αυτό, ως κενή έτσι ώστε αν πριν ο χρήστης επιλέξει την σύγκριση
    #είχε επιλέξει κάποιο model, να μην εμφανίζεται η εικόνα που υπήρχε από την προηγούμενη αναγνώριση
    panel.config(image="")
    panel.image=""



def check_MobileNetSSD() :
    global val1
    val1=is_checked1.get()#παίρνουμε την τιμή της μεταβλητής is_checked1
    #(0 αν δεν είναι τικαρισμένο το MobileNetSSD και 1 αν είναι τικαρισμένο)
    if val1== 1:
        #αν είναι τικαρισμένο το MobileNetSSD, τα υπόλοιπα ξετικάρονται 
        is_checked2.set(0)
        is_checked3.set(0)
        button1.place(x=390,y=625)#το κουμπι της επιλογής εικόνας επανέρχεται στην αρχική του θέση
        button2.place(x=590,y=900)#το κουμπί της αποθήκευσης εικόνας τοποθετείται εκτός παραθύρου
    return val1

def check_YOLO():
    global val2
    val2=is_checked2.get()
    if val2 == 1:
        #αν είναι τικαρισμένο το Yolo, τα υπόλοιπα ξετικάρονται
        is_checked1.set(0)
        is_checked3.set(0)
        button1.place(x=390,y=625)#το κουμπι της επιλογής εικόνας επανέρχεται στην αρχική του θέση
        button2.place(x=590,y=900)#το κουμπί της αποθήκευσης εικόνας τοποθετείται εκτός παραθύρου
    return val2


def check_compare():
    global val3
    val3=is_checked3.get()
    if val3==1:
        #αν είναι τικαρισμένη η Σύγκριση, τα υπόλοιπα ξετικάρονται
            is_checked1.set(0)
            is_checked2.set(0)
            button1.place(x=390,y=625)#το κουμπι της επιλογής εικόνας επανέρχεται στην αρχική του θέση
            button2.place(x=590,y=900)#το κουμπί της αποθήκευσης εικόνας τοποθετείται εκτός παραθύρου
    return val3	


def model_choice():
        try:
            if not filetype.is_image(input_image): #αν το αρχείο που θα επιλέξει ο χρήστης δεν είναι εικόνα,
                                                   #εμφανίζεται μήνυμα σφάλματος
                    messagebox.showerror("Error","Δεν υποστηρίζεται αυτό το format αρχείου.")
            check_MobileNetSSD()
            check_YOLO()
            check_compare()
            if val1==1: 
                object_det_MobileNetSSD()
                img1 = ImageTk.PhotoImage(imageSSD1)
                #Εμφανίζεται η αναγνωρισμένη εικόνα στο panel,επαναφέρονται τα υπόλοιπα label στην αρχική μορφή 
                #και τοποθετούνται σε σημείο εκτός της επιφάνειας της εικόνας που εμφανίζεται
                panel.config(image=img1)
                panel.image=img1
                panel1.config(image="")
                panel1.image=""
                panel2.config(image="")
                panel2.image=""
                panel2.place(x=994,y=150)
                title1.config(text="")
                title1.place(x=15,y=100)
                title2.config(text="")
                title2.place(x=990,y=100)
                button1.place(x=290,y=625)#τοποθετείται το κουμπί της επιλογής εικόνας πιο αριστερά
                button2.place(x=590,y=625)#τοποθετείται το κουμπί της αποθήκευσης μέσα στο παράθυρο
                
            elif val2==1:
                object_det_Yolo()
                img2 = ImageTk.PhotoImage(imageYolo1)
                #Εμφανίζεται η αναγνωρισμένη εικόνα στο panel,επαναφέρονται τα υπόλοιπα label στην αρχική μορφή 
                #και τοποθετούνται σε σημείο εκτός της επιφάνειας της εικόνας που εμφανίζεται
                panel.config(image=img2)
                panel.image=img2
                panel1.config(image="")
                panel1.image=""
                panel2.config(image="")
                panel2.image=""
                panel2.place(x=994,y=150)
                title1.config(text="")
                title1.place(x=15,y=100)
                title2.config(text="")
                title2.place(x=990,y=100)
                button1.place(x=290,y=625)#τοποθετείται το κουμπί της επιλογής εικόνας πιο αριστερά
                button2.place(x=590,y=625)#τοποθετείται το κουμπί της αποθήκευσης μέσα στο παράθυρο

            elif val3==1:
                model_compare()
                panel2.place(x=500,y=150)
                #Εμφανίζεται το ονόμα του κάθε model πάνω από την αντίστοιχη εικόνα
                title1.config(text="YOLO")
                title1.place(x=220,y=100)
                title2.config(text="MobileNetSSD")
                title2.place(x=650,y=100)
                button1.place(x=290,y=625)#τοποθετείται το κουμπί της επιλογής εικόνας πιο αριστερά
                button2.place(x=490,y=625)#τοποθετείται το κουμπί της αποθήκευσης μέσα στο παράθυρο

            elif filetype.is_image(input_image):#αν ο χρήστης επιλέξει εικόνα χωρίς να έχει επιλέξει model,
                                                #εμφανίζεται μήνυμα σφάλματος
                    messagebox.showerror("Error", "Δεν έχεις επιλέξει model.")
        except:pass

def showinstructions():#οι οδηγίες χρήσης του προγράμματος
        messagebox.showinfo(title="Τρόποι χρήσης",message='''
    ======HOW TO USE THIS PROGRAM=======

    1)Στο menu 'Detection Models' επιλέγετε το model το οποίο        θα πραγματοποίησει την ανίχνευση της εικόνας 
       που θα επιλέξετε.

    Έπειτα πατήστε το κουμπί: 'Επιλογή Εικόνας' και επιλέξτε 
    την εικόνα που θέλετε για αναγνώριση.

    Αφού εμφανιστεί η αναγνωρισμένη εικόνα μπορείτε να την     αποθηκεύσετε πατώντας το κουμπί 'Αποθήκευση Εικόνας'.

    2)Στο menu 'Λειτουργίες' υπάρχουν 3 εντολές.

    Με την εντολή 'Σύγκριση' παρουσιάζονται στο παράθυρο 
    τα αποτελέσματα και των 2 models για την ίδια εικόνα.

    Με την εντολή 'Reset' επανέρχεται το παράθυρο 
    στην αρχική του κατάσταση.

    Με την εντολή 'Exit' πραγματοποείται έξοδος από το 
    πρόγραμμα.

    ''')

def showhelp():#κάποιες πληροφορίες για τα 2 models του προγράμματος
    messagebox.showinfo(title="Πληροφορίες",message='''
    ==========MODELS EXPLANATION==========

    1) YOLO: 
       Το μοντέλο YOLO αναγνωρίζει περισσότερα αντικείμενα 
       ίδιας κλάσης στην ίδια εικόνα σε σχέση με το 
       MobileNetSSD.


    2) MobileNetSSD: 
       Προορίζεται για κινητές συσκευές κι έτσι υστερεί 
       συνήθως στην ακρίβεια σε σχέση με τα YOLO 
       models.
    ''')

def reset():#επαναφέρει το παράθυρο στην αρχική του κατάσταση
    panel.config(image="")
    panel.image=""
    panel1.config(image="")
    panel1.image=""
    panel2.config(image="")
    panel2.image=""
    title1.config(text="")
    title2.config(text="")
    button2.place(x=590,y=900)
    button1.place(x=390,y=625)


def callback():#ανοίγει παράθυρο για επιλογή εικόνας όταν ο χρήστης πατήσει το κουμπί "Επιλογή Εικόνας"
    global input_image
    input_image= filedialog.askopenfilename(title="Επιλέξτε εικόνα παρακαλώ")
    model_choice()

def saveimage():#ανοίγει παράθυρο για αποθύήκευση της αναγνωρισμένης εικόνας όταν ο χρήστης πατήσει το κουμπί "Αποθήκευση εικόνας"
    filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")#αποθηκεύεται η αναγνωρισμένη εικόνα σε αρχείο.jpg
    if not filename: 
        return 
    if val1==1:  imageSSD.save(filename)
    elif val2==1: imageYolo.save(filename)
    elif val3==1:
        image1=imageYolo2
        image1_size=image1.size
        new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
        new_image.paste(image1,(0,0))
        new_image.paste(imageSSD2,(image1_size[0],0))
        new_image.save(filename) 

root=tk.Tk()
root.title("Object Detection-Ομάδα 12")
root.resizable(False, False)
window_height = 670 #ύψος παραθύρου
window_width = 1000 #πλάτος παραθύρου
#Λαμβάνονται τα όρια της οθόνης του χρήστη
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))
#Τοποθετείται το παράθυρο στο κέντρο της οθόνης του χρήστη
root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate-45, y_cordinate-45))
root.configure(background='grey')
is_checked1 = tk.IntVar()
is_checked2 = tk.IntVar()
is_checked3 = tk.IntVar()
menubar=tk.Menu(root)#δημιουργία μενού
selection=tk.Menu(menubar,tearoff=0)
commands=tk.Menu(menubar,tearoff=0)
menubar.add_cascade(label="Detection Models",menu=selection)
selection.add_checkbutton(label="MobileNetSSD",onvalue = 1, offvalue = 0, variable = is_checked1, 
command = check_MobileNetSSD)
selection.add_checkbutton(label="YOLO",onvalue = 1, offvalue = 0, variable = is_checked2, 
command = check_YOLO)
selection.add_separator()#προστίθεται μια γραμμή ανάμεσα στην προηγούμενη και την επόμενη επιλογή
selection.add_command(label="Πληροφορίες",command=showhelp)
selection.add_separator()
menubar.add_cascade(label="Λειτουργίες",menu=commands)
commands.add_checkbutton(label="Σύγκριση",onvalue = 1, offvalue = 0, variable = is_checked3, command = check_compare)
commands.add_command(label="Επαναφορά",command=reset)
commands.add_separator()
commands.add_command(label="Exit",command=root.destroy)
commands.add_separator()
menubar.add_cascade(label="Οδηγίες",command=showinstructions)
root.config(menu=menubar)
#Δημιουργία των labels στα οποία θα εμφανίζονται οι αναγνωρισμένες εικόνες
w1=tk.Label(root,text="Αναγνώριση Αντικειμένων",font = "Impact 36", bg ='lightgray', width = 900, borderwidth=4, relief="solid")
w1.pack(fill="x")
panel = tk.Label(root,bg="grey") #το label που θα εμφανίζονται οι εικόνες όταν επιλέγεται συγκεκριμένο model
panel.pack()
panel1=tk.Label(root,bg="grey")#το label που θα εμφανίζεται η αριστερή εικόνα όταν επιλέγεται η σύγκριση
panel1.pack()
panel1.place(x=10,y=150)
panel2=tk.Label(root,bg="grey")#το label που θα εμφανίζεται η δεξιά εικόνα όταν επιλέγεται η σύγκριση
panel2.pack()
panel2.place(x=500,y=150)
#Δημιουργία των τίτλων των models για την αντίστοιχη εικόνα στην Σύγκριση 
title1=tk.Label(root,bg="grey",font = "Impact 25")
title1.pack()
title1.place(x=220,y=100)
title2=tk.Label(root,bg="grey",font="Impact 25")
title2.pack()
title2.place(x=650,y=100)
#Δημιουργία των κουμπιών
button1=tk.Button(root,text="Επιλογή Εικόνας",font = "Impact 15", fg = "lightgray", 
highlightbackground="lightgray", bg ="black",command=callback)
button1.pack()
button1.place(x=390,y=625)
button2=tk.Button(root,text="Αποθήκευση Εικόνας",font = "Impact 15", fg = "lightgray", 
highlightbackground="lightgray", bg ="black",command=saveimage)
button2.pack()
button2.place(x=590,y=900)#Τοποθέτηση εκτός παραθύρου
root.mainloop()

##blob= Binary Large OBject.Το blob είναι ένας τύπος δεδομένων που μπορεί να αποθηκεύσει δεδομένα σε δυαδική μορφή.
#Είναι διαφορετικό όμως από τα περισσότερους τύπους δεδομένων (όπως integers, floating point numbers, characters, and strings),που αποθηκεύουν γράμματα και αριθμούς.
#Ουσιαστικά το BLOB χρησιμοποείται για αποθήκευση αρχείων όπως εικόνες,βίντεο και αρχεία ήχου, καθώς παρέχει μεγάλες ταχύτητες μεταφορές αρχείων πολυμέσων.