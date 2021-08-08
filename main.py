import cv2
import numpy as np 
import pandas as pd
from datetime import datetime
import  pyttsx3



def personDetector(image):
    cou=0
    global lx,ly,bordercolor
    
    Width = image.shape[1]
    Height = image.shape[0]

        
    net1.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))
        
    person_layer_names = net1.getLayerNames()
    person_output_layers = [person_layer_names[i[0] - 1] for i in net1.getUnconnectedOutLayers()]
    person_outs = net1.forward(person_output_layers)
        
    person_class_ids = []
    person_confidences = []
    person_boxes = []

    for operson in person_outs:
        for detection in operson:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                person_class_ids.append(class_id)
                person_confidences.append(float(confidence))
                person_boxes.append([x, y, w, h])

    pindex = cv2.dnn.NMSBoxes(person_boxes, person_confidences, 0.5, 0.4)

    for i in pindex:
        i = i[0]
        box = person_boxes[i]
        lx=round(box[0]+box[2]/2)
        ly=round(box[1]+box[3])-10
        if person_class_ids[i]==0:
            label = str(coco_classes[person_class_ids[i]])
            cou+=1
            #cou%4==0 for exactly 3 people tailgating
            if cou >= 2:
                bordercolor = (0,0,255)
                df.loc[inde,"Tailgated"] = "YES"
                
                print("ALERT!! TAILGATING DETECTED")
                 
            cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), bordercolor, 2)
            cv2.putText(image, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, bordercolor, 2)
            cv2.circle(image, (lx,ly), radius=5, color=(255, 0, 0), thickness=-1)


def doorDetector(image):
   

    confThresholold = 0.25
    nsmThresholold = 0.40
    inpWidth = 416
    inpHeight = 416
    
    classesFile = "H:/Custome-YOLO-with-custome-dataset/obj.names" # add extension of .names file for custom dataset 
    classes = None

    with open(classesFile,'rt') as f :
        classes =f.read().rstrip('\n').split('\n')
    
    
    blob = cv2.dnn.blobFromImage(image, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)
    
    modelConf='H:/Custome-YOLO-with-custome-dataset/yolo-obj.cfg' # add extension of configuration file for custom YOLO model
    modelWeights='H:/Custome-YOLO-with-custome-dataset/yolo-obj.weights' # add extension of weights file for custom YOLO model
    net = cv2.dnn.readNetFromDarknet(modelConf,modelWeights)
    
    net.setInput(blob)
    

    
   # outs = net.forward(getOutputsNames(net))
    

   

    frameHeight=image.shape[0]
    frameWidth=image.shape[1]


    classIDs = []
    confidences = []
    boxes = []


     # Get the names of all the layers in the network
    layersNames = net.getLayerNames()

        # Get the names of the output layers, i.e. the layers with unconnected outputs
    outs= [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    door_outs = net.forward(outs)
    
    for out in door_outs:
        for detection in out:
            scores = detection[5:]
            classID =np.argmax(scores)
            confidence =scores[classID]

            if confidence > confThresholold:
                centerX = int(detection[0]*frameWidth)
                centerY = int(detection[1]*frameWidth)

                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)

                left = int(centerX-width/2)
                top = int(centerY-height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes (boxes,confidences, confThresholold, nsmThresholold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        right=left+width
        bottom=top+height
        classId=classIDs[i]
        
       
        #label ='%.2f' %cconf
        label='%s' % ("")
        if classes[classIDs[i]]!="handle":
            cv2.rectangle(image, (left, top), (right, bottom), (128,0,128), 2)
            assert(classId<len(classes))
            label='%s' % ("door")

        cv2.putText(image, label,(left,top),cv2.FONT_HERSHEY_SIMPLEX, 1, (128,0,128), 2)
        engine = pyttsx3.init()
        engine.say("door")
        engine.runAndWait()
    

cap = cv2.VideoCapture('tailgate.mp4')
df = pd.read_csv('access_info.csv')
now = datetime.now()

coco_classes = None
with open('labels.txt', 'r') as f:
    coco_classes = [line.strip() for line in f.readlines()]


net1 = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
#net2 = cv2.dnn.readNet("yolov3_door.weights", "yolov3_door.cfg")
#frames per second of the video.
fps = 1
#final coordinates of the bounding box 
xxx,yyy,www,hhh = 0,0,0,0

lx=0
ly=0

cou=-1
#green border of the box for detected object
bordercolor = (0,255,0)

id = int(input("ENTER EMPLOYEE ID  "))
print(' ')
#find if the employee id is present in csv file
emp_id = df[df["Employee ID"] == id]
inde = emp_id.index[0]

try:

    if(emp_id["Authorisation Result"].item() == 1):
        df.loc[inde,"Time of Swipe"] = now.strftime("%d/%m/%Y %H:%M:%S")

        while True:

            cap.set(cv2.CAP_PROP_POS_FRAMES, fps)
            _, image =cap.read()

            if fps < 30:
             
                doorDetector(image)
                
            
            personDetector(image)
            
            
            cv2.line(image, (xxx,yyy+hhh),(xxx+www+20,yyy+(hhh-40)), bordercolor, thickness=3)
            cv2.line(image, (xxx,yyy+hhh),(xxx,yyy+round(hhh/2)), bordercolor, thickness=3)
            cv2.line(image, (xxx+www+20,yyy+(hhh-40)),(xxx+www+10,yyy+round(hhh/2)), bordercolor, thickness=3)


            t1 = (lx - xxx)*((yyy+(hhh-40)) - (yyy+hhh))
            t2 = (ly - (yyy+hhh))*((xxx+www+20) - xxx)
            d = t1 - t2

            if d>0:
                cou+=1

            if cou >= 2:
                bordercolor = (0,0,255)
                df.loc[inde,"Tailgated"] = "YES"
                
                print("ALERT!! TAILGATING DETECTED")
                
            
            fps = fps + 5
            cv2.imshow("Result",image)
            key = cv2.waitKey(1)
            if key == 27:
                break


    else:
        print("INVALID ID\n")


except AttributeError:
    
    print("\nEND of VIDEO")
    

except Exception as e:
    print(e)


df.to_csv('access_info.csv',index=False)
cap.release()
cv2.destroyAllWindows()
