#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import cv2
import numpy as np
import tensorflow as tf
import math
from collections import deque
from sort import *

# In[3]:


import sys
import joblib


# In[4]:


#import adafruit_dht
import time
import socket
import threading
import traceback
import netifaces as ni
from udp import FrameSegment
from dateutil import tz
import datetime

#send to API
import mysql.connector
import requests

# In[5]:


#ret = cap.set(3,IM_WIDTH)
#ret = cap.set(4,IM_HEIGHT)
fgbg = cv2.createBackgroundSubtractorMOG2()


# In[6]:


bg = fgbg.getBackgroundImage()



# In[7]:

def withinRect(pts, vtx1, vtx2):
    if pts[0] >vtx1[0] and pts[0] <vtx2[0] and pts[1] >vtx1[1] and pts[1] <vtx2[1]:
        return True
    else:
        return False

    
class VideoInputThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        print('VideoThread Init!')
    def run(self):
        print('VideoThread Start!')
        
        global restart
        global getVideo
        global trigger
        global count1
        global count2
        global count3
        global vtx11
        global vtx12
        global vtx21
        global vtx22
        global vtx31
        global vtx32
        ids1 = deque(maxlen=20)
        ids2 = deque(maxlen=20)
        ids3 = deque(maxlen=20)
        
        IM_WIDTH = 640
        IM_HEIGHT = 480  
        #cap = cv2.VideoCapture("rtsp://192.168.1.88:554/1/h264major/user=admin&password=admin&channel=1&stream=0.sdp?")
        ###
        #You take your own risk to use the codes blow to capture image data.
        ###
        while True:
            cap = cv2.VideoCapture("rtsp://192.168.1.18:554/1/h264major/user=admin&password=admin&channel=1&stream=0.sdp?")
            ret, inputframe = cap.read()
            if ret:
                break
        
        fgbg = cv2.createBackgroundSubtractorMOG2()
        kernelOp = np.ones((5,5),np.uint8)
        kernelCl = np.ones((25,25),np.uint8)
        
        tracker1 = Sort()
        tracker2 = Sort()
        tracker3 = Sort()
        
        while True:
            if restart == True:
                break
            ret, bgframe = cap.read()
            bgframe = cv2.resize(bgframe, (IM_WIDTH, IM_HEIGHT))
            inputframe = bgframe.copy()
            from_zone = tz.gettz('UTC')
            date_time = datetime.datetime.now(tz.gettz('Ameria/LosAngeles'))
            #print(date_time)
            ts = date_time.replace(tzinfo=from_zone).timestamp()
            if ret == True:
                getVideo = True
            
            trigger = False
            
            bg = fgbg.getBackgroundImage()
            
            fgmask = fgbg.apply(bgframe)
            ret1,imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            mask0 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
            mask = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, kernelCl)
            #cv2.imshow("mask",mask)
            #if cv2.waitKey(10) & 0xFF == ord('q'):
            #    break
            contours0, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            
            detections1 = []
            detections2 = []
            detections3 = []
            
            
            if len(contours0)!=0:
                for cnt in contours0:
                    #cv2.drawContours(frame, cnt, -1, (0,255,0), 3, 8)
                    area = cv2.contourArea(cnt)
                    areaTH = 1000
                    if area > areaTH:          
                        M = cv2.moments(cnt)
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        x,y,w,h = cv2.boundingRect(cnt)
                        cv2.rectangle(bgframe,(x,y),(x+w,y+h),[0,255,0],2)
                        
                        center = [x+w/2,y+h/2]

                        if withinRect(center, vtx11, vtx12):
                            #print(videoInputQueue)
                            trigger = True
                            now=time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
                            detections1.append([x,y,x+w,y+h,0.6])
                            

                        if withinRect(center, vtx21, vtx22):
                            #print(videoInputQueue)
                            trigger = True
                            now=time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
                            detections2.append([x,y,x+w,y+h,0.6])
                            

                        if withinRect(center, vtx31, vtx32):
                            #print(videoInputQueue)
                            trigger = True
                            now=time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
                            detections3.append([x,y,x+w,y+h,0.6])
                            
                        if trigger:
                            bgQueue.append(bg)
                            videoInputQueue.append(inputframe)
                            time_list.append(ts)
                            if len(videoInputQueue) > 1:
                                videoInputQueue.pop(0)
                            if len(bgQueue) > 1:
                                bgQueue.pop(0)
                            if len(time_list) > 1:
                                time_list.pop(0)
                            time.sleep(0.04)



            else:
                now=time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
                #detections.append([x,y,x+w,y+h,0.6])
                bgQueue.append(bg)
                videoInputQueue.append(inputframe)
                time_list.append(ts)
                if len(videoInputQueue) > 1:
                    videoInputQueue.pop(0)
                if len(bgQueue) > 1:
                    bgQueue.pop(0)
                if len(time_list) > 1:
                    time_list.pop(0)
                time.sleep(0.05)
            
            dets1 = np.asarray(detections1)
            dets2 = np.asarray(detections2)
            dets3 = np.asarray(detections2)
            #print('dets:',dets)
            if len(dets1) == 0:
                dets1 = np.empty((0, 5))
            if len(dets2) == 0:
                dets2 = np.empty((0, 5))
            if len(dets3) == 0:
                dets3 = np.empty((0, 5))


            tracks1 = tracker1.update(dets1)
            tracks2 = tracker2.update(dets2)
            tracks3 = tracker3.update(dets3)
            #print('tracks:',tracks)
            
            for track in tracks1:
                if track[4] not in ids1:
                    count1 += 1
                    ids1.append(track[4])

            for track in tracks2:
                if track[4] not in ids2:
                    count2 += 1
                    ids2.append(track[4])

            for track in tracks3:
                if track[4] not in ids3:
                    count3 += 1
                    ids3.append(track[4])
            
            #cv2.rectangle(bgframe, (vtx11[0], vtx11[1]), (vtx12[0], vtx12[1]), (0,255,0), thickness=3)
            #cv2.rectangle(bgframe, (vtx21[0], vtx21[1]), (vtx22[0], vtx22[1]), (255,0,0), thickness=3)
            #cv2.rectangle(bgframe, (vtx31[0], vtx31[1]), (vtx32[0], vtx32[1]), (0,0,255), thickness=3)

            #cv2.putText(bgframe,'Count:'+str(count1),(vtx11[0], vtx11[1]-10),font,1,(0,255,0),2,cv2.LINE_AA)
            #cv2.putText(bgframe,'Count:'+str(count2),(vtx21[0], vtx21[1]-10),font,1,(255,0,0),2,cv2.LINE_AA)
            #cv2.putText(bgframe,'Count:'+str(count3),(vtx31[0], vtx31[1]-10),font,1,(0,0,255),2,cv2.LINE_AA)

            
            
            #cv2.rectangle(bgframe, (vtx1[0], vtx1[1]), (vtx3[0], vtx3[1]), (0,255,0), thickness=3)
            #cv2.putText(bgframe,'Count:'+str(count),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
            #cv2.imshow('image',bgframe)
            #print("queue length:",len(bgQueue))
            #if cv2.waitKey(5) & 0xFF == ord('q'):
            #    break
            cv2.waitKey(5)
        
videoInputQueue = []
bgQueue = []
time_list = []
restart = False
getVideo = False
trigger = False
frame = None
count1 = 0
count2 = 0
count3 = 0
vtx11 = [100,187]
vtx12 = [299,442]
vtx21 = [431,180]
vtx22 = [598,309]
vtx31 = [244,120]
vtx32 = [313,183]

# API Requests

class dbConnect():
    def __init__(self):
        self.mydb = mysql.connector.connect(user='root',
        password='StarLabPass1!',host='127.0.0.1',database='aiwaysion')
        self.mycursor = self.mydb.cursor(buffered=True)
    def finish(self):
        self.mydb.commit()
        self.mycursor.close()

def toEnv(date, hum, temp, rdCon, speed, device_id):
    url = "https://api.staging.aiwaysion.com/v1/remote/device/message"
    data = {"device_id": device_id, "date": date, "hum": hum, "temp": temp, "rdCon": rdCon, "speed": speed}
    myResponse = requests.post(url , data=data)
    if(myResponse.ok):
        print('Success to api')
    else:
        curr = datetime.strptime(date, '%Y-%m-%d-%H_%M_%S')
        database = dbConnect()
        stmt = (
            "INSERT INTO device_envs(device_id, temperature, humidity, road_con, traffic_count, traffic_speed, created_at, updated_at) \
            SELECT %s, %s, %s, %s,flow, %s, NOW(), NOW() FROM device_counts where ID = (select MAX(ID) as maxid from device_counts WHERE device_id = %s)"
        )
        data = (device_id, temp, hum, rdCon, speed, device_id)
        database.mycursor.execute(stmt, data)
        database.finish()
        print('Success to db')

def toCount(upstream, downstream, device_id):
    url = "https://api.staging.aiwaysion.com/v1/remote/device/count"
    data = {"device_id": device_id, "upstream": upstream, "downstream": downstream}
    myResponse = requests.post(url , data=data)
    print('Success to api Count')


def toImage(path, device_id):
    url = "https://api.staging.aiwaysion.com/v1/remote/device/image"
    data = {"device_id": device_id, "image": path}
    myResponse = requests.post(url , data=data)
    if(myResponse.ok):
        print('Success to api')
    else:
        database = dbConnect()
        insert_stmt = (
            "INSERT INTO device_images(device_id, image, created_at, updated_at) VALUES (%s, %s,  NOW(), NOW());"
        )
        data = (device_id, path)
        database.mycursor.execute(insert_stmt, data)
        database.finish()
        print('Success to db')








# In[9]:


sys.path.append('..')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    #print([h, w])
    imsz = h * w
    #print(imsz)
    numpx = int(max(math.floor(imsz / 1000), 1))
    #print(numpx)
    darkvec = dark.reshape(imsz)
    #print(darkvec)
    imvec = im.reshape(imsz, 3)
    #print(imvec)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95;
    im3 = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz);
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r));
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r));
    cov_Ip = mean_Ip - mean_I * mean_p;

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r));
    var_I = mean_II - mean_I * mean_I;

    a = cov_Ip / (var_I + eps);
    b = mean_p - a * mean_I;

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r));
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r));

    q = mean_a * im + mean_b;
    return q;


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray) / 255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray, et, r, eps)
    k = t.tolist()
    #print(min(k))

    return t;


def Recover(im, t, A, tx=0.1):
    #print(im.shape);
    res = np.empty(im.shape, im.dtype);
    t = cv2.max(t, tx);
    #print("t is:", t)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res

# In[10]:


MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')
NUM_CLASSES = 90


# In[11]:


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef() 
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')



# In[12]:


frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX




# In[13]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

fileConf = open('config.txt','r')
lines = fileConf.readlines()
addr = ''
port = -1
for line in lines:
    if line[0:4] == 'addr':
        addr = line[5:].strip()
    elif line[0:4] == 'port':
        port = int(line[5:].strip())
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(addr, port)
fs = FrameSegment(s, port=port, addr=addr)
fs2 = FrameSegment(s, port= 1246, addr="54.219.161.172")
# In[14]:


if __name__ == '__main__':
    videoInputThread = VideoInputThread()
    videoInputThread.start()
    #RF=joblib.load('rf.model')
    #print(RF)
    time_send_imgs1 = time.time()
    time_send_imgs2 = time.time()
    RF = joblib.load('RF_model2.sav')
    
    while(True):
        time.sleep(1)
        t1 = cv2.getTickCount()
        try:
            frame = videoInputQueue[0]
            pure_frame = frame
            
        
            #print(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_expanded = np.expand_dims(frame_rgb, axis=0)
            
        
            ts = time_list[0]
            
            
            now = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d-%H_%M_%S')
            udpstr=now+ ','
        
            bg = bgQueue[0]
            gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            m = np.median(gray)
            
            b,g,r = cv2.split(bg)
            dark_channel = cv2.min(cv2.min(r,g),b)
            d = np.median(dark_channel)
        
            dirs = [x[0] for x in os.walk('/home/pi/program/log')]
            max_folder = 0
            for item in dirs:
                dir_list = item.split('/')
                last = dir_list[-1]
                if last != 'log':
                    if int(last) > max_folder:
                        max_folder = int(last)
            dir_ht = '/home/pi/program/log/' + str(max_folder) + '/airTemp'
            #print(dir_ht)
            fileConf1 = open(dir_ht, 'r')
            lines1 = fileConf1.readlines()
            h = float(lines1[-1].split(',')[0].split(' ')[1])
            temp = float(lines1[-1].split(',')[1].split(' ')[1])
            #print('h'+ str(h))

    # Visibility
            src = np.copy(frame)

            I = src.astype('float64') / 255;

            dark = DarkChannel(I, 15);
            A = AtmLight(I, dark);
            te = TransmissionEstimate(I, A, 15);
            t = TransmissionRefine(frame, te);
            J = Recover(I, t, A, 0.1)
            J = J.astype('float64')*255
            #J = cv2.resize(J, (640, 480) ,interpolation = cv2.INTER_AREA);
            #cv2.imshow("J", J);
            #cv2.waitKey();
            

    # Road prediction 
            udpstr = ''
            if temp == None or h == None:
                print("No reading for temperature or humidity")
                temp = 'NA'
                h = 'NA'
            else:
                y_pred = RF.predict([[temp,h,m,d]])
            udpstr = udpstr + str(now) + ',' + str(temp) + ',' + str(h) + ','  + str(y_pred) + ','
            #print(udpstr)
            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: frame_expanded})
            
            #print(boxes)
            #print('boxes:'+str(boxes[0][0:numindex])+'prob:'+str(scores[0][0:numindex])+' classes:'+str(classes[0][0:numindex])) 
            
            numindex = 0
            for i in range(len(boxes[0])):
                if boxes[0][i][0] == 0.0 and boxes[0][i][1] == 0.0 and boxes[0][i][2] == 0.0:
                    numindex = i
                    break
            udpstr = udpstr + str(scores[0][0:numindex]) + ',' + str(classes[0][0:numindex]) + ',' + str(boxes[0][0:numindex])
            #print(udpstr)
            
            
            #if time.time() - time_send_imgs > 30:
            #    
            #    #time_send_imgs = time.time()
            #    print('send!')
            #    fs.udp_frame(pure_frame)
            #    time.sleep(0.5)
            
    # API Requests
            device_id = "232" 
            speed = "45" 
            toEnv(now, h, temp, y_pred, speed, device_id)          
           
    # Bounding Box on frame                       
                            
            fs.udp_event(bytes(udpstr, 'utf-8'))
            fs2.udp_event(bytes(udpstr, 'utf-8'))
            #pure_frame = frame
            '''
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3,
                min_score_thresh=0.5)
            '''

                #cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
            
            #print("time1", time.time())
            #print("time2", time_send_imgs)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            s.sendto(udpstr.encode(), ("54.219.161.172", 2245))
            if time.time() - time_send_imgs1 > 30:
                pts_down = np.array([[215,135], [570,330],[600,263], [280,120]],np.int32)
                #pts = pts.reshape((-1,1,2))
                cv2.polylines(frame, [pts_down], True, (0,0,255),thickness=3)
                cv2.putText(frame, 'DownStream', (420,170),font, 0.7, (0,0,255), 2, cv2.LINE_AA)

                pts_up = np.array([[210,140],[100,200], [325,460], [565,335]],np.int32)
                cv2.polylines(frame, [pts_up], True, (0,255,0),thickness=3)
                cv2.putText(frame, 'UpStream', (120,400),font, 0.7, (0,255,0), 2, cv2.LINE_AA)

                #cv2.rectangle(frame, (vtx11[0], vtx11[1]), (vtx12[0], vtx12[1]), (0,255,0), thickness=2)
                #cv2.rectangle(frame, (vtx21[0], vtx21[1]), (vtx22[0], vtx22[1]), (255,0,0), thickness=2)
                #cv2.rectangle(frame, (vtx31[0], vtx31[1]), (vtx32[0], vtx32[1]), (0,0,255), thickness=2)

                #cv2.putText(frame,'Count1:'+str(count1),(vtx11[0], vtx11[1]-10),font,0.7,(0,255,0),2,cv2.LINE_AA)
                #cv2.putText(frame,'Count2:'+str(count2),(vtx21[0], vtx21[1]-10),font,0.7,(255,0,0),2,cv2.LINE_AA)
                #cv2.putText(frame,'Count3:'+str(count3),(vtx31[0], vtx31[1]-10),font,0.7,(0,0,255),2,cv2.LINE_AA)

                count_result = "Upstream:" + str(count1) + "," + "Downstream:" + str(count2+count3)

                print(count_result)

                #s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
                #s.sendto(count_result.encode(), ("54.219.161.172", 1248))

                #count1 = 0
                #count2 = 0
                #count3 = 0
                
                #tracker1 = Sort()
                #tracker2 = Sort()
                #tracker3 = Sort()
                
                time_send_imgs1 = time.time()
                
                #print('send!')
                fs.udp_frame(frame)
                fs2.udp_frame(J)
                
                #time.sleep(0.5)
                #fs.udp_frame(pure_frame)
                print('send!')
            
            if time.time() - time_send_imgs2 > 900:

                count_result = "Upstream:" + str(count1) + "," + "Downstream:" + str(count2+count3)
                upstream = str(count1)
                downstream = str(count2+count3)
                
                print(count_result)

    # API Count
                device_id = "2814f8dc-4506-4b49-892a-948e8d6da29f"
                toCount(upstream, downstream, device_id)
                print(count_result)

                s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
                s.sendto(count_result.encode(), ("54.219.161.172", 2246))

                count1 = 0
                count2 = 0
                count3 = 0
                
                tracker1 = Sort()
                tracker2 = Sort()
                tracker3 = Sort()
                
                time_send_imgs2 = time.time()

                
                #time.sleep(0.5)
                #fs.udp_frame(pure_frame)
                print('count send!')
            
            
            #cv2.imshow('Object detector', frame)
            #cv2.imshow('bg', bg)
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc = 1/time1

            if cv2.waitKey(1) == ord('q'):
                break
                time.sleep(4)
                
            #cap.release()
            cv2.destroyAllWindows()
        except:
            continue
            
            
            



