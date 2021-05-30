from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
#Motor kontrol
Motor1A = 23
Motor1B = 24
Motor2A = 27
Motor2B = 22
         
GPIO.setup(Motor1A,GPIO.OUT)
GPIO.setup(Motor1B,GPIO.OUT)
GPIO.setup(Motor2A,GPIO.OUT)
GPIO.setup(Motor2B,GPIO.OUT)
def hls_lthresh(img, thresh=(0, 90)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls_l = hls[:,:,1]
    hls_l = hls_l*(255/np.max(hls_l))
    binary_output = np.zeros_like(hls_l)
    binary_output[(hls_l <= thresh[1])] = 1
    return binary_output
def unwarp(img, src, dst):
    h,w = img.shape[:2]
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv 
def main():
    
    camera = PiCamera()
    camera.resolution =(320,240)
    camera.framerate =32
    rawCapture=PiRGBArray(camera,size=(320,240))
    
    time.sleep(0.1)
 
    for frame in camera.capture_continuous(rawCapture,format="bgr",use_video_port=True):
        image = frame.array
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imshow("ilk ekran",image)
        img2=image.copy()
        h,w = image.shape[:2]
        a=30
        src = np.float32([(85,180),(235,180),(30,230),(290,230)])
        dst = np.float32([(a,0),(w-a,0),(a,h),(w-a,h)])
        vertices =np.array([[(85,180),(30,230),(290,230),(235,180)]],np.int32)
        img2=cv2.fillPoly(img2, vertices, 255)
        cv2.imshow("Analiz",img2)
        exampleImg_unwarp, M, Minv = unwarp(image, src, dst)
        cv2.imshow("kus",exampleImg_unwarp)
        img_LThresh = hls_lthresh(exampleImg_unwarp)
        Blackline = cv2.inRange(img_LThresh,1,1)
        kernel = np.ones((3,3), np.uint8)
        Blackline = cv2.erode(Blackline, kernel, iterations=5)
        Blackline = cv2.dilate(Blackline, kernel, iterations=9)
        cv2.imshow("cizgi algilama",Blackline)
        
        img_blk,contours_blk, hierarchy_blk = cv2.findContours(Blackline.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours_blk) > 0:
                    
         blackbox = cv2.minAreaRect(contours_blk[0])
         (x_min, y_min), (w_min, h_min), ang = blackbox
         if -50>ang>-82:
             cv2.putText(exampleImg_unwarp,"sag",(240, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
             GPIO.output(Motor1A,GPIO.HIGH)
             GPIO.output(Motor1B,GPIO.HIGH)
             GPIO.output(Motor2A,GPIO.LOW)
             GPIO.output(Motor2B,GPIO.HIGH)
         elif -82>ang>-90 or 0>=ang>=-10:
             cv2.putText(exampleImg_unwarp,"ileri",(240, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
             GPIO.output(Motor1A,GPIO.HIGH)
             GPIO.output(Motor1B,GPIO.LOW)
             GPIO.output(Motor2A,GPIO.HIGH)
             GPIO.output(Motor2B,GPIO.HIGH)
         elif -10>ang>-50:
             cv2.putText(exampleImg_unwarp,"sol",(240, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
             GPIO.output(Motor1A,GPIO.HIGH)
             GPIO.output(Motor1B,GPIO.HIGH)
             GPIO.output(Motor2A,GPIO.HIGH)
             GPIO.output(Motor2B,GPIO.LOW)
         else:
             cv2.putText(exampleImg_unwarp,"geri",(240, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
             GPIO.output(Motor1A,GPIO.LOW)
             GPIO.output(Motor1B,GPIO.HIGH)
             GPIO.output(Motor2A,GPIO.HIGH)
             GPIO.output(Motor2B,GPIO.HIGH)
             
         setpoint = 160
         error = int(x_min - setpoint) 
         ang = int(ang)  
         box = cv2.boxPoints(blackbox)
         box = np.int0(box)
         cv2.drawContours(exampleImg_unwarp,[box],0,(0,0,255),3)     
         cv2.putText(exampleImg_unwarp,str(ang),(5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
         cv2.putText(exampleImg_unwarp,str(error),(5, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
         cv2.line(exampleImg_unwarp, (int(x_min),100 ), (int(x_min),125 ), (255,0,0),3)
        
           
        cv2.imshow("Kus2", exampleImg_unwarp)
 
        key=cv2.waitKey(1)&0xFF
 
        rawCapture.truncate(0)
 
        if key==ord("q"):
            cv2.destroyAllWindows()
            camera.close()
            break
 
if __name__ == "__main__":
    main()


