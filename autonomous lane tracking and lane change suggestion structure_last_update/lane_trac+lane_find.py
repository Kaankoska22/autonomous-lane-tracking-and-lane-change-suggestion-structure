# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:24:58 2021

@author: kaan koska
"""

"""
# -*- coding: utf-8 -*-
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:37:15 2021

@author: kaan koska
"""
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 17:39:33 2021

@author: kaan koska
"""


########################################kütüphaneler

import cv2

import numpy as np

from tracker import *

import matplotlib.pyplot as plt

#########################################

tracker = EuclideanDistTracker()#tracker scriptnden trakc fonksiyonunun alınması
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)#kameradan alınan görüntünün gereksiz arka planının çıkarılması

#########################################
###############################################################fonksiyonlar


def interested_area_for_lane_find(frame):#şerit takibi için belirlenenözel alan   
    mask = np.zeros_like(frame)#uygulanıcak maske
    triangle = np.array([[#maskenin özellikleri konumu yükseklik ve genişliği
    (400, 310),
    (735, 70),
    (1300, 310),]], np.int32)
    cv2.fillPoly(mask, triangle, 255)#maskenin yani belirlenmiş özel bölgenin triangle ile birleştirilmesi
    masked_image = cv2.bitwise_and(frame, mask)#maskenin frame üzerine oturtulması
    return masked_image


def interested_area_for_right_lane_find(frame): # sağ şerit çizgileri için özel alan maskesinin oluşturulması  
    mask = np.zeros_like(frame)#uygulanıcak maske
    triangle = np.array([[#maskenin özellikleri konumu yükseklik ve genişliği
    (900, 310),
    (850, 200),
    (1000, 200),
    (1200, 310),]], np.int32)
    cv2.fillPoly(mask, triangle, (0,0,255))#maskenin yani belirlenmiş özel bölgenin triangle ile birleştirilmesi
    masked_image = cv2.bitwise_and(frame, mask)#maskenin frame üzerine oturtulması
    return masked_image


def interested_area_for_left_lane_find(frame):#sol şerit çizgileri için özel alan maskesinin oluşturulması   
    mask = np.zeros_like(frame)#uygulanıcak maske
    triangle = np.array([[#maskenin özellikleri konumu yükseklik ve genişliği
    (450, 310),
    (550, 200),
    (700, 200),
    (700, 310),]], np.int32)
    cv2.fillPoly(mask, triangle, (0,0,255))#maskenin yani belirlenmiş özel bölgenin triangle ile birleştirilmesi
    masked_image = cv2.bitwise_and(frame, mask)#maskenin frame üzerine oturtulması
    return masked_image


def find_draw_coordinats_for_lines(image, line):##şeritlerin çizgilerinin koordinatlarının belirlenmesi
    start, finish = line   
    y1 = int(310)#çizimin başlıyacağı nokta
    y2 = int(186)#zimin biteceği nokta resimde framde ters
    x1 = int((y1 - finish)/start)#sol şerit 
    x2 = int((y2 - finish)/start)#sağ şerit
    return [[x1, y1, x2, y2]]#şeritlerin koordinatları
 
def average_lane_coordinats(image, lines):#koordinatları alınan çizgilerin optimize edilmesi daha doğru ve düzenli bi şerit çizimi için
    left_side    = []# sol şerit bilgileri
    right_side   = []#sağ şerit bilgileri
    if lines is None:# houghP ile bulduğumuz çizgileri attığımız lines dizisi boş değil ise yani bir çizgi varsa bu diziyi for döngüsünden geçirerek şerit bilgilerine ulaşıyoruz
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:              
            fit = np.polyfit((x1,x2), (y1,y2), 1)#dönen değerleri iki değer olarak fit dizisine atıyoruz 
            start = fit[0]#başlangıç
            finish = fit[1]#bitiş           
            if start < 0: #dönen slope değeri sıfırdan küçük ise çizgi bilgisi sol tarafa ait bu yüzden sol tarafa append ediyoruz
                left_side.append((start, finish))
            else:
                right_side.append((start, finish))

    if len(left_side) and len(right_side):#dönen bilgiler yani çizgi dizilei boş değilse girilen değerlerin ortalamasını bulup sağ ve sol şerit dizilerine atıyoruz
            left_fit_average  = np.average(left_side, axis=0)#dizinin horizontal ortalamasını bulmak için axis=0 çünkü şerit çizgileri dik ortalamada olmalı
            right_fit_average = np.average(right_side, axis=0)
            left_line  =  find_draw_coordinats_for_lines(image, left_fit_average)#bu dizilerdeki bilgileri koordinat haline çevirmek için make_points fonksiyonunu kullanıyoruz
            right_line =  find_draw_coordinats_for_lines(image, right_fit_average)
            averaged_lines = [left_line, right_line]#alınan koordinatları sağ ve sol şekline averaged_lines dizimize atıyoruz burda sağ ve sol şerit çizgilerinin koordinatları oluyor [[100,120],[200,220]]şeklinde çoklu bir dizi değeri döndürecektir           
            return averaged_lines

   
def show_lines(img,lines):#çizgilerin ekranda gösterilmesi 
    line_image = np.zeros_like(img)#çizim yapılıcak img boyutunda çizim yapılıcak resim
    ###########sağ sol ve düz yol işareti resimlerinin import edilmesi ve resize edilmesi
    right_image=cv2.imread("C:/Users/casper/Desktop/autonomous lane tracking and lane change suggestion structure/turn_right.png")
    #print(right_image.shape())
    smaller_right_image = cv2.resize(right_image,(300,300))
    left_image=cv2.imread("C:/Users/casper/Desktop/autonomous lane tracking and lane change suggestion structure/turn_left_image.png")
    smaller_left_image = cv2.resize(left_image,(300,300))
    straight_image=cv2.imread("C:/Users/casper/Desktop/autonomous lane tracking and lane change suggestion structure/straight.png")
    smaller_straight_image = cv2.resize(straight_image,(300,300))
    overlay=np.zeros((line_image.shape[0],line_image.shape[1],3),dtype="uint8")# resimleri gösterebilmek için overlay tanımlıyoruz ana resimle aynı boyutta       
    if lines is not None:
        for line in lines:#lines içinde gezinerek şerit koordinatlarını alıyorum
            for x1, y1, x2, y2 in line:                
                a=np.array(lines)
                left_bottom_corner=a[0,0,0]
                left_up_corner=a[0,0,2]
                right_bottom_corner=a[1,0,0]
                right_up_corner=a[1,0,2]                
                curve=815-left_bottom_corner-215#şeridin dönüş eğiminin hesaplanması
                curvesafe=815-left_bottom_corner-215 #aracın görüntüsünden seçilen bölümün sürüş yapılabilir alanının hesaplanması             
                safe=right_bottom_corner-left_bottom_corner+120 #iki şerit arasındaki güvenli bölgenin hesaplanamsı               
                triangle = np.array([[#kırmızı güvenli sürüş alanı köşe koordinatları
                (left_bottom_corner+5, y1),
                (left_up_corner+5, y2),
                (right_up_corner, y2),
                (right_bottom_corner, y1),]], np.int32)                
                triangle_error= np.array([[#kırmızı güvensiz sürüş alanı köşe koordinatları
                (520, 310),
                (790, 166),
                (1200, 310),]], np.int32)                
               
                print(curvesafe)
                
        
                if safe>100 and safe <1000:
                    cv2.fillPoly(line_image, triangle, (255,0,0))
                    cv2.line(line_image,(x1,y1),(x2,y2),(0,120,0),15)#şeritlerin üzerine 
                    cv2.circle(line_image,(x1,y1), 20, (255,255,255),-1)#tespit edilen köşelerin nokta ile işaretlenmesi
                    cv2.circle(line_image,(x2,y2), 20, (255,255,255), -1)
########################################################################################
                if curvesafe<0 or curvesafe>550:#850
                    cv2.fillPoly(line_image, triangle_error, (0,0,255))
                    cv2.putText(line_image,"no-drive area" ,(700,230),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 3, cv2.LINE_AA)
########################################################################################                
                if curvesafe>0 and curvesafe<550:                   
                    cv2.putText(line_image, 'Safe Driving Area',(680,220),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0), 3, cv2.LINE_AA)
########################################################################################                    
                if curve>80:
                    overlay[0:smaller_right_image.shape[0],0:smaller_right_image.shape[1]]=smaller_right_image
                    if  curvesafe<0 or curvesafe>190:
                        cv2.putText(line_image,"right" ,(left_bottom_corner+300,270),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 3, cv2.LINE_AA)
                        cv2.putText(line_image,"car location" ,(left_bottom_corner+100,270),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(line_image,"of the drive safe area lane" ,(left_bottom_corner+50,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 1, cv2.LINE_AA) 
                    else:
                        cv2.putText(line_image,"right" ,(left_bottom_corner+300,270),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255), 3, cv2.LINE_AA)
                        cv2.putText(line_image,"car location" ,(left_bottom_corner+100,270),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(line_image,"of the drive safe area lane" ,(left_bottom_corner+50,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 1, cv2.LINE_AA) 
########################################################################################                
                if curve<80 and curve>30:
                    overlay[0:smaller_straight_image.shape[0],0:smaller_straight_image.shape[1]]=smaller_straight_image 
                    if  curvesafe<0 or curvesafe>190:
                        cv2.putText(line_image,"center" ,(left_bottom_corner+300,270),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 3, cv2.LINE_AA)
                        cv2.putText(line_image,"car location" ,(left_bottom_corner+100,270),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(line_image,"of the drive safe area lane" ,(left_bottom_corner+50,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 1, cv2.LINE_AA) 
                    else:
                        cv2.putText(line_image,"center" ,(left_bottom_corner+300,270),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 3, cv2.LINE_AA)
                        cv2.putText(line_image,"car location" ,(left_bottom_corner+100,270),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(line_image,"of the drive safe area lane" ,(left_bottom_corner+50,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 1, cv2.LINE_AA) 
########################################################################################                
                if curve<10:
                    overlay[0:smaller_left_image.shape[0],0:smaller_left_image.shape[1]]=smaller_left_image 
                    if curvesafe<0 or curvesafe>190:
                        cv2.putText(line_image,"Left" ,(left_bottom_corner+300,270),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 3, cv2.LINE_AA)
                        cv2.putText(line_image,"car location" ,(left_bottom_corner+100,270),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(line_image,"of the drive safe area lane" ,(left_bottom_corner+50,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 1, cv2.LINE_AA) 
                    else:
                        cv2.putText(line_image,"Left" ,(left_bottom_corner+300,270),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255), 3, cv2.LINE_AA)
                        cv2.putText(line_image,"car location" ,(left_bottom_corner+100,270),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(line_image,"of the drive safe area lane" ,(left_bottom_corner+50,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 1, cv2.LINE_AA) 
########################################################################################
                              
    combo_image1 = cv2.addWeighted(line_image, 1, overlay, 1, 0)                  
    return combo_image1
######################################################################şerit çizgilerinin çizilmesi



cap = cv2.VideoCapture("C:/Users/casper/Desktop/autonomous lane tracking and lane change suggestion structure/test_video_Trim.mp4")#test vidosunun yolu
while(cap.isOpened()):# her frame i tek tek okuyabilmek için while döngümü kuruyorum 
    ret, frame = cap.read()
    frame = cv2.resize(frame,(1480,582))
    if ret == True:                
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#beyaz şeritleri yakalamak için frame i hsv türüne dönüştürüyorum
        lower = np.array([0,0,158])#beya lower ve upper bilgileri
        upper = np.array([179,255,255])
        maskf = cv2.inRange(hsv,lower,upper)#hsv kodlarıma ait olan maskeyi oluşturuyorum        
        edges = cv2.Canny(maskf,20,100)#olaşan maske içinde kalan çizgileri buluyorum
        kernel = 5
        blur = cv2.GaussianBlur(edges,(kernel, kernel),0)# daha iyi bi sonuç için görüntümü yumuşatıyorum                      
        cropped_canny = interested_area_for_lane_find(blur)#fonksiyondan özel alanımı alıyorum                 
        lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=50,maxLineGap=30)#özel alan içinde olan çizgilerimi houghp ile buluyorum
        averaged_lines = average_lane_coordinats(frame, lines)#ortalama şerit çizgilerimin koordinatlarını alıyorum                
        line_image = show_lines(frame, averaged_lines)#çigilerimi frame üzerinde gösteriyorum
        combo_image3 = cv2.addWeighted(frame, 0.8, line_image, 1, 1)     
        
################################################################################
################################################################################
################################################################################
#####################################################lane change decide
        roi_right =interested_area_for_right_lane_find(frame) #sağ ve sol şerit çizgilerim için özel alanlarımı belirliyorum
        roi_left =interested_area_for_left_lane_find(frame) 
        hsv_right = cv2.cvtColor(roi_right,cv2.COLOR_BGR2HSV)#bu alanları renk lower upper tespiti için hvs ye dönüştürüyorum
        hsv_left = cv2.cvtColor(roi_left,cv2.COLOR_BGR2HSV)        
        lower_white =  np.array([0,0,158])
        upper_white = np.array([179,255,255])    
        mask_right = cv2.inRange(hsv_right,lower_white,upper_white)#masklerime değerlerini atıyorum
        mask_left = cv2.inRange(hsv_left,lower_white,upper_white)
#############################################################################sol şerit tespiti ve karar yapısı       
        mask = object_detector.apply(mask_left)#background çıkarma işlemi yapıyorum
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)#treshold uyguluyarak 0 ve 1 yani siyah ve beyaz keskinliklerini ayırıyorum ve ayırmak istediğim renk kod aralığını giriyorum yani beyaz
        contours, _ = cv2.findContours(mask_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#belirlenen alan üzerindeki köşelri buluyorum
        detections_left = []
        for cnt in contours:#counturlara rectangel çiziyorum
            area = cv2.contourArea(cnt)            
            if area > 100 and area<2000: #çizgi uzunluğuna göre kesik şerit mi düz şerit mi olduğunu anlıyoruz
                cv2.putText(frame,"lane change to left possible" ,(590,385),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 3, cv2.LINE_AA)
                x, y, w, h = cv2.boundingRect(cnt)
                detections_left.append([x, y, w, h])
            if area > 2000:
                cv2.putText(frame,"lane change to left impossible" ,(590,385),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255), 3, cv2.LINE_AA)
#########################################################şerit çizgi takibi
        boxes_ids = tracker.update(detections_left)# tespit edilen kesik şerit çizgisi takip ediliyor
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.rectangle(roi_left, (x, y), (x + w, y + h), (0, 255,0), 3)     
        combo_image1 = cv2.addWeighted(frame, 0.8, roi_left, 1, 1) 
############################################################################sol şerit takibi ve karar yapısı sonu
    
    
    
############################################################sağ şerit takibi ve karar yapısı
        mask = object_detector.apply(mask_right)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100 and area<2000:
                cv2.putText(combo_image1,"lane change to right possible" ,(590,355),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 3, cv2.LINE_AA)
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])
            if area > 2000:
                cv2.putText(combo_image1,"lane change to right impossible" ,(590,355),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255), 3, cv2.LINE_AA)           
###########################################################sağ şerit takibi
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.rectangle(roi_right, (x, y), (x + w, y + h), (0, 255, 0), 3)      
        combo_image = cv2.addWeighted(combo_image1, 0.8,roi_right, 1, 1)        
        combo_image4 = cv2.addWeighted(combo_image3, 1,combo_image, 1, 1)
#######################################################################sağ şerit takibi ve karar yapısı sonu ve iki farklı tespitin ana frame de birleştirilmesi

        
################################################################################
################################################################################
################################################################################
#####################################################endlane change decide
      
        cv2.imshow("frame",cropped_canny)######istenilen alan
        cv2.imshow("mask",combo_image4)#framein tamamı

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()