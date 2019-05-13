# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 22:55:59 2017

@author: GULSUM
"""

from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from skimage import io,color,feature,exposure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.widgets as widgets

import numpy,cv2,sys,scipy
from scipy import ndimage

from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
from skimage.segmentation import clear_border

from sklearn.metrics import mean_squared_error
from skimage.measure import structural_similarity as ssim

from skimage.transform import resize
import math,random
import os

from tasarim import Ui_Form

class MainWindow(QtGui.QMainWindow, Ui_Form):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)
        
        self.btn_ryukle.clicked.connect(self.image_load)
        self.btn_gray.clicked.connect(self.gray_resme_donustur)
        self.btn_binary.clicked.connect(self.binary_resme_donustur)
        self.btn_piksel.clicked.connect(self.pikselleriGoster)
        self.combo_rotate.currentIndexChanged.connect(self.rotateYap)   
        self.btn_resize.clicked.connect(self.resizeYap)
        self.combo_filter.currentIndexChanged.connect(self.filtrelemeYap)
        self.btn_hist_goster.clicked.connect(self.histogramGoster)
        self.btn_hist_esitle.clicked.connect(self.histogramEsitle)
        self.r_yukle_labelling.clicked.connect(self.image_load_labelling)
        self.btn_labelling.clicked.connect(self.labellingYap)
        self.combo_labelling.currentIndexChanged.connect(self.labelling_goster)
        self.btn_karemi.clicked.connect(self.kareKontrolu)
        self.r_yukle_mantik1.clicked.connect(self.image_load_mantik1)
        self.r_yukle_mantik2.clicked.connect(self.image_load_mantik2)
        self.combo_mantik.currentIndexChanged.connect(self.mantiksalIslemler)
        self.btn_benzerlik.clicked.connect(self.benzerlikBul)
        self.btn_hist_benzerlik.clicked.connect(self.hist_benzerlik)
        self.verticalSlider.valueChanged.connect(self.gurultuYukle)
        self.btn_gurultu.clicked.connect(self.gurultuOlustur)
        self.radio_eroison.clicked.connect(self.eroisonYap)
        self.radio_dilation.clicked.connect(self.dilationYap)
         
    def image_load(self):
        self.file_name=unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.png)"))       
        self.image_show(self.file_name,self.resim_goster1)
        
    def image_show(self,file_name,gosterme_yeri):
        w,h=gosterme_yeri.width()-5,gosterme_yeri.height()-5
        pixMap = QtGui.QPixmap(file_name) 
        pixMap=pixMap.scaled(w,h)            
        pixItem = QtGui.QGraphicsPixmapItem(pixMap)
        scene2 = QGraphicsScene()
        scene2.addItem(pixItem)       
        gosterme_yeri.setScene(scene2)
                
    def gray_resme_donustur(self):
        img = cv2.imread(self.file_name)
        self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        deg_res='./resimler/gray_image.png'
        cv2.imwrite(deg_res,self. gray_image)     
        self.image_show(deg_res,self.resim_goster2)
    
    def binary_resme_donustur(self):
        img=cv2.imread(self.file_name,0)
        ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        file_="./resimler/binary_resim.png"
        cv2.imwrite(file_,thresh)    
        self.image_show(file_,self.resim_goster2)
        
    def pikselleriGoster(self):      
        img=Image.open(self.file_name)
        piksel=img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                r,g,b=img.getpixel((i,j))
                L=r * 299/1000 + g * 587/1000 + b * 114/1000
                piksel[i,j]=(i,j,int(L))
                print piksel[i,j],L
                
    def rotateYap(self): 
        aci=int(self.combo_rotate.currentText()) 
        img=Image.open(self.file_name)
        A=img.rotate(aci)
        A.save("./resimler/rotate.png")
        path_r="./resimler/rotate.png"      
        self.image_show(path_r,self.resim_goster2)
        
    def resizeYap(self):
       img=cv2.imread(self.file_name)
       w,h=img.shape[:2]
       resize_img=cv2.resize(img,(int(self.txt_en.text())*w,int(self.txt_boy.text())*h),interpolation = cv2.INTER_CUBIC)
       path="./resimler/resize_resim.png"
       cv2.imwrite(path,resize_img)
       self.lbl_resize.setText("Resize Basarili")
       self.image_show(path,self.resim_goster2)     
               
    def filtrelemeYap(self):
        filtre=self.combo_filter.currentText()
        if filtre=="Sobel":
            im = scipy.misc.imread(self.file_name)
            im = im.astype('int32')
            dx = ndimage.sobel(im, 0)  # horizontal derivative
            dy = ndimage.sobel(im, 1)  # vertical derivative
            mag = numpy.hypot(dx, dy)  # magnitude
            mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
            scipy.misc.imsave('./resimler/sobel.png', mag)
            path='./resimler/sobel.jpg'
            self.image_show(path,self.resim_goster2)
        elif filtre=="Canny":
            img=cv2.imread(self.file_name)
            edge=cv2.Canny(img,100,200)
            yol="./resimler/canny_edge.png"
            cv2.imwrite(yol,edge)
            self.image_show(yol,self.resim_goster2)
        elif filtre=="Prewit":
            image=io.imread(self.file_name)
            prewit_edge=scipy.ndimage.filters.prewitt(image)
            f_path="./resimler/prewit_edge.png"
            cv2.imwrite(f_path,prewit_edge)
            self.image_show(f_path,self.resim_goster2)
            
    def histogramGoster(self):
        img=cv2.imread(self.file_name,0)
        hist,bins=np.histogram(img.flatten(),256,[0,256])
        cdf=hist.cumsum()
        cdf_normalized=cdf * hist.max()/cdf.max()
        plt.plot(cdf_normalized,color='b')
        plt.hist(img.flatten(),256,[0,256],color='r')
        plt.xlim([0,256])
        plt.legend(('cdf','histogram'),loc='upper left')
        path="./resimler/hist_grafik.png"
        plt.savefig(path)
        self.image_show(path,self.resim_goster2)
        
    def histogramEsitle(self):
        img=cv2.imread(self.file_name,0)
        equ=cv2.equalizeHist(img)
        pat="./resimler/hist_esitle.png"
        cv2.imwrite(pat,equ)
        self.image_show(pat,self.resim_goster3)
    
    def hist_benzerlik(self):
        image1=cv2.imread(self.file_name)
        image2=cv2.imread("./resimler/hist_esitle.png")
        
        image1=cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY) 
        
        ssim_benzer = ssim(image1, image2)
        self.label_2.setText(str(round(ssim_benzer,2)))
        
    def gurultuYukle(self):      
        self.lbl_gurultu.setText(str(self.verticalSlider.value()))
            
    def gurultuOlustur(self):
        image=cv2.imread(self.file_name)
        w,h=image.shape[:2]
        gurultu_orani=self.verticalSlider.value()
        piksel_toplami=w * h
        gurultulu_piksel_sayisi=(piksel_toplami) * gurultu_orani /100
        for i in range(gurultulu_piksel_sayisi):
            image[random.randint(0,w-1)][random.randint(0,h-1)]= [random.choice([0,255]),random.choice([0,255]),random.choice([0,255])]
        p="./resimler/gurultu_resim.png"
        cv2.imwrite(p,image)
        self.image_show(p,self.resim_goster2)        
                
    def benzerlikBul(self):
        image1=cv2.imread(self.file_name)
        image2=cv2.imread("./resimler/gurultu_resim.png")
        
        image1=cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        
        err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
        err /=(float(image1.shape[0] * image1.shape[1]))
        
        ssim_benzerlik = ssim(image1, image2)
        
        psnr=math.log10(255.0/math.sqrt(err))
        
        self.lbl1.setText(str(round(err,2)))
        self.lbl2.setText(str(round(ssim_benzerlik,2)))
        self.lbl3.setText(str(round(psnr,2)))
        
    def eroisonYap(self):
        image=cv2.imread(self.file_name,1)      
        kernel = np.ones((5,5), np.uint8)       
        img_erosion = cv2.erode(image, kernel, iterations=1)
        path="./resimler/eroison_image.png"
        cv2.imwrite(path,img_erosion)
        self.image_show(path,self.resim_goster2)
        
    def dilationYap(self):
        image=cv2.imread(self.file_name,1)      
        kernel = np.ones((5,5), np.uint8)       
        img_dilation = cv2.dilate(image, kernel, iterations=1)
        path="./resimler/dilation_image.png"
        cv2.imwrite(path,img_dilation)
        self.image_show(path,self.resim_goster2)
        
    def image_load_labelling(self):
        self.file_name=unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.png)"))       
        self.image_show(self.file_name,self.r_goster_labelling1)
            
    def labellingYap(self):
        image = color.rgb2gray(io.imread(self.file_name))
        cleared=clear_border(image)
        label_image = label(cleared)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(image,cmap=plt.cm.gray)
        for i,region in enumerate(regionprops(label_image)):
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor='red')    
            ax.add_patch(rect)       
            bolge=image[minr:maxr,minc:maxc]             
            io.imsave('./resimler/sekiller/reg'+str(i)+'.png',bolge)
            self.combo_labelling.addItem("reg"+str(i))
        file_n="./resimler/sekiller/labelling.png"
        plt.savefig(file_n)
        self.image_show(file_n,self.r_goster_labelling2)
        
    def kareKontrolu(self):
        deger=self.combo_labelling.currentText()
        path="./resimler/sekiller/"+deger+".png"
        image=Image.open(path)
        w,h=image.size        
        piksel=[]
        pikselMap=image.load()
        count1=0
        count2=0
        for i in range(w):
            for j in range(h):
                piksel.append(pikselMap[i,j])
        for i in range(len(piksel)):
            if piksel[i]==0:
                count1+=1
            elif piksel[i]==255: 
                count2+=1
        print count1,count2
        if count1==0 and count2!=0:
             self.lbl_kare_mi.setText("Bu sekil karedir")
        else: 
             self.lbl_kare_mi.setText("Bu sekil kare degildir")            
        
    def labelling_goster(self):
        deger=self.combo_labelling.currentText()
        self.lbl_kare_mi.setText(" ")
        path="./resimler/sekiller/"+deger+".png"
        self.image_show(path,self.r_goster_labelling3)
            
    def image_load_mantik1(self):
        self.imageA=unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.png)"))       
        self.image_show(self.imageA,self.r_goster_mantik1)
    def image_load_mantik2(self):
        self.imageB=unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.png)"))       
        self.image_show(self.imageB,self.r_goster_mantik2)
        
    def mantiksalIslemler(self):
        islem=self.combo_mantik.currentText()
        a=Image.open(self.imageA)
        b=Image.open(self.imageB)
        pixelMapA=a.load()
        pixelMapB=b.load()
        w,h=a.size
        
        if islem=="NOT":
            imageNew=Image.new(a.mode,a.size)
            pixelNew=imageNew.load()           
            for i in range(0,w):
                for j in range(0,h):
                    a,b,c=pixelMapA[i,j]
                    pixelNew[i,j]=255-a,255-b,255-c
            p1="./resimler/mantik/not_image.png"
            imageNew.save("./resimler/mantik/not_image.png")
            self.image_show(p1,self.r_goster_mantik3)
            
        elif islem=="AND":
            imageNew2=Image.new(a.mode,a.size)
            pixelNew2=imageNew2.load()            
            for i in range(0,w):
                for j in range(0,h):
                    a,b,c=pixelMapA[i,j]
                    d,e,f=pixelMapB[i,j]
                    pixelNew2[i,j]=a and d, b and e , c and f
            p2="./resimler/mantik/and_image.png"
            imageNew2.save("./resimler/mantik/and_image.png")
            self.image_show(p2,self.r_goster_mantik3)
            
        elif islem=="OR":
            imageNew3=Image.new(a.mode,a.size)
            pixelNew3=imageNew3.load()      
            for i in range(0,w):
                for j in range(0,h):
                    a,b,c=pixelMapA[i,j]
                    d,e,f=pixelMapB[i,j]
                    pixelNew3[i,j]=a or d, b or e , c or f
            p3="./resimler/mantik/or_image.png"
            imageNew3.save("./resimler/mantik/or_image.png")
            self.image_show(p3,self.r_goster_mantik3)
            
        elif islem=="XOR":
            imageNew4=Image.new(a.mode,a.size)
            pixelNew4=imageNew4.load()      
            for i in range(0,w):
                for j in range(0,h):
                    a,b,c=pixelMapA[i,j]
                    d,e,f=pixelMapB[i,j]
                    pixelNew4[i,j]=a^d, b^e,c^f

            p4="./resimler/mantik/xor_image.png"
            imageNew4.save("./resimler/mantik/xor_image.png")
            print "xor yapildi"
            self.image_show(p4,self.r_goster_mantik3)