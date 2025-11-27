import cv2, numpy as np, random
from augmentations.base import PixelAug

class GrayAug(PixelAug):
    name = "gray"
    def augment(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

class RedAug(PixelAug):
    name = "red"
    def augment(self, img):
        out = img.copy(); out[:,:,2] = np.clip(out[:,:,2]*1.6,0,255); return out

class GreenAug(PixelAug):
    name = "green"
    def augment(self, img):
        out = img.copy(); out[:,:,1] = np.clip(out[:,:,1]*1.3,0,255); return out

class BlueAug(PixelAug):
    name = "blue"
    def augment(self, img):
        out = img.copy(); out[:,:,0] = np.clip(out[:,:,0]*1.3,0,255); return out

class NoiseAug(PixelAug):
    name = "noise"
    def augment(self, img):
        noise = np.random.normal(0,10,img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32)+noise,0,255).astype(np.uint8)

class BlurAug(PixelAug):
    name = "blur"
    def augment(self, img):
        return cv2.GaussianBlur(img,(5,5),0)

class LensFlareAug(PixelAug):
    name = "lensflare"
    def augment(self, img):
        img = img.copy(); h,w = img.shape[:2]; overlay=np.zeros_like(img,dtype=np.uint8)
        cx=random.randint(int(w*0.2),int(w*0.8)); cy=random.randint(0,int(h*0.4)); r=random.randint(80,150)
        color=(255,random.randint(220,255),random.randint(180,230))
        cv2.circle(overlay,(cx,cy),r,color,-1); overlay=cv2.GaussianBlur(overlay,(51,51),0)
        return cv2.addWeighted(img,1.0,overlay,0.6,0)

class RainAug(PixelAug):
    name="rain"
    def augment(self,img):
        rain=np.zeros_like(img)
        for _ in range(random.randint(100,150)):
            x=random.randint(0,img.shape[1]-1); y=random.randint(0,img.shape[0]-1)
            length=random.randint(10,20); cv2.line(rain,(x,y),(x,y+length),(200,200,200),1)
        return cv2.addWeighted(img,1,cv2.blur(rain,(3,3)),0.3,0)

class FogAug(PixelAug):
    name="fog"
    def augment(self,img):
        return cv2.addWeighted(img,0.7,np.full_like(img,255),0.3,0)

class GammaAug(PixelAug):
    name="gamma"
    def augment(self,img):
        gamma=1.3; invGamma=1.0/gamma
        table=np.array([(i/255)**invGamma*255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img,table)
