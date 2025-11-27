import cv2, random, numpy as np
from augmentations.base import GeoAug

class FlipHAug(GeoAug):
    name = "flip_h"
    def apply(self, img, boxes):
        h, w = img.shape[:2]
        new_boxes = [(cls, [w-xmax, ymin, w-xmin, ymax]) for cls,(xmin,ymin,xmax,ymax) in boxes]
        return cv2.flip(img, 1), new_boxes

class FlipVAug(GeoAug):
    name = "flip_v"
    def apply(self, img, boxes):
        h, w = img.shape[:2]
        new_boxes = [(cls, [xmin, h-ymax, xmax, h-ymin]) for cls,(xmin,ymin,xmax,ymax) in boxes]
        return cv2.flip(img, 0), new_boxes

class RotateAug(GeoAug):
    name = "rotate"
    def apply(self, img, boxes):
        h, w = img.shape[:2]
        angle = random.uniform(-30,30)
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1)
        out = cv2.warpAffine(img, M, (w,h), borderValue=(0,0,0))
        new_boxes = [(cls, self.transform_bbox(bbox, M, w, h)) for cls,bbox in boxes]
        return out, new_boxes

class ShearAug(GeoAug):
    name = "shear"
    def apply(self, img, boxes):
        h, w = img.shape[:2]
        factor = random.uniform(-0.12,0.12)
        pts1 = np.float32([[0,0],[w,0],[0,h]])
        pts2 = np.float32([[0,0],[w,h*factor],[w*factor,h]])
        M = cv2.getAffineTransform(pts1, pts2)
        out = cv2.warpAffine(img, M, (w,h), borderValue=(0,0,0))
        new_boxes = [(cls, self.transform_bbox(bbox, M, w, h)) for cls,bbox in boxes]
        return out, new_boxes

class CropAug(GeoAug):
    name = "crop"
    def apply(self, img, boxes):
        h, w = img.shape[:2]
        ratio = random.uniform(0.4,0.9)
        new_h, new_w = int(h*ratio), int(w*ratio)
        top = random.randint(0,h-new_h)
        left = random.randint(0,w-new_w)
        cropped = img[top:top+new_h, left:left+new_w]
        new_boxes = []
        for cls,(xmin,ymin,xmax,ymax) in boxes:
            xmin, xmax = xmin-left, xmax-left
            ymin, ymax = ymin-top, ymax-top
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(new_w, xmax), min(new_h, ymax)
            if xmax-xmin>3 and ymax-ymin>3:
                new_boxes.append((cls,[xmin,ymin,xmax,ymax]))
        return cropped, new_boxes
