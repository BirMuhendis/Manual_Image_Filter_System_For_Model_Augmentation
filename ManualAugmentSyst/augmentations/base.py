import numpy as np

class Augmentation:
    name = "base"
    def apply(self, img, boxes):
        raise NotImplementedError

class PixelAug(Augmentation):
    def apply(self, img, boxes):
        return self.augment(img), boxes
    def augment(self, img):
        raise NotImplementedError

class GeoAug(Augmentation):
    def transform_bbox(self, bbox, M, w, h):
        xmin, ymin, xmax, ymax = bbox
        pts = np.array([[xmin,ymin,1],[xmax,ymin,1],[xmax,ymax,1],[xmin,ymax,1]])
        transformed = (M @ pts.T).T
        xs, ys = transformed[:,0], transformed[:,1]
        return [max(0, xs.min()), max(0, ys.min()), min(w, xs.max()), min(h, ys.max())]
