import os

class LabelManager:
    @staticmethod
    def yolo_to_xyxy(line,img_w,img_h):
        cls=int(line[0]); xc=float(line[1])*img_w; yc=float(line[2])*img_h
        w=float(line[3])*img_w; h=float(line[4])*img_h
        return cls,[xc-w/2,yc-h/2,xc+w/2,yc+h/2]

    @staticmethod
    def xyxy_to_yolo(bbox,img_w,img_h,cls):
        xmin,ymin,xmax,ymax=bbox
        xmin=max(0,min(xmin,img_w-1)); ymin=max(0,min(ymin,img_h-1))
        xmax=min(img_w-1,max(0,xmax)); ymax=min(img_h-1,max(0,ymax))
        w=xmax-xmin; h=ymax-ymin
        if w<=0 or h<=0: return None
        return f"{cls} {(xmin+xmax)/2/img_w:.6f} {(ymin+ymax)/2/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}"

    @staticmethod
    def load_labels(base,labels_dir):
        path=os.path.join(labels_dir,base+".txt")
        if not os.path.exists(path): return []
        with open(path,"r") as f: return [line.split() for line in f.read().strip().splitlines() if len(line.split())==5]

    @staticmethod
    def write_labels(base,boxes,out_dir,img_w,img_h):
        os.makedirs(out_dir,exist_ok=True)
        path=os.path.join(out_dir,base+".txt")
        lines=[LabelManager.xyxy_to_yolo(bbox,img_w,img_h,cls) for cls,bbox in boxes if LabelManager.xyxy_to_yolo(bbox,img_w,img_h,cls)]
        with open(path,"w") as f: f.write("\n".join(lines))
