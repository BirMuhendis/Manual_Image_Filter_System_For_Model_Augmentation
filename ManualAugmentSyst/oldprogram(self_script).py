import cv2, glob, random, os
import numpy as np

def yolo_to_xyxy(line, img_w, img_h):
    cls = int(line[0])
    xc = float(line[1]) * img_w
    yc = float(line[2]) * img_h
    w = float(line[3]) * img_w
    h = float(line[4]) * img_h
    xmin = xc - w/2
    ymin = yc - h/2
    xmax = xc + w/2
    ymax = yc + h/2
    return cls, [xmin, ymin, xmax, ymax]

def xyxy_to_yolo(bbox, img_w, img_h, cls):
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, min(xmin, img_w-1))
    ymin = max(0, min(ymin, img_h-1))
    xmax = max(0, min(xmax, img_w-1))
    ymax = max(0, min(ymax, img_h-1))
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        return None
    xc = (xmin + xmax)/2.0 / img_w
    yc = (ymin + ymax)/2.0 / img_h
    wn = w / img_w
    hn = h / img_h
    return f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"

def clip_bbox(bbox, img_w, img_h):
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, min(xmin, img_w-1))
    ymin = max(0, min(ymin, img_h-1))
    xmax = max(0, min(xmax, img_w-1))
    ymax = max(0, min(ymax, img_h-1))
    return [xmin, ymin, xmax, ymax]

def bbox_area_valid(bbox, min_dim=3):
    xmin, ymin, xmax, ymax = bbox
    return (xmax - xmin) >= min_dim and (ymax - ymin) >= min_dim

def add_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def add_red_filter(frame, coeff=1.1):
    frame = frame.copy()
    frame[:,:,2] = np.clip(frame[:,:,2] * coeff, 0, 255)
    return frame

def add_green_filter(frame, coeff=0.5):
    frame = frame.copy()
    frame[:,:,1] = np.clip(frame[:,:,1] * coeff, 0, 255)
    return frame

def add_blue_filter(frame, coeff=0.5):
    frame = frame.copy()
    frame[:,:,0] = np.clip(frame[:,:,0] * coeff, 0, 255)
    return frame

def add_noise(frame):
    noise = np.random.normal(0, 10, frame.shape).astype(np.float32)
    out = frame.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def add_blur(frame):
    return cv2.GaussianBlur(frame, (5,5), 0)

def add_lens_flare(frame):
    frame = frame.copy()
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame, dtype=np.uint8)
    cx = random.randint(int(w*0.2), int(w*0.8))
    cy = random.randint(0, int(h*0.4))
    main_radius = random.randint(80, 150)
    color = (255, random.randint(220,255), random.randint(180,230))
    cv2.circle(overlay, (cx, cy), main_radius, color, -1)
    for i in range(random.randint(3, 5)):
        fx = int(cx + (i+1) * (w//10))
        fy = int(cy + (i+1) * (h//12))
        radius = random.randint(20, 50)
        color = (random.randint(200,255), random.randint(180,255), random.randint(180,255))
        flare = np.zeros_like(frame)
        cv2.circle(flare, (fx, fy), radius, color, -1)
        overlay = cv2.addWeighted(overlay, 1, flare, random.uniform(0.15, 0.4), 0)
    overlay = cv2.GaussianBlur(overlay, (51, 51), 0)
    return cv2.addWeighted(frame, 1.0, overlay, 0.6, 0)

def add_rain(frame):
    rain = np.zeros_like(frame)
    dropnum = random.randint(100,150)
    for _ in range(dropnum):
        x = random.randint(0, frame.shape[1]-1)
        y = random.randint(0, frame.shape[0]-1)
        length = random.randint(10,20)
        cv2.line(rain, (x, y), (x, y+length), (200,200,200), 1)
    rain = cv2.blur(rain, (3,3))
    return cv2.addWeighted(frame, 1, rain, 0.3, 0)

def add_fog(frame, intensity=0.3):
    overlay = np.full_like(frame, 255)
    return cv2.addWeighted(frame, 1-intensity, overlay, intensity, 0)

def adjust_gamma(frame, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([(i/255.0)**invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(frame, table)

def rotate_image_with_M(img, angle=15):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    out = cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    return out,M

def shear_image_with_M(img,shear_factor=0.05):
    h, w = img.shape[:2]
    pts1 = np.float32([[0,0],[w,0],[0,h]])
    pts2 = np.float32([[0,0],[w,h*shear_factor],[w*shear_factor,h]])
    M = cv2.getAffineTransform(pts1,pts2)
    out = cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    return out,M

def crop_image_with_params(img,crop_ratio=0.6):
    h,w = img.shape[:2]
    new_h,new_w = max(1,int(h*crop_ratio)), max(1,int(w*crop_ratio))
    top = random.randint(0,h-new_h) if h-new_h>0 else 0
    left = random.randint(0,w-new_w) if w-new_w>0 else 0
    out = img[top:top+new_h,left:left+new_w]
    return out,(top,left,new_w,new_h)

def flip_h(img):
    return cv2.flip(img,1)

def flip_v(img):
    return cv2.flip(img,0)

def transform_bbox_affine(bbox,M):
    xmin,ymin,xmax,ymax = bbox
    pts = np.array([[xmin,ymin,1],[xmax,ymin,1],[xmax,ymax,1],[xmin,ymax,1]])
    transformed = (M @ pts.T).T
    xs = transformed[:,0]
    ys = transformed[:,1]
    return [xs.min(),ys.min(),xs.max(),ys.max()]

def load_labels(base, labels_dir):
    path = os.path.join(labels_dir, base+".txt")
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path,'r') as f:
        for line in f.read().strip().splitlines():
            parts = line.strip().split()
            if len(parts)!=5: continue
            boxes.append(parts)
    return boxes

def parse_labels(box_lines,img_w,img_h):
    out=[]
    for parts in box_lines:
        cls,bbox = yolo_to_xyxy(parts,img_w,img_h)
        out.append((cls,bbox))
    return out

def write_yolo_labels(base, boxes_xyxy, out_dir,img_w,img_h):
    os.makedirs(out_dir,exist_ok=True)
    out_path = os.path.join(out_dir, base+".txt")
    lines=[]
    for cls,bbox in boxes_xyxy:
        s = xyxy_to_yolo(bbox,img_w,img_h,cls)
        if s: lines.append(s)
    with open(out_path,'w') as f:
        f.write("\n".join(lines))

def main():
    os.makedirs("filtered", exist_ok=True)
    os.makedirs("filtered_labels", exist_ok=True)
    images = random.sample(glob.glob("original_images/*"), 6)
    filtercoeff = 1.3
    labels_dir = "original_labels"

    choices = [
        "gray","red","green","blue","noise","blur",
        "lensflare","rain","fog","gamma",
        "flip_h","flip_v","rotate","shear","crop"
    ]

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None: continue
        h0,w0 = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0]

        label_lines = load_labels(base,labels_dir)
        boxes = parse_labels(label_lines,w0,h0) if label_lines else []

        choice = random.choice(choices)
        new_img = img.copy()
        new_boxes = boxes.copy()

        if choice=="gray": new_img=add_gray(new_img)
        elif choice=="red": new_img=add_red_filter(new_img,filtercoeff+0.2)
        elif choice=="green": new_img=add_green_filter(new_img,filtercoeff)
        elif choice=="blue": new_img=add_blue_filter(new_img,filtercoeff)
        elif choice=="noise": new_img=add_noise(new_img)
        elif choice=="blur": new_img=add_blur(new_img)
        elif choice=="lensflare": new_img=add_lens_flare(new_img)
        elif choice=="rain": new_img=add_rain(new_img)
        elif choice=="fog": new_img=add_fog(new_img)
        elif choice=="gamma": new_img=adjust_gamma(new_img,filtercoeff+0.2)

        elif choice=="flip_h":
            new_img=flip_h(new_img)
            new_boxes = [(cls,[w0-xmax,ymin,w0-xmin,ymax]) for cls,[xmin,ymin,xmax,ymax] in new_boxes]
        elif choice=="flip_v":
            new_img=flip_v(new_img)
            new_boxes = [(cls,[xmin,h0-ymax,xmax,h0-ymin]) for cls,[xmin,ymin,xmax,ymax] in new_boxes]
        elif choice=="rotate":
            angle=random.uniform(-30,30)
            new_img,M=rotate_image_with_M(new_img,angle)
            new_boxes=[(cls,clip_bbox(transform_bbox_affine(bbox,M),w0,h0)) for cls,bbox in new_boxes]
        elif choice=="shear":
            factor=random.uniform(-0.12,0.12)
            new_img,M=shear_image_with_M(new_img,factor)
            new_boxes=[(cls,clip_bbox(transform_bbox_affine(bbox,M),w0,h0)) for cls,bbox in new_boxes]
        elif choice=="crop":
            ratio=random.uniform(0.4,0.9)
            new_img,(top,left,new_w,new_h)=crop_image_with_params(new_img,ratio)
            new_boxes_trans=[]
            for cls,bbox in new_boxes:
                xmin,ymin,xmax,ymax=bbox
                xmin-=left
                xmax-=left
                ymin-=top
                ymax-=top
                tb=clip_bbox([xmin,ymin,xmax,ymax],new_w,new_h)
                if bbox_area_valid(tb): new_boxes_trans.append((cls,tb))
            new_boxes=new_boxes_trans

        out_base=f"{base}_{choice}"
        cv2.imwrite(f"filtered/{out_base}.jpg", new_img)
        h_new,w_new=new_img.shape[:2]

        if choice in ["flip_h","flip_v","rotate","shear","crop"]:
            write_yolo_labels(out_base,new_boxes,"filtered_labels",w_new,h_new)
        else:
            orig_txt_path=os.path.join(labels_dir,base+".txt")
            if os.path.exists(orig_txt_path):
                dst_txt_path=os.path.join("filtered_labels",out_base+".txt")
                with open(orig_txt_path,'r') as f1, open(dst_txt_path,'w') as f2:
                    f2.write(f1.read())

        print(f"Saved filtered/{out_base}.jpg with {len(new_boxes)} boxes (choice={choice})")

if __name__=="__main__":
    main()
