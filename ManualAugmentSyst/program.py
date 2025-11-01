import cv2, glob, random, os
import numpy as np


def add_gray(frame):# Important Filter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

def add_red_filter(frame,coeff=1.5):
    frame[:,:,2] = np.clip(frame[:,:,2]*coeff, 0, 255)
    return frame

def add_green_filter(frame,coeff=1.5):
    frame[:,:,1] = np.clip(frame[:,:,1]*coeff, 0, 255)
    return frame

def add_blue_filter(frame,coeff=1.5):
    frame[:,:,0] = np.clip(frame[:,:,0]*coeff, 0, 255)
    return frame

def add_noise(frame):
    frame = frame + np.random.normal(0, 10, frame.shape)
    return frame

def add_blur(frame):
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    return frame

def add_lens_flare(frame):
    frame = frame.copy()
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame, dtype=np.uint8)

    # 1️⃣ Ana ışık kaynağı (güneş parlama merkezi)
    cx = random.randint(int(w * 0.2), int(w * 0.8))
    cy = random.randint(0, int(h * 0.4))
    main_radius = random.randint(80, 150)
    color = (255, random.randint(220, 255), random.randint(180, 230))
    cv2.circle(overlay, (cx, cy), main_radius, color, -1)

    # 2️⃣ Lens flare halkaları
    for i in range(random.randint(3, 5)):
        # Her halka merkezin tam karşı yönünde ilerlesin
        fx = int(cx + (i + 1) * (w // 10))
        fy = int(cy + (i + 1) * (h // 12))
        radius = random.randint(20, 50)
        color = (random.randint(200, 255), random.randint(180, 255), random.randint(180, 255))
        alpha = random.uniform(0.15, 0.4)
        flare = np.zeros_like(frame, dtype=np.uint8)
        cv2.circle(flare, (fx, fy), radius, color, -1)
        overlay = cv2.addWeighted(overlay, 1, flare, alpha, 0)

    # 3️⃣ Gaussian blur ile yumuşatma
    overlay = cv2.GaussianBlur(overlay, (51, 51), 0)

    # 4️⃣ Orijinalle karıştır
    output = cv2.addWeighted(frame, 1.0, overlay, 0.6, 0)
    return output

def add_rain(frame):
    rain = np.zeros_like(frame)
    dropnumber=random.randint(100,150)
    for _ in range(dropnumber):  # damla sayısı
        x = random.randint(0, frame.shape[1] - 1)
        y = random.randint(0, frame.shape[0] - 1)
        length = random.randint(10, 20)
        cv2.line(rain, (x, y), (x, y + length), (200, 200, 200), 1)
    rain = cv2.blur(rain, (3, 3))
    frame = cv2.addWeighted(frame, 1, rain, 0.3, 0)
    return frame

def add_fog(frame, intensity=0.3):
    overlay = np.full_like(frame, 255, dtype=np.uint8)  # beyaz sis
    return cv2.addWeighted(frame, 1-intensity, overlay, intensity, 0)

def adjust_gamma(frame, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

def flip_image(img,a=0):# Important Filter
    if a==1:
        img = cv2.flip(img, 1)
    else:
        img = cv2.flip(img, 0) 
    return img

def rotate_image(img, angle=45):# Important Filter
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (w, h))
    return img_rot

def shear_image(img, shear_factor=0.05):
    h, w = img.shape[:2]
    pts1 = np.float32([[0,0],[w,0],[0,h]])
    pts2 = np.float32([[0,0],[w,h*shear_factor],[w*shear_factor,h]])
    M = cv2.getAffineTransform(pts1, pts2)
    img_sheared = cv2.warpAffine(img, M, (w,h))
    return img_sheared

def random_crop_zoom(img, crop_ratio=0.5):# Important Filter
    h, w = img.shape[:2]
    new_h, new_w = int(h*crop_ratio), int(w*crop_ratio)
    top = random.randint(0, h-new_h)
    left = random.randint(0, w-new_w)
    img_cropped = img[top:top+new_h, left:left+new_w]
    return img_cropped




os.makedirs("filtered", exist_ok=True)
images = random.sample(glob.glob("originals/*"), 6)
filtercoeff= 1.6

for img_path in images:
    img = cv2.imread(img_path)
    choice = random.choice(["gray","red","green","blue","noise","blur","lensframe","rain","fog","gama","flipimage","rotate","shear","crop"])
    
    if choice == "gray": img = add_gray(img)
    elif choice == "red": img = add_red_filter(img,filtercoeff)#standart coeff = 1.5
    elif choice == "green": img = add_green_filter(img,filtercoeff)#standart coeff = 1.5
    elif choice == "blue": img = add_blue_filter(img,filtercoeff)#standart coeff = 1.5
    elif choice == "noise": img = add_noise(img)
    elif choice == "blur": img = add_blur(img)
    elif choice == "lensframe" : img=add_lens_flare(img)
    elif choice == "add_rain" : img=add_rain(img)
    elif choice == "fog" : img = add_fog(img)
    elif choice == "gama" : img = adjust_gamma(img,filtercoeff)
    elif choice == "flipimage" : img = flip_image(img,0) #vertical
    elif choice == "rotate" : img = rotate_image(img) #standart angle = 45
    elif choice == "shear" : img = shear_image(img)
    elif choice == "crop" : img = random_crop_zoom(img) #standart ratio = 1
    cv2.imwrite("filtered/" + os.path.basename(img_path).split('.')[0] + f"_{choice}.jpg", img)
