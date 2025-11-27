import os, glob, random, cv2
from augmentations.labels import LabelManager
from augmentations.geo_augs import *
from augmentations.pixel_augs import *

class ImageProcessor:
    def __init__(self,img_dir="original_images",label_dir="original_labels"):
        self.img_dir=os.path.abspath(img_dir)
        self.label_dir=os.path.abspath(label_dir)

        self.filtered_img_dir=os.path.join(os.getcwd(),"filtered_images")
        self.filtered_label_dir=os.path.join(os.getcwd(),"filtered_labels")
        os.makedirs(self.filtered_img_dir,exist_ok=True)
        os.makedirs(self.filtered_label_dir,exist_ok=True)

        self.pixel_augs=[
            GrayAug(),
            RedAug(),
            GreenAug(),
            BlueAug(),
            NoiseAug(),
            BlurAug(),
            LensFlareAug(),
            RainAug(),
            FogAug(),
            GammaAug()
        ]
        self.geo_augs=[
            FlipHAug(),
            FlipVAug(),
            RotateAug(),
            ShearAug(),
            CropAug()
        ]

    def process_random_images(self,count=6):
        all_images=glob.glob(os.path.join(self.img_dir,"*"))
        if not all_images: print(f"[ERROR] {self.img_dir} bulunamadı"); return
        images=random.sample(all_images,min(count,len(all_images)))

        for img_path in images:
            base=os.path.splitext(os.path.basename(img_path))[0]
            img=cv2.imread(img_path)
            if img is None: print(f"[SKIP] {img_path} okunamadı"); continue
            print(f"[INFO] İşleniyor: {base}, boyut: {img.shape}")

            labels=LabelManager.load_labels(base,self.label_dir)
            h,w=img.shape[:2]; boxes=[LabelManager.yolo_to_xyxy(l,w,h) for l in labels]

            aug=random.choice(self.pixel_augs+self.geo_augs)
            new_img,new_boxes=aug.apply(img,boxes)
            out_base=f"{base}_{aug.name}"
            out_path=os.path.join(self.filtered_img_dir,f"{out_base}.jpg")
            if cv2.imwrite(out_path,new_img): print(f"[OK] {out_path} kaydedildi")
            else: print(f"[ERROR] {out_base} kaydedilemedi")

            if new_boxes:
                LabelManager.write_labels(out_base,new_boxes,self.filtered_label_dir,new_img.shape[1],new_img.shape[0])
                print(f"[OK] Label kaydedildi: {os.path.join(self.filtered_label_dir,out_base+'.txt')}")


if __name__=="__main__":
    p=ImageProcessor()
    p.process_random_images(count=6)