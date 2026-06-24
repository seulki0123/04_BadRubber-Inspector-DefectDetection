import os

import cv2

from defect_detection import Detector

if __name__ == "__main__":
    # 1. input datas
    img_dir = "/home/s1k2/04_BadRubber/src/점이물테스트/NBR/orig09"
    imgs_path = [os.path.join(img_dir, img_path) for img_path in os.listdir(img_dir) if img_path.endswith(".jpg")]
    imgs = [cv2.imread(img_path) for img_path in imgs_path]
    dst_dir = "/home/s1k2/04_BadRubber/src/점이물테스트/NBR/orig09_detect"
    os.makedirs(dst_dir, exist_ok=True)
    detected = 0
    
    # 2. inspect
    detector = Detector()
    for idx, img in enumerate(imgs):
        results = detector.detect([img])

        # 3. visualize
        for img_path, result in zip(imgs_path, results):
            print(result.anomaly_cls.regions)
            if len(result.anomaly_cls.regions):
                detected += 1
                print("detected:", detected)
            # image_name = os.path.basename(img_path)
            image_name = os.path.basename(imgs_path[idx])
            cv2.imwrite(os.path.join(dst_dir, image_name), result.visualize())