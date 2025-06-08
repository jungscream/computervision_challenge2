import os
import cv2
from tqdm import tqdm

input_root = './Balanced'
output_root = './resized/training'  # 저장 위치
os.makedirs(output_root, exist_ok=True)

target_size = (224, 224)

# 제외할 클래스가 있으면 여기에
excluded_classes = ['Mountain', 'Other']

for cls in sorted(os.listdir(input_root)):
    if cls in excluded_classes:
        continue

    src_dir = os.path.join(input_root, cls)
    dst_dir = os.path.join(output_root, cls)
    os.makedirs(dst_dir, exist_ok=True)

    for fname in tqdm(os.listdir(src_dir), desc=f'Resizing {cls}'):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(src_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        save_path = os.path.join(dst_dir, fname)
        cv2.imwrite(save_path, resized)

print("✅ 모든 이미지를 224x224로 리사이즈 완료 → ./resized/train/")
