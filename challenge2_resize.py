import os
import cv2
from tqdm import tqdm

input_root = './testset'
output_root = './resized/test'
os.makedirs(output_root, exist_ok=True)

target_size = (224, 224)

for fname in tqdm(sorted(os.listdir(input_root)), desc='Resizing test images'):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_root, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue

    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    save_path = os.path.join(output_root, fname)
    cv2.imwrite(save_path, resized)

print("✅ 테스트 이미지 224x224 리사이즈 완료 → ./resized/test/")
