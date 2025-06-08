import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import albumentations as A

# 설정
original_root = './Large'
augmented_root = './Augmented'
balanced_root = './Balanced'
excluded_classes = ['Mountain', 'Other']
target_total = 500
max_original = 300

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.1, shear=10, p=0.5),
    A.Resize(120, 120)  # 최종적으로 크기 고정
])

os.makedirs(augmented_root, exist_ok=True)
os.makedirs(balanced_root, exist_ok=True)

classes = sorted([
    d for d in os.listdir(original_root)
    if os.path.isdir(os.path.join(original_root, d)) and d not in excluded_classes
])

for cls in classes:
    ori_dir = os.path.join(original_root, cls)
    aug_dir = os.path.join(augmented_root, cls)
    bal_dir = os.path.join(balanced_root, cls)

    os.makedirs(aug_dir, exist_ok=True)
    os.makedirs(bal_dir, exist_ok=True)

    ori_imgs = [f for f in os.listdir(ori_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    np.random.shuffle(ori_imgs)
    selected_ori = ori_imgs[:max_original]
    num_ori = len(selected_ori)

    for fname in selected_ori:
        shutil.copy(os.path.join(ori_dir, fname), os.path.join(bal_dir, f'ori_{fname}'))

    needed_aug = target_total - num_ori
    i = 0
    while len(os.listdir(aug_dir)) < needed_aug:
        img_name = ori_imgs[i % len(ori_imgs)]
        img_path = os.path.join(ori_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            i += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = augment(image=image)['image']
        save_name = f'aug_{i}.jpg'
        save_path_aug = os.path.join(aug_dir, save_name)
        save_path_bal = os.path.join(bal_dir, save_name)
        cv2.imwrite(save_path_aug, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
        shutil.copy(save_path_aug, save_path_bal)
        i += 1

print("✅ 원본 + 증강 포함 균형 데이터셋 생성 완료 → ./recaptcha-dataset/Balanced")
