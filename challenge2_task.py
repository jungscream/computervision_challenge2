import os
import csv
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 전처리 정의 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === 모델 로딩 ===
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('best_resnet18_challenge2.pth', map_location=device))
model = model.to(device)
model.eval()

# === Feature extractor (FC 제외) ===
feature_extractor = nn.Sequential(*list(model.children())[:-1])

# === Feature 추출 함수 ===
def extract_features_train(folder_path):
    features, labels = [], []
    for cls_idx, cls in enumerate(sorted(os.listdir(folder_path))):
        cls_dir = os.path.join(folder_path, cls)
        if not os.path.isdir(cls_dir): continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith('.png'):
                continue
            img_path = os.path.join(cls_dir, fname)
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = feature_extractor(input_tensor).squeeze().cpu().numpy()
            features.append(feat)
            labels.append(cls_idx)
    return np.array(features), labels

def extract_features_test(folder_path):
    features, filenames = [], []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith('.png'):
            continue
        img_path = os.path.join(folder_path, fname)
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = feature_extractor(input_tensor).squeeze().cpu().numpy()
        features.append(feat)
        filenames.append(fname)
    return np.array(features), filenames

# === Feature 추출 ===
train_feats, y_train = extract_features_train('./resized/train')
test_feats, test_fnames = extract_features_test('./resized/test')

# === 정규화 및 차원 축소 ===
scaler = StandardScaler()
train_feats_scaled = scaler.fit_transform(train_feats)
test_feats_scaled = scaler.transform(test_feats)

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(train_feats_scaled)
X_test_pca = pca.transform(test_feats_scaled)

# === 클래스 이름 로딩 ===
label_map = sorted(os.listdir('./resized/train'))

# === KNN 학습 (n_neighbors=10 for Top-10) ===
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train_pca, y_train)

# === Top-10 예측 ===
_, indices = knn.kneighbors(X_test_pca)  # (num_test, 10)
top10_labels = [[label_map[y_train[i]] for i in idx_row] for idx_row in indices]

# === c2_t2_a1.csv: Top-10 예측 저장 ===
with open('c2_t2_a1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for fname, labels in zip(test_fnames, top10_labels):
        writer.writerow([fname] + labels)

# === c2_t1_a1.csv: Top-1(최상위) 클래스 분류 결과 저장 ===
with open('c2_t1_a1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'predicted_class'])  # 헤더
    for fname, labels in zip(test_fnames, top10_labels):
        top1_class = labels[0]  # 가장 가까운 이웃의 클래스
        writer.writerow([fname, top1_class])

print("✅ Top-10 예측 → c2_t2_a1.csv 저장됨")
print("✅ Top-1 분류 결과 → c2_t1_a1.csv 저장됨")