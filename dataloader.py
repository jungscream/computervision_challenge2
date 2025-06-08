from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 전처리 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 기준
                         std=[0.229, 0.224, 0.225])
])

# 이미지 로딩
train_dataset = datasets.ImageFolder(root='./resized/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 클래스 확인
print("클래스 목록:", train_dataset.classes)
