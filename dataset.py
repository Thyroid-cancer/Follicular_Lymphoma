import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

class FLDataset(Dataset):
    def __init__(self, root_dir, phase='train', img_size=(224, 224), val_split=0.2):
        """
        root_dir: 根目录 "汇总数据-统一为6张"
        phase: 'train', 'val' 或 'test'
        img_size: 目标图像大小 (H, W)
        val_split: 训练集中分割出 20% 作为验证集
        """
        self.root_dir = root_dir
        self.phase = phase
        self.img_size = img_size
        self.data = []

        # 训练 & 测试医院划分
        train_hospitals = ['妇幼', '广东医', '广中药', '昆明医', '南方医', '珠海', '省医']
        test_hospitals = ['桂林医', '华医', '文山']

        # 设定类别标签
        self.classes = {'滤泡亚型': 0, '滤泡癌': 0, '腺瘤': 1}
        # self.classes = {'滤泡亚型': 0, '滤泡癌': 1, '腺瘤': 2}
        if phase == 'test':
            hospitals = test_hospitals
        else:
            hospitals = train_hospitals

        all_patients = []  # 用于存放所有病人的数据

        for hospital in hospitals:
            hospital_path = os.path.join(root_dir, hospital)
            for disease, label in self.classes.items():
                disease_path = os.path.join(hospital_path, disease)
                if not os.path.exists(disease_path):
                    continue

                patient_folders = os.listdir(disease_path)
                for patient in patient_folders:
                    image_folder = os.path.join(disease_path, patient, 'imageCutErWei')
                    if os.path.exists(image_folder):
                        image_files = sorted(os.listdir(image_folder))  # 确保顺序一致
                        image_paths = [os.path.join(image_folder, img) for img in image_files if
                                       img.endswith(('.png', '.jpg', '.jpeg'))]

                        if len(image_paths) == 6:  # 确保每个病人都有 6 张图像
                            all_patients.append((image_paths, label))

            # 训练集 & 验证集划分（在病人级别上）
        if phase in ['train', 'val']:
            # random.shuffle(all_patients)  # 打乱病人顺序
            split_idx = int(len(all_patients) * (1 - val_split))
            train_patients = all_patients[:split_idx]
            val_patients = all_patients[split_idx:]

            self.data = train_patients if phase == 'train' else val_patients
        else:
            self.data = all_patients  # 测试集直接使用所有数据

            # 设置数据变换
        self.transform = self.get_transforms(phase)

    def get_transforms(self, phase):
        """
        获取不同阶段的数据增强和预处理方法
        """
        if phase == 'train':
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.RandomRotation(30),  # 随机旋转
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  # 颜色抖动
                transforms.GaussianBlur(3),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # 随机平移
                transforms.ToTensor(),
                transforms.Normalize(mean=0.485, std=0.229)
            ])
        else:  # 'val' 和 'test' 只进行标准化处理
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.485, std=0.229)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_paths, label = self.data[idx]
        images = [self.transform(Image.open(img).convert('RGB')) for img in image_paths]
        images_tensor = torch.cat(images, dim=0)  # 形状: (6xC, H, W)
        return images_tensor, label, image_paths

if __name__ == '__main__':
    dataset = FLDataset('汇总数据-统一为6张', 'train')
    print(len(dataset))
