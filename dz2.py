import os
import torch
import shutil
import yaml
import cv2
from sklearn.model_selection import train_test_split

# 设置本地路径
yolov7_path = "D:\dz2/yolov7"
dataset_path = "D:\dz2/dataset"
os.chdir(yolov7_path)

# 如果尚未克隆 YOLOv7，可手动执行以下命令：
# os.system("git clone https://github.com/augmentedstartups/yolov7.git")
# os.chdir("yolov7")
# os.system("pip install -r requirements.txt")
# os.system("pip install numpy==1.23.5")

# 划分数据集

test_size = 0.2
valid_size = 0.1

image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
annotation_files = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]

image_files_with_annotations = [f for f in image_files if f.replace('.jpg', '.txt') in annotation_files]

train_files, test_files = train_test_split(image_files_with_annotations, test_size=test_size, random_state=42)
train_files, valid_files = train_test_split(train_files, test_size=len(image_files_with_annotations)*valid_size/len(train_files), random_state=42)

def copy_files(files, source_path, dest_path):
    for file in files:
        shutil.copy(os.path.join(source_path, file), os.path.join(dest_path, 'images'))
        shutil.copy(os.path.join(source_path, file.replace('.jpg', '.txt')), os.path.join(dest_path, 'labels'))

for folder in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(dataset_path, folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, folder, 'labels'), exist_ok=True)

copy_files(train_files, dataset_path, os.path.join(dataset_path, 'train'))
copy_files(valid_files, dataset_path, os.path.join(dataset_path, 'valid'))
copy_files(test_files, dataset_path, os.path.join(dataset_path, 'test'))

# 清理无用文件
for file in os.listdir(dataset_path):
    if not file.endswith('.names') and os.path.isfile(os.path.join(dataset_path, file)):
        os.remove(os.path.join(dataset_path, file))

# 生成 data.yaml 文件
train_path = os.path.join(dataset_path, 'train/images')
val_path = os.path.join(dataset_path, 'valid/images')
test_path = os.path.join(dataset_path, 'test/images')

names_file = os.path.join(dataset_path, 'obj.names')
with open(names_file, 'r') as file:
    class_names = [line.strip() for line in file.readlines()]

num_classes = len(class_names)

data = {
    'train': train_path.replace('\\', '/'),
    'val': val_path.replace('\\', '/'),
    'test': test_path.replace('\\', '/'),
    'nc': num_classes,
    'names': class_names
}

output_file = os.path.join(dataset_path, 'data.yaml')
with open(output_file, 'w') as file:
    yaml.dump(data, file)


# 删除 obj.names 文件(到这步为止要注释掉)
os.remove(names_file)

# 下载 yolov7.pt 权重（如果还未下载）
# os.system("wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt -P .")

# 加载权重检查是否正常（可选）
#weights = torch.load("yolov7.pt", map_location=torch.device('cpu'))
#print("Loaded YOLOv7 weights successfully.")

output_file = os.path.join(dataset_path, 'data.yaml')


# 启动训练
train_command = f"python train.py --batch 4 --cfg cfg/training/yolov7.yaml --epochs 220 --data {output_file} --weights yolov7.pt --device 0"
os.system(train_command)

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # 修改为你本地的训练结果路径
# image_dir = "D:/train/python/dz2/yolov7/runs/train/exp"

# # 显示 F1 曲线
# f1_img = mpimg.imread(f"{image_dir}/F1_curve.png")
# plt.figure(figsize=(5, 5))
# plt.title("F1 Curve")
# plt.imshow(f1_img)
# plt.axis('off')
# plt.show()

# # 显示 PR 曲线
# pr_img = mpimg.imread(f"{image_dir}/PR_curve.png")
# plt.figure(figsize=(5, 5))
# plt.title("PR Curve")
# plt.imshow(pr_img)
# plt.axis('off')
# plt.show()

# # 显示混淆矩阵
# cm_img = mpimg.imread(f"{image_dir}/confusion_matrix.png")
# plt.figure(figsize=(6, 6))
# plt.title("Confusion Matrix")
# plt.imshow(cm_img)
# plt.axis('off')
# plt.show()
