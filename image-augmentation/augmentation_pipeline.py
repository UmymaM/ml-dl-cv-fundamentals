import albumentations as A
import cv2
import matplotlib.pyplot as plt
class AugmentationPipeline:
    def __init__(self,image_size=224):
        self.transform=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20,p=0.5),
            A.RandomResizedCrop(size=(image_size, image_size),scale=(0.8, 1.0),
                ratio=(0.9, 1.1),p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.5),
            A.GaussNoise(var_limit=(10.0,50.0),p=0.3)])
    def apply(self,image):
        transformed=self.transform(image=image)
        return transformed["image"]

def visualize_augmentations(image_path, pipeline, num_versions=5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12,4))
    for i in range(num_versions):
        transformed=pipeline.apply(image)
        plt.subplot(1, num_versions, i+1)
        plt.imshow(transformed)
        plt.title(f"Aug {i+1}")
        plt.axis("off")

    plt.show()



