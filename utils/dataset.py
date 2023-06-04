import cv2
import torchvision
import torch

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]
            # img = torch.from_numpy(img.transpose(2, 0, 1))

        return img, label