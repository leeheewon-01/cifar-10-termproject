import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm
from tqdm import tqdm
import torch.nn.functional as F

model_num = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean, std)(transforms.ToTensor()(crop)) for crop in crops]))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

model_nums= [5,0,2,4,1,3]
models = []
for i in model_nums:
    model = timm.create_model('resnet18', num_classes=10)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(f"./baselinev4_Models/baselinev4_models_seed_{i}.pt"))
    model.eval()
    model = model.to(device)
    models.append(model)

correct, total = 0, 0

preds = []

# base
# with torch.no_grad():
#     for data in tqdm(testloader):
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         bs, ncrops, c, h, w = images.size()
#         outputs = torch.zeros(bs, 10).to(device)
#         for model in models:
#             model_output = model(images.view(-1, c, h, w))
#             model_output = model_output.view(bs, ncrops, -1).mean(1)
#             outputs += model_output
#         _, predicted = torch.max(outputs.data, 1)
#         preds.append(predicted)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# # soft voting
with torch.no_grad():
    for data in tqdm(testloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
        bs, ncrops, c, h, w = images.size()       
        outputs = torch.zeros(bs, 10).to(device)
        for model in models:
            model_output = model(images.view(-1, c, h, w))
            model_output = model_output.view(bs, ncrops, -1).mean(1)
            model_output = F.softmax(model_output, dim=1)  # apply softmax to get probabilities
            outputs += model_output
        outputs /= len(models)  # average the probabilities
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()



from sklearn.metrics import classification_report
import pandas as pd
preds, true_labels = [], []

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

with torch.no_grad():
    for imgs, labels in tqdm(iter(testloader)):
        imgs = imgs.float().to(device)
        labels = labels.to(device)
        
        bs, ncrops, c, h, w = imgs.size()
        outputs = torch.zeros(bs, 10).to(device)
        for model in models:
            model_output = model(imgs.view(-1, c, h, w))
            model_output = model_output.view(bs, ncrops, -1).mean(1)
            outputs += model_output

        preds += outputs.argmax(1).detach().cpu().numpy().tolist()
        true_labels += labels.detach().cpu().numpy().tolist()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

report_df = pd.DataFrame(classification_report(true_labels, preds, target_names=class_names, output_dict=True)).transpose()
print(report_df)

print('Accuracy of the ensemble on the 10000 test images: %f %%' % (100 * correct / total))
