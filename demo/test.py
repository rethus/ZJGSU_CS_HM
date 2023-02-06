import torch
from dataset.read_data import Mydata
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from model import VGG16Net

testdata = Mydata('../data/data/test')
test_loader = torch.utils.data.DataLoader(testdata,batch_size=1,shuffle=False)


device = torch.device("cpu")
model = torch.load('../dataset/model.pth').to(device)


model.eval()
correct = 0.0
total = 0
with torch.no_grad():
    for input, label in test_loader:
        inputs = input.to(device)
        outputs = model(inputs)
        pred = outputs.argmax(dim=1)# 返回每一行中最大元素索引
        total += inputs.size(0)
        correct += torch.eq(pred,label).sum().item()# 比较实际结果和标签
print('Accuracy of the network on the 10000 test images:%.2f%%'%(100.0*correct/total))
print(total)

