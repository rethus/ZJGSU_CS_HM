import torch
from dataset.read_data import Mydata
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from model import VGG16Net, HighResolutionNet
import time

#超参数设置
batch_size  = 32 # 每次训练的数据量
learn_rate = 0.01 # 学习率
step_size = 10 # 控制学习率变化
epuch_num = 1 # 总的训练次数
num_print = 10 # 每n个batch打印一次

mydata = Mydata("../data/data/train")

# 利用dataloader加载数据集
train_loader = torch.utils.data.DataLoader(mydata,batch_size=batch_size,shuffle=True,drop_last=True)

# 生成驱动器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 利用驱动器驱动模型
model = VGG16Net().to(device)

# 损失函数
get_loss = nn.CrossEntropyLoss()

# SGD优化器  第一个参数是输入需要优化的参数,第二个是学习率，第三个是动量，大致就是借助上一次导数结果，加快收敛速度。
'''
这一行代码里面实际上包含了多种优化:
一个是动量优化,增加了一个关于上一次迭代得到的系数的偏置，借助上一次的指导，减小梯度震荡，加快收敛速度
一个是权重衰减，通过对权重增加一个(正则项),该正则项会使得迭代公式中的权重按照比例缩减，这么做的原因是，过拟合的表现一般为参数浮动大，使用小参数可以防止过拟合
'''
optimizer = optim.SGD(model.parameters(),lr=learn_rate,momentum=0.8,weight_decay=0.001)

# 动态调整学习率 StepLR 是等间隔调整学习率，每step_size 令lr=lr*gamma
# 学习率衰减，随着训练的加深，目前的权重也越来越接近最优权重，原本的学习率会使得，loss上下震荡，逐步减小学习率能加快收敛速度。
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=0.5,last_epoch=-1)

loss_list=[]
start=time.time()

for epoch in range(epuch_num):
    running_loss = 0.0
    # enumerate()是python自带的函数，用于迭代字典。参数1，是需要迭代的对象，第二参数是迭代的起始位置
    for i,(inputs, labels) in enumerate(train_loader,0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()# 将梯度初始化为0
        outputs = model(inputs)# 前向传播求出预测的值
        loss = get_loss(outputs,labels).to(device)# 求loss,对应loss +=

        loss.backward() # 反向传播求梯度
        optimizer.step() # 更新所有参数

        running_loss += loss.item()# loss是张量，访问值时需要使用item()
        loss_list.append(loss.item())

        if i % num_print == num_print - 1:
            print('[%d epoch, %d] loss: %.6f' % (epoch + 1, i + 1, running_loss / num_print))
            running_loss = 0.0
    lr = optimizer.param_groups[0]['lr']# 查看目前的学习率
    print('learn_rate : %.15f'% lr)
    scheduler.step()# 根据迭代epoch更新学习率

end = time.time()
print('time:{}'.format(end-start))

torch.save(model,'./model.pth')
