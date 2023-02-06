from torch import nn


class VGG16Net(nn.Module):  # 继承父类nn.Module
    def __init__(self):
        super(VGG16Net, self).__init__()
        '''
        如果不用super，每次调用父类的方法是需要使用父类的名字
        使用super就避免这个麻烦
        super()实际上的含义远比这个要复杂。
        有兴趣可以通过这篇博客学习：https://blog.csdn.net/zhangjg_blog/article/details/83033210
        '''
        '''
        A sequential container.
        Modules will be added to it in the order they are passed in the
        constructor. Alternatively, an ``OrderedDict`` of modules can be
        passed in. The ``forward()`` method of ``Sequential`` accepts any
        input and forwards it to the first module it contains. It then
        "chains" outputs to inputs sequentially for each subsequent module,
        专业术语：一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数
        这是一个有序模型容器，输入会按照顺序逐层通过每一模型，最终会返回最后一个模型的输出。
        实现原理：利用for循环 将所有的参数(即子模块)加入到self._module,然后在__call__中调用forward()，
        而forward()函数则会将self.module中的子模块推理一遍，返回值也就是最终结果。
        参考博客：https://blog.csdn.net/dss_dssssd/article/details/82980222 
        '''
        # 第一层，2个卷积层和一个最大池化层
        self.layer1 = nn.Sequential(
            # 输入3通道，输出64通道。卷积核大小3，填充1，步长默认为1(输入224*224*3的样本图片，输出为224*224*64)
            nn.Conv2d(3, 64, 3, padding=1),
            # 对数据做归一化，这是由于经过训练后的数据分布改变，需要将其拉回到N(1,0)的正态分布，防止梯度消失，读入的参数是通道数。
            nn.BatchNorm2d(64),
            # 激活函数。参数inplace =false 是默认返回对象，需要重开地址，这里True返回地址，为了节省开销
            nn.ReLU(inplace=True),

            # 输入为224*224*64，输出为224*224*64
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 进行卷积核2*2,步长为2的最大池化，输入224*224*64，输出112*112*64
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第二层，2个卷积层和一个最大池化层
        self.layer2 = nn.Sequential(
            # 输入为112*112*64，输出为112*112*128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 输入为112*112*128，输出为112*112*128
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 112*112*128 --> 56*56*128
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第三层，3个卷积层和一个最大池化
        self.layer3 = nn.Sequential(
            # 56*56*128 --> 56*56*256
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 56*56*256 --> 56*56*256
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 56*56*256 --> 56*56*256
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 56*56*256 --> 28*28*256
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第四层，3个卷积层和一个最大池化
        self.layer4 = nn.Sequential(
            # 28*28*256 --> 28*28*512
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 28*28*512 --> 28*28*512
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 28*28*512 --> 28*28*512
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 28*28*512 --> 14*14*512
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第五层，3个卷积层和一个最大池化
        self.layer5 = nn.Sequential(
            # 14*14*512 --> 14*14*512
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 14*14*512 --> 14*14*512
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 14*14*512 --> 14*14*512
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 14*14*512 --> 7*7*512
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 将卷积层打包
        self.conv_layer = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )
        # 全连接层
        self.fc = nn.Sequential(

            # 1*1*25088 --> 1*1*4096
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=1),
            nn.Dropout(),
            # 1*1*4096 --> 1*1*4096
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=1),
            nn.Dropout(),
            # 1*1*4096 --> 1*1*1000
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        # 7*7*512 --> 1*1*25088
        x = x.view(-1, 25088)
        x = self.fc(x)
        return x
