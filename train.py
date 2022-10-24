import torch.utils.data
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

from model import Model
import torch
import torch.nn as nn

# dataset设置
train_data = torchvision.datasets.CIFAR10(root="../CIFAR10/data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../CIFAR10/data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练集的长度为{}'.format(train_data_size))
print('测试集的长度为{}'.format(test_data_size))

# dataloader设置
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

# 创建网络模型
model = Model()

# 创建损失函数
loss_function = nn.CrossEntropyLoss()

# 定义优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../CIFAR10/logs_train")

# 主训练过程
# 第一个循环设定训练轮数
for i in range(epoch):
    print("------第{}轮开始-------".format(i+1))

    model.train()
    # 第二个循环用于取出数据
    for data in train_dataloader:
        images, targets = data
        # 正向：计算输出、损失
        # 将输入通过网络
        outputs = model(images)
        # 计算损失
        loss = loss_function(outputs, targets)

        # 反向：优化器优化模型
        # 优化器的梯度清零
        optimizer.zero_grad()
        # 反向传播求梯度
        loss.backward()
        # 优化器利用梯度进行优化
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    model.eval()
    # 一轮循环结束，测试集测试
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, targets = data
            outputs = model(images)
            loss = loss_function(outputs, targets)
            total_test_loss += loss
    print("整体测试集上loss是{}".format(total_test_loss.item()))
    writer.add_scalar("test_loss", total_test_loss.item(), total_test_step)
    total_test_step += 1

    # 保存训练模型
    torch.save(model.state_dict(), "model_{}.pth".format(i))
    print("model have saved")

writer.close()
