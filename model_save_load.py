import torch
from nn_module import Module

model = Module()

# model_save方法
# 利用torch.save将model的参数以字典的方式保存在路径"model.pth"中
torch.save(model.state_dict(), "model.pth")


# model_load方法
# 先实例化一个模型结构出来
model = Module()
# torch.load(路径)将保存的参数字典拿出来，用.load_state_dict给入model中
model.load_state_dict(torch.load("model.pth"))