'''
CLIP能力演示
对图片做分类
'''
from dataset import MNIST
import matplotlib.pyplot as plt 
import torch 
from clip import CLIP
import torch.nn.functional as F
import numpy as np

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

dataset=MNIST() # 数据集

model=CLIP().to(DEVICE) # 模型
model.load_state_dict(torch.load('model1.pth'))

model.eval()    # 预测模式

'''
1、对图片分类
'''
true = np.array([])
predict = np.array([])
for i in range(100):

    image,label=dataset[i]
    # print('正确分类:',label)
    true=np.append(true, label)
    # plt.imshow(image.permute(1,2,0))
    targets=torch.arange(0,10)  #10种分类
    logits=model(image.unsqueeze(0).to(DEVICE),targets.to(DEVICE)) # 1张图片 vs 10种分类
    # print(logits)
    # print('CLIP分类:',logits.argmax(-1).item())
    predict = np.append(predict, logits.argmax(-1).item())

print(true == predict)
