# mnist-clip
基于CLIP实现mnist数据集分类

基于mnist手写数字训练的clip模型，用作学习多模态模型的用途，只能预测0-9

## 模型

尝试过CLIP训练图文数据（coco等），loss收敛效果不好，可能是模型给的太复杂数据集太小导致的，所以就用mnist作为数据集用简单模型做一下了。

* img_encoder采用resnet残差网络结构，简单输出image embedding
* text_encoder没有用transformer，接收0~9数字ID，简单embedding+dense输出text embedding
* image embedding和text embedding做点积，得到logits，点积最大的(image,text)对最为相似

## loss图

loss.png表示仓库中的模型经过train.py后绘制的图像

loss2.png表示模型从0开始训练后绘制的图像
