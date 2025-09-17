from torch import nn 
import torch 
import torch.nn.functional as F

# 定义图像编码器
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化一个词嵌入层，将输入的整数映射为16维的嵌入向量
        # num_embeddings=10 表示词汇表大小为10，embedding_dim=16表示每个词的嵌入向量维度为16
        self.emb=nn.Embedding(num_embeddings=10,embedding_dim=16)
        # 第一层全连接层，输入维度16，输出维度64
        self.dense1=nn.Linear(in_features=16,out_features=64)

        # 第二层全连接层，输入维度64，输出维度16
        self.dense2=nn.Linear(in_features=64,out_features=16)

        # 第三层全连接层，输入维度16，输出维度8
        self.wt=nn.Linear(in_features=16,out_features=8)

        # 层归一化
        self.ln=nn.LayerNorm(8)

    # 定义前向传播函数
    def forward(self,x):
        # 通过词嵌入层，输入的x是整数，经过嵌入变成16维
        x=self.emb(x)

        # 第一层全连接层，经过ReLU激活函数
        x=F.relu(self.dense1(x))

        # 第二层全连接层，经过ReLU激活函数
        x=F.relu(self.dense2(x))

        # 经过第三层全连接层，输出8维的向量
        x=self.wt(x)

        # 通过层归一化，规范化输出
        x=self.ln(x)
        return x

if __name__=='__main__':
    text_encoder=TextEncoder()
    x=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    y=text_encoder(x)
    print(y.shape)