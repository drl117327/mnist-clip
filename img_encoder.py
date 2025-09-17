from torch import nn 
import torch 
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=stride)
        self.bn1=nn.BatchNorm2d(out_channels)
        
        self.conv2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        
        self.conv3=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=0,stride=stride)
    
    def forward(self,x):
        # 将输入 x 传入第一个卷积层 self.conv1，进行卷积操作，之后通过批归一化 self.bn1 进行归一化，
        # 再通过 ReLU （f=max(0,x)）激活函数（通过 F.relu）进行非线性变换
        y=F.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        z=self.conv3(x)
        return F.relu(y+z)
        

class ImgEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个残差块：输入通道为1（灰度图像输入通道为1），输出16个通道，步幅为2
        self.res_block1=ResidualBlock(in_channels=1,out_channels=16,stride=2) # (batch,16,14,14)

        # 第二个残差快，输入通道是16，输出通道为4，步幅为2
        self.res_block2=ResidualBlock(in_channels=16,out_channels=4,stride=2) # (batch,4,7,7)

        # 第三个残差块，输入通道是4，输出通道是1，步幅为2
        self.res_block3=ResidualBlock(in_channels=4,out_channels=1,stride=2) # (batch,1,4,4)

        # 定义了一个线性层（Linear），输入特征数为16（来自第一个残差块的输出），输出特征数为8。
        # 这个层将特征图展平并通过全连接层进行变换。
        self.wi=nn.Linear(in_features=16,out_features=8)
        self.ln=nn.LayerNorm(8)
        
    def forward(self,x):
        x=self.res_block1(x)
        x=self.res_block2(x)
        x=self.res_block3(x)
        x=self.wi(x.view(x.size(0),-1))
        x=self.ln(x)
        return x
    
if __name__=='__main__':
    img_encoder=ImgEncoder()
    out=img_encoder(torch.randn(1,1,28,28))
    print(out.shape)