import torch
from dataset import MNIST
from clip import CLIP
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# ======================
# 配置
# ======================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ITER_BATCH_COUNT = 100_000   # 迭代次数
BATCH_SIZE = 128             # 批次大小（稍微大一点，方便选10个数字）
TARGET_COUNT = 10            # 共10种数字
LR = 1e-3                    # 学习率
SAVE_INTERVAL = 1000         # 保存间隔
NUM_WORKERS = 0              # 数据加载的工作线程数，避免Windows中的多进程问题

# ======================
# 数据 & 模型
# ======================
dataset = MNIST()
dataloader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        persistent_workers=False)  # Windows下使用较低的 num_workers

model = CLIP().to(DEVICE)
# model.load_state_dict(torch.load('model.pth'))

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 用于记录损失
los = []
x = []

# ======================
# 训练循环
# ======================
if __name__ == '__main__':
        # 保证 batch 内覆盖到 10 个类别
    for it in range(ITER_BATCH_COUNT):
        while True:
            imgs, labels = next(iter(dataloader))
            if torch.unique(labels).shape[0] < TARGET_COUNT:  # 未覆盖10种数字
                continue
            # 挑选出10个数字
            target = set()
            indexes = []
            for j in range(BATCH_SIZE):
                if labels[j].item() in target:
                    continue
                target.add(labels[j].item())
                indexes.append(j)
                if len(target) == TARGET_COUNT:
                    break
            imgs = imgs[indexes]
            labels = labels[indexes]
            break

        # 前向传播
        logits = model(imgs.to(DEVICE), labels.to(DEVICE))

        # 构造 ground truth
        targets = torch.arange(0, TARGET_COUNT, device=DEVICE)

        # 对称 loss
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.permute(1, 0), targets)
        loss = (loss_i + loss_t) / 2

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 日志 & 保存
        if it % 1000 == 0:
            print(f"iter: {it}, loss: {loss.item():.4f}")
            torch.save(model.state_dict(), "model_tmp.pth")
            os.replace("model_tmp.pth", "model2.pth")

            # 保存损失值
            los.append(loss.item())
            x.append(it)

            # 保存损失曲线图
            plt.plot(x, los, label='Loss', color='r')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.savefig('loss.png')
            plt.close()  # 关闭当前图，防止内存泄漏


