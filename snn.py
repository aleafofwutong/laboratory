import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import spikegen
from torchvision import datasets, transforms

# 1. 配置设备（自动识别GPU/CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 2. 定义脉冲神经网络（类脑分类器）
class SpikingCNN(nn.Module):
    def __init__(self, num_steps=20):
        super().__init__()
        self.num_steps = num_steps
        
        # 脉冲神经元层（LIF模型，类脑核心）
        self.fc1 = nn.Linear(28*28, 128)  # 输入：28*28（MNIST图像）
        self.lif1 = snn.Leaky(beta=0.9, threshold=1.0, reset_mechanism="subtract")
        
        self.fc2 = nn.Linear(128, 10)  # 输出：10类（0-9数字）
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x):
        # 输入形状：(batch_size, 1, 28, 28) → 展平为 (batch_size, 28*28)
        batch_size = x.shape[0]  # 动态获取当前batch_size（避免固定值不灵活）
        x = x.view(batch_size, -1)  # 形状：(32, 784)
        
        # 脉冲编码：强制输出 (num_steps, batch_size, input_dim)
        x_spike = spikegen.rate(x, num_steps=self.num_steps)
        # 强制维度顺序：确保第0维是num_steps，第1维是batch_size
        if x_spike.shape[1] != batch_size:
            x_spike = x_spike.permute(1, 0, 2)  # 调整为 (20, 32, 784)
        # 再次校验：若仍不对，直接reshape（终极保障）
        x_spike = x_spike.reshape(self.num_steps, batch_size, 28*28)
        
        # 初始化神经元状态（动态适配batch_size）
        mem1 = torch.zeros(batch_size, 128).to(device)  # (32, 128)
        mem2 = torch.zeros(batch_size, 10).to(device)   # (32, 10)
        
        # 收集每个时间步的输出脉冲
        spk2_list = []
        for step in range(self.num_steps):
            # 取当前时间步输入：(32, 784)
            x_step = x_spike[step].to(device)
            
            # 第一层：全连接 + LIF神经元
            cur1 = self.fc1(x_step)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # 第二层：全连接 + LIF神经元
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_list.append(spk2)
        
        # 计算脉冲发放率（确保输出形状为(batch_size, 10)）
        spk2_stack = torch.stack(spk2_list, dim=0)  # (20, 32, 10)
        output = spk2_stack.sum(dim=0) / self.num_steps  # (32, 10)
        
        # 最终维度校验（避免隐患）
        assert output.shape == (batch_size, 10), f"输出维度错误：{output.shape}，预期({batch_size},10)"
        return output

# 3. 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载并加载数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 数据加载器（batch_size=32）
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

# 4. 初始化模型、损失函数、优化器
model = SpikingCNN(num_steps=20).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. 训练模型（类脑网络突触权重学习）
model.train()
num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播+参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 每100个batch打印进度
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

# 6. 测试模型准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f'\n测试准确率: {accuracy:.2f}%')