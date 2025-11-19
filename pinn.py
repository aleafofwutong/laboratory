import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 1. 配置参数与设备 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = 0.01  # 热传导扩散系数
epochs = 5000  # 训练轮数
lr = 1e-3      # 学习率

# -------------------------- 2. 生成训练数据（边界/初始条件 + 物理残差采样点）--------------------------
def generate_training_data():
    # 2.1 初始条件数据（t=0，x∈[0,1]）：u(x,0)=sin(πx)
    x_initial = torch.linspace(0, 1, 100).unsqueeze(1).to(device)  # (100,1)
    t_initial = torch.zeros_like(x_initial).to(device)  # (100,1)
    u_initial = torch.sin(np.pi * x_initial).to(device)  # 初始温度分布（真实标签）
    
    # 2.2 边界条件数据（x=0 或 x=1，t∈[0,1]）：u(0,t)=0，u(1,t)=0
    t_boundary = torch.linspace(0, 1, 100).unsqueeze(1).to(device)  # (100,1)
    x0_boundary = torch.zeros_like(t_boundary).to(device)  # x=0
    x1_boundary = torch.ones_like(t_boundary).to(device)   # x=1
    u0_boundary = torch.zeros_like(t_boundary).to(device)  # 边界温度=0
    u1_boundary = torch.zeros_like(t_boundary).to(device)
    
    # 2.3 物理残差采样点（随机采样 x∈[0,1]、t∈[0,1]，用于计算 PDE 残差）
    x_pde = torch.rand(500, 1).to(device)  # (500,1) 随机位置
    t_pde = torch.rand(500, 1).to(device)  # (500,1) 随机时间
    x_pde.requires_grad = True  # 开启自动微分（需计算对 x 的二阶导数）
    t_pde.requires_grad = True  # 开启自动微分（需计算对 t 的一阶导数）
    
    # 整理所有训练数据
    data = {
        "initial": (x_initial, t_initial, u_initial),
        "boundary": (torch.cat([x0_boundary, x1_boundary]), 
                     torch.cat([t_boundary, t_boundary]), 
                     torch.cat([u0_boundary, u1_boundary])),
        "pde": (x_pde, t_pde)
    }
    return data

training_data = generate_training_data()

# -------------------------- 3. 定义 PINN 模型（简单全连接网络）--------------------------
class PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1, num_layers=4):
        super(PINN, self).__init__()
        # 构建网络层：输入(x,t) → 隐藏层 → 输出u(x,t)（温度）
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers-2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, t):
        # 输入拼接：(x,t) → (batch_size, 2)，修复形状问题
        input = torch.cat([x, t], dim=1)  # 要求 x 和 t 都是 (N,1) 形状，拼接后为 (N,2)
        return self.model(input)  # 输出预测温度 u(x,t)

# 初始化模型
model = PINN().to(device)

# -------------------------- 4. 定义双损失函数（数据损失 + 物理损失）--------------------------
def compute_loss(model, data):
    # 4.1 数据损失：拟合初始条件和边界条件
    x_initial, t_initial, u_initial = data["initial"]
    x_boundary, t_boundary, u_boundary = data["boundary"]
    
    # 模型预测
    u_pred_initial = model(x_initial, t_initial)
    u_pred_boundary = model(x_boundary, t_boundary)
    
    # MSE 损失（拟合真实标签）
    loss_data_initial = nn.MSELoss()(u_pred_initial, u_initial)
    loss_data_boundary = nn.MSELoss()(u_pred_boundary, u_boundary)
    loss_data = loss_data_initial + loss_data_boundary  # 总数据损失
    
    # 4.2 物理损失：强制满足 PDE（残差最小化）
    x_pde, t_pde = data["pde"]
    u_pred_pde = model(x_pde, t_pde)
    
    # 自动微分计算 PDE 所需导数（核心！无需手动推导）
    # 一阶导数：∂u/∂t
    du_dt = torch.autograd.grad(
        outputs=u_pred_pde,
        inputs=t_pde,
        grad_outputs=torch.ones_like(u_pred_pde),
        create_graph=True,  # 保留计算图，用于二阶导数
        retain_graph=True
    )[0]
    
    # 一阶导数：∂u/∂x
    du_dx = torch.autograd.grad(
        outputs=u_pred_pde,
        inputs=x_pde,
        grad_outputs=torch.ones_like(u_pred_pde),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 二阶导数：∂²u/∂x²
    d2u_dx2 = torch.autograd.grad(
        outputs=du_dx,
        inputs=x_pde,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True
    )[0]
    
    # PDE 残差：residual = ∂u/∂t - α∂²u/∂x²（理想情况下残差=0）
    residual = du_dt - alpha * d2u_dx2
    loss_pde = nn.MSELoss()(residual, torch.zeros_like(residual))  # 残差MSE损失
    
    # 总损失：数据损失 + 物理损失（可调整权重平衡）
    total_loss = loss_data + loss_pde
    return total_loss, loss_data, loss_pde

# -------------------------- 5. 模型训练 --------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 训练日志
loss_history = {"total": [], "data": [], "pde": []}

for epoch in range(epochs):
    optimizer.zero_grad()
    total_loss, loss_data, loss_pde = compute_loss(model, training_data)
    
    # 反向传播与参数更新
    total_loss.backward()
    optimizer.step()
    
    # 记录损失
    loss_history["total"].append(total_loss.item())
    loss_history["data"].append(loss_data.item())
    loss_history["pde"].append(loss_pde.item())
    
    # 每 500 轮打印一次
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss.item():.6f}, "
              f"Data Loss: {loss_data.item():.6f}, PDE Loss: {loss_pde.item():.6f}")

# -------------------------- 6. 结果可视化（修复形状不匹配问题）--------------------------
def plot_results(model):
    # 生成测试网格（x∈[0,1]，t∈[0,1]）
    x_test = torch.linspace(0, 1, 100).unsqueeze(1).to(device)  # (100,1)
    t_test = torch.linspace(0, 1, 100).unsqueeze(1).to(device)  # (100,1)
    
    # 生成网格点：用 stack + reshape 确保 x 和 t 都是 (10000,1) 形状（适配模型输入）
    x_grid, t_grid = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing="ij")  # (100,100)
    x_grid = x_grid.unsqueeze(2).reshape(-1, 1)  # (10000,1)：展平为批量样本
    t_grid = t_grid.unsqueeze(2).reshape(-1, 1)  # (10000,1)：展平为批量样本
    
    # 模型预测
    model.eval()
    with torch.no_grad():
        u_pred = model(x_grid, t_grid).cpu().numpy()  # (10000,1)
        u_pred = u_pred.reshape(100, 100)  # 恢复为 (100,100) 网格用于绘图
    
    # 解析解（热传导方程的理论解）
    x_analytic = torch.linspace(0, 1, 100).numpy()
    t_analytic = torch.linspace(0, 1, 100).numpy()
    X, T = np.meshgrid(x_analytic, t_analytic, indexing="ij")
    u_analytic = np.sin(np.pi * X) * np.exp(-alpha * np.pi**2 * T)  # (100,100)
    
    # 绘制 2D 热力图（温度分布）
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 解析解
    im1 = ax1.imshow(u_analytic, extent=[0,1,0,1], origin="lower", cmap="hot")
    ax1.set_title("Analytic Solution (Ground Truth)")
    ax1.set_xlabel("x (Position)")
    ax1.set_ylabel("t (Time)")
    plt.colorbar(im1, ax=ax1)
    
    # 预测解
    im2 = ax2.imshow(u_pred, extent=[0,1,0,1], origin="lower", cmap="hot")
    ax2.set_title("PINN Prediction")
    ax2.set_xlabel("x (Position)")
    plt.colorbar(im2, ax=ax2)
    
    # 误差图
    error = np.abs(u_pred - u_analytic)
    im3 = ax3.imshow(error, extent=[0,1,0,1], origin="lower", cmap="viridis")
    ax3.set_title("Absolute Error")
    ax3.set_xlabel("x (Position)")
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.show()
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history["total"], label="Total Loss")
    plt.plot(loss_history["data"], label="Data Loss")
    plt.plot(loss_history["pde"], label="PDE Loss")
    plt.yscale("log")  # 对数坐标，更清晰
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.title("Loss History During Training")
    plt.show()

# 执行可视化
plot_results(model)