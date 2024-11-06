import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Dataset  # 假设你的 Dataset 类在 dataset.py 中
from models import SimpleCLIP  # 导入你自己实现的 CLIP 模型
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 配置参数
class CFG:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32 #越大越好
    learning_rate = 1e-4
    num_epochs = 200
    path = "./AgNCs/combined_data.xlsx"  # 数据路径
    base = "./AgNCs/bias.xlsx"  # 替换为实际的 base 信息
    flag = 1  # 替换为实际的 flag 信息

# 绘制热图的函数
def plot_heatmap(similarity_matrix, batch, save_path, batch_idx):
    # 获取批次中的索引作为标签
    batch_size = similarity_matrix.shape[0]
    labels = list(range(batch_size))

    # 找到每行最大值的位置
    max_indices = np.argmax(similarity_matrix, axis=1)

    # 创建一个新的矩阵，只保留每行最大值的位置
    new_matrix = np.zeros_like(similarity_matrix)
    new_matrix[np.arange(batch_size), max_indices] = 1  # 将最大值位置设置为1

    # 使用 seaborn 绘制热图
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(new_matrix, cmap="Blues", xticklabels=labels, yticklabels=labels, annot=False, cbar=False, square=True, linewidths=.5, linecolor='white')

    # 设置背景为白色
    ax.set_facecolor('white')

    plt.title(f"Similarity Matrix Heatmap for Batch {batch_idx}")
    plt.xlabel("Predicted Indices")
    plt.ylabel("True Indices")

    # 保存热图
    save_file = os.path.join(save_path, f"heatmap_batch_{batch_idx}.png")
    plt.savefig(save_file)
    plt.close()

def test_epoch(model, test_loader):
    model.eval()
    total_loss = 0.0  # 用于累加损失
    correct_predictions = 0  # 用于计算准确率
    total_predictions = 0  # 用于总预测数
    tqdm_object = tqdm(test_loader, total=len(test_loader))

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm_object):
            # 将数据移到 GPU
            dna = batch['dna'].to(CFG.device)
            spectrum = batch['spectrum'].to(CFG.device)

            # 前向传播
            outputs = model(spectrum, dna)

            # 计算损失
            loss = model.compute_loss(outputs[0],outputs[1])

            # 累加损失
            total_loss += loss.item()

             # 计算相似度
            logits = (outputs[0] @ outputs[1].T)  # (batch_size, batch_size)
            preds = logits.argmax(dim=1)  # 预测的文本索引


            # 计算正确预测的数量
            correct_predictions += (preds == torch.arange(len(outputs[0])).to(CFG.device)).sum().item()
            total_predictions += len(outputs[0])

            # 更新进度条
            tqdm_object.set_description(f"Loss: {loss.item():.4f}, Accuracy: {correct_predictions / total_predictions:.4f}")
            # 绘制热图并保存
            if batch_idx == 0:  # 只绘制第一个批次的热图
                plot_heatmap(logits.cpu().numpy(), batch, './', batch_idx)

    # 计算平均损失和准确率
    avg_loss = total_loss / total_predictions
    accuracy = correct_predictions / total_predictions

    # 打印结果
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    

def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0  # 用于累加损失
    correct_predictions = 0  # 用于计算准确率
    total_predictions = 0  # 用于总预测数
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        optimizer.zero_grad()

        # 将数据移到 GPU
        dna = batch['dna'].to(CFG.device)
        spectrum = batch['spectrum'].to(CFG.device)

        # 模型前向传播
        outputs = model(spectrum, dna)
        loss = model.compute_loss(outputs[0],outputs[1])  # 根据模型返回的损失计算

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # 累加损失

         # 计算相似度
        logits = (outputs[0] @ outputs[1].T)  # (batch_size, batch_size)
        preds = logits.argmax(dim=1)  # 预测的文本索引

        # 计算正确预测的数量
        correct_predictions += (preds == torch.arange(len(outputs[0])).to(CFG.device)).sum().item()
        total_predictions += len(outputs[0])

        tqdm_object.set_postfix(train_loss=total_loss / (len(tqdm_object)), accuracy=correct_predictions / total_predictions)

    return total_loss / len(train_loader), correct_predictions / total_predictions  # 返回平均损失和准确率



def main():
    # 初始化数据集和数据加载器
    dataset = Dataset(CFG.path, CFG.base, CFG.flag)
    train_loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True)

    test_dataset = Dataset("./AgNCs/combined_data.xlsx" , CFG.base, 2)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=True)
    # 初始化自定义 CLIP 模型
    model = SimpleCLIP(image_model = "vgg",text_model = "linear").to(CFG.device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate)

    for epoch in range(CFG.num_epochs):
        print(f"Epoch {epoch + 1}/{CFG.num_epochs}")
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        # 保存模型
    model_save_path = os.path.join('./', f'model_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
    test_epoch(model, test_loader)

if __name__ == "__main__":
    main()
