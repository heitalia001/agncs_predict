import torch
from torch.utils.data import DataLoader
from dataset import Dataset  
from models import SimpleCLIP  
import os
import getdata
import numpy as np

count_sql,label_sql, _, _ = getdata.Extract("./AgNCs/combined_data.xlsx","./AgNCs/bias.xlsx", 2).every()

count,label, _, _ = getdata.Extract("./AgNCs/combined_data.xlsx","./AgNCs/bias.xlsx", 3).every() #检验的50个
nucleotides = ['A', 'C', 'G', 'T', 'N', '-']  # DNA 的碱基及链接符号
#model = SimpleCLIP()#.to(CFG.device)
model = SimpleCLIP(image_model = "vgg",text_model = "linear")
model.load_state_dict(torch.load("model_epoch_200.pth"))
model.eval()

def one_hot_encode(dna_sequence):
        # 创建一个空的 One-Hot 编码矩阵
        one_hot = np.zeros((len(dna_sequence), len(nucleotides)), dtype=np.float32)

        # 填充 One-Hot 编码
        for i, nucleotide in enumerate(dna_sequence):
            if nucleotide in nucleotides:
                one_hot[i, nucleotides.index(nucleotide)] = 1

        # 转换为 PyTorch Tensor
        return torch.tensor(one_hot)

def acc_count(matrix1, matrix2, limits):
    # 获取两个矩阵的最大值及其位置
    max_pos_matrix1 = np.unravel_index(np.argmax(matrix1), matrix1.shape)
    max_pos_matrix2 = np.unravel_index(np.argmax(matrix2), matrix2.shape)
    
    # 距离是否小于或等于 limits
    if abs(max_pos_matrix1[0] - max_pos_matrix2[0]) <= limits and abs(max_pos_matrix1[1] - max_pos_matrix2[1]) <= limits:
        return True
    else:
        return False

# 反归一化函数
def denormalize_expm1(matrix):
    return np.array([[np.expm1(x) for x in row] for row in matrix])

# 判断两个最大值反归一化之后的最大值的值是否在20%以内
def acc_pow(matrix1, matrix2, tolerance=0.5):
    # 获取两个矩阵的最大值及其位置
    max_val_matrix1 = np.max(matrix1)
    max_val_matrix2 = np.max(matrix2)

    # 反归一化
    max_val_matrix1_denorm = np.expm1(max_val_matrix1)
    max_val_matrix2_denorm = np.expm1(max_val_matrix2)

    # 计算两个最大值的相对差异
    print(max_val_matrix1_denorm , max_val_matrix2_denorm)
    relative_difference = abs(max_val_matrix1_denorm - max_val_matrix2_denorm) / max_val_matrix2_denorm
    
    # 判断差异是否在20%以内
    if relative_difference <= tolerance:
        return True
    else:
        return False


corrct = 0
corrct_pow = 0
with torch.no_grad():
    for i in range(len(label)):
        text = label[i]
        text = one_hot_encode(text)
        text = text.unsqueeze(0)
        image = count[i]
        item = []
        resulridx = 0
        for j in range(len(count_sql)):
            count_tmp = count_sql[j]
            count_tmp = torch.tensor(count_tmp, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            print(count_tmp.shape, text.shape)
            image_embeddings, text_embeddings = model(count_tmp,text)
            # 计算相似度矩阵
            logits = image_embeddings @ text_embeddings.T
            item.append(logits)
            if max(item) == logits:
                resulridx = j
                if acc_count(count_sql[resulridx], image, 0):
                    corrct += 1
                    break

with torch.no_grad():
    for i in range(len(label)):
        text = label[i]
        text = one_hot_encode(text)
        text = text.unsqueeze(0)
        image = count[i]
        item = []
        resulridx = 0
        for j in range(len(count_sql)):
            count_tmp = count_sql[j]
            count_tmp = torch.tensor(count_tmp, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            image_embeddings, text_embeddings = model(count_tmp,text)
            # 计算相似度矩阵
            logits = image_embeddings @ text_embeddings.T
            item.append(logits)
            if max(item) == logits:
                resulridx = j
                if acc_pow(count_sql[resulridx], image,0.05):
                    corrct_pow += 1
                    break

print("corrct: ", corrct/len(label))
print("corrct_pow: ", corrct_pow/len(label))
        



            
