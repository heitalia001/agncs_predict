import getdata
import numpy as np
import torch

class Dataset:
    def __init__(self, path, base, flag):
        # 获取数据
        self.spectra, self.dna, self.max, self.min = getdata.Extract(path, base, flag).every()
        self.data_length = len(self.dna)  # 假设 dna 和 spectra 长度相同
        self.nucleotides = ['A', 'C', 'G', 'T', 'N', '-']  # DNA 的碱基及链接符号

    def __len__(self):
        return self.data_length

    def one_hot_encode(self, dna_sequence):
        # 创建一个空的 One-Hot 编码矩阵
        one_hot = np.zeros((len(dna_sequence), len(self.nucleotides)), dtype=np.float32)

        # 填充 One-Hot 编码
        for i, nucleotide in enumerate(dna_sequence):
            if nucleotide in self.nucleotides:
                one_hot[i, self.nucleotides.index(nucleotide)] = 1

        # 转换为 PyTorch Tensor
        return torch.tensor(one_hot)

    def __getitem__(self, idx):
        #print(f"Index: {idx}")  # 输出索引
        #print(f"DNA: {self.dna}, Spectra: {self.spectra}")  # 输出数据
        # 获取 DNA 链和对应的光谱热图
        dna_sequence = self.dna[idx]
        spectrum = self.spectra[idx]

        dna_one_hot = self.one_hot_encode(dna_sequence)
        spectrum_tensor = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0)

        return {'dna': dna_one_hot, 'spectrum': spectrum_tensor}  # 返回 One-Hot 编码的 DNA 链和光谱热图
