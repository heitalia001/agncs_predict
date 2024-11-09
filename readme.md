本项目是一个基于深度学习的 DNA 链分类系统，结合了图像编码器和文本编码器的双模态模型，
使用 PyTorch 实现，并支持定制化的 CLIP 架构。


项目结构如下：
├── AgNCs/                     # 训练的资源和数据
├── log/                       # 训练和评估过程中的日志文件 
├── dataset.py                 # 数据集加载和预处理代码
├── eval.py                    # 评估模型的脚本
├── getdata.py                 # 获取数据的工具函数
├── heatmap_batch_0.png        # 热图示例
├── models.py                  # 定义深度学习模型的代码
├── result.xlsx                # 评估结果存储表
└── train.py                   # 模型训练脚本

环境要求
Python 3.8
PyTorch 2.2


使用说明
1. 准备数据
运行以下命令获取数据：

bash:
cd AgNCs && python deal.py
2. 训练模型
运行以下命令开始训练模型：

bash:
python train.py
3. 评估模型
训练完成后，可使用以下命令评估模型性能：

bash:
python eval.py
