import torch
import torch.nn as nn
from torchvision import models


class SimpleVGG(nn.Module):
    def __init__(self, embedding_dim=512):
        super(SimpleVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (64, 10, 10)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (128, 5, 5)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (256, 2, 2)

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (512, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平
        x = self.classifier(x)
        return x
        
class SimpleCLIP(nn.Module):
    def __init__(self, 
                 image_embedding_dim=512, 
                 text_embedding_dim=512, 
                 num_classes=6*43, 
                 image_model='resnet50',  # 可选值: 'resnet50', 'resnet101', 'vgg', 'conv'
                 text_model='transformer'):  # 可选值: 'transformer', 'linear'
        super(SimpleCLIP, self).__init__()

        # 图像编码器选择
        self.image_model = image_model
        self.input_image = nn.Conv2d(1, 3, kernel_size=1)  # 将单通道图像转换为三通道
        if image_model == 'resnet50':
            self.image_encoder = models.resnet50(pretrained=True)
            self.image_projection = nn.Linear(self.image_encoder.fc.in_features, image_embedding_dim)
            self.image_encoder.fc = nn.Identity()  # 禁用原有的全连接层
        elif image_model == 'resnet101':
            self.image_encoder = models.resnet101(pretrained=True)
            self.image_projection = nn.Linear(self.image_encoder.fc.in_features, image_embedding_dim)
            self.image_encoder.fc = nn.Identity()
        elif image_model == 'vgg':
            self.image_encoder = SimpleVGG(embedding_dim=image_embedding_dim)
            
        elif image_model == 'conv':
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(64 *100, image_embedding_dim)  # 假设输入为224x224图像
            )
        else:
            raise ValueError(f"Unknown image model: {image_model}")

        # 文本编码器选择
        if text_model == 'transformer':
            self.text_embedding = nn.Embedding(num_classes, text_embedding_dim)  # 嵌入层
            encoder_layers = nn.TransformerEncoderLayer(d_model=text_embedding_dim, nhead=8)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
            self.text_projection = nn.Linear(text_embedding_dim, text_embedding_dim)  # 确保输出维度一致
        elif text_model == 'linear':
            self.text_projection = nn.Linear(num_classes, text_embedding_dim)
        else:
            raise ValueError(f"Unknown text model: {text_model}")

    def forward(self, images, one_hot_texts):
        # 图像编码
        images = self.input_image(images)
        image_embeddings = self.image_encoder(images)
        if self.image_model != 'vgg' and self.image_model != 'conv':
            image_embeddings = self.image_projection(image_embeddings)
        # 文本编码
        one_hot_texts = one_hot_texts.view(one_hot_texts.size(0), -1)
        if hasattr(self, 'transformer_encoder'):
            text_embeddings = self.text_embedding(one_hot_texts.argmax(dim=1))  # 通过 one-hot 获取嵌入
            text_embeddings = self.transformer_encoder(text_embeddings.unsqueeze(1))  # 添加序列维度
            text_embeddings = text_embeddings.squeeze(1)  # 移除序列维度
            text_embeddings = self.text_projection(text_embeddings)  # 确保输出维度一致
        else:
            text_embeddings = self.text_projection(one_hot_texts)

        return image_embeddings, text_embeddings

    def compute_loss(self, image_embeddings, text_embeddings):
        # 计算对比损失
        logits = image_embeddings @ text_embeddings.T
        labels = torch.arange(logits.size(0)).to(logits.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss
