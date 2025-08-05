import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNBackbone(nn.Module):
    def __init__(self, input_channels=1, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x

class AttentionMIL(nn.Module):
    def __init__(self, embedding_dim=128, num_families=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_families = num_families

        self.cnn_backbone = SimpleCNNBackbone(1, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_families)  

    
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, patches):
        batch_size, num_patches, C, H, W = patches.shape
        patches = patches.view(batch_size * num_patches, C, H, W)

        embeddings = self.cnn_backbone(patches) 
        embeddings = embeddings.view(batch_size, num_patches, self.embedding_dim)  

        
        instance_logits = self.classifier(embeddings) 

        
        attn_scores = self.attention(embeddings)  
        attn_scores = attn_scores.squeeze(-1)     
        attn_weights = F.softmax(attn_scores, dim=1)

        
        weighted_logits = attn_weights.unsqueeze(-1) * instance_logits  
        bag_logits = weighted_logits.sum(dim=1)  

        bag_probs = torch.sigmoid(bag_logits) 

        return bag_probs, instance_logits, attn_weights


if __name__ == "__main__":
    
    dummy_patches = torch.randn(2, 10, 1, 64, 32)

    model = AttentionMIL(embedding_dim=128, num_families=4)
    bag_probs, instance_logits, attn_weights = model(dummy_patches)

    print("Bag-level probabilities shape:", bag_probs.shape)       
    print("Instance logits shape:", instance_logits.shape)         
    print("Attention weights shape:", attn_weights.shape)           

    
    target = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)  
    loss_fn = nn.BCELoss()
    loss = loss_fn(bag_probs, target)
    print("Example loss:", loss.item())
