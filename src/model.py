import torch, torch.nn as nn, os
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1=nn.Linear(input_size, hidden_size)
        self.linear2=nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x=torch.relu(self.linear1(x)); x=self.linear2(x); return x
    def save(self, file_name="model.pth"):
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(self.state_dict(), f"./checkpoints/{file_name}")
