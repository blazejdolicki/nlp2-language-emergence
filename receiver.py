import torch.nn as nn
import torch

class Receiver(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(embed_size, hidden_size)

    def forward(self, hidden_state, imgs):
        # hidden_state is the message passed by the sender and is of shape (batch_size, hidden_size)
        # imgs is of shape (batch_size, num_imgs, embed_size)
        embed_imgs = self.fc(imgs).tanh() # (batch_size, num_imgs, hidden_size)
        
        hidden_state = torch.unsqueeze(hidden_state, dim=-1) # (batch_size, hidden_size, 1)

        # (batch_size, num_imgs, hidden_size) x (batch_size, hidden_size, 1) = (batch_size, num_imgs, 1)
        # because (num_imgs, hidden_size) x (hidden_size, 1) = (num_imgs, 1)
        energies = torch.matmul(embed_imgs, hidden_state) 

        return energies.squeeze() # (batch_size, num_imgs)
