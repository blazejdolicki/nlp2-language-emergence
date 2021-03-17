import torch.nn as nn
import torch

class Receiver(nn.Module):
    def __init__(self, hidden_size, embed_size, img_clas):
        super(Receiver, self).__init__()
        self.img_embeddings = nn.Linear(embed_size, 2*hidden_size)
        self.msg_fc = nn.Linear(hidden_size, 2*hidden_size)
        self.img_clas_fc =  nn.Linear(hidden_size, 2*hidden_size)
        self.img_clas = img_clas

    def forward(self, hidden_state, imgs):
        # hidden_state is the message passed by the sender and is of shape (batch_size, hidden_size)
        # imgs is of shape (batch_size, num_imgs, embed_size)
        embed_imgs = self.img_embeddings(imgs).tanh() # (batch_size, num_imgs, hidden_size)
        
        embed_msg = self.msg_fc(hidden_state)
        embed_msg = torch.unsqueeze(embed_msg, dim=-1) # (batch_size, hidden_size, 1)

        # (batch_size, num_imgs, hidden_size) x (batch_size, hidden_size, 1) = (batch_size, num_imgs, 1)
        # because (num_imgs, hidden_size) x (hidden_size, 1) = (num_imgs, 1)
        sg_energies = torch.matmul(embed_imgs, embed_msg) 
        sg_energies = sg_energies.squeeze()
        
        if self.img_clas:
            img_clas_msg = self.img_clas_fc(hidden_state)
            img_clas_msg = torch.unsqueeze(img_clas_msg, dim=-1) # (batch_size, hidden_size, 1)
            
            img_clas_energies = torch.matmul(embed_imgs, img_clas_msg)
            img_clas_energies = img_clas_energies.squeeze()
            energies = torch.cat((sg_energies, img_clas_energies), dim=1)
        else:
            energies = sg_energies
            
         # (batch_size, 2*num_imgs)
            
        return energies # (batch_size, num_imgs)
               
