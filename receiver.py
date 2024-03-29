import torch.nn as nn
import torch

class Receiver(nn.Module):
    def __init__(self, hidden_size, embed_size, task, num_classes, game_type="SenderReceiverRnnGS", vocab_size=100):
        super(Receiver, self).__init__()
        self.task = task
        self.num_classes = num_classes
        self.game_type = game_type
        if game_type != "SymbolGameReinforce": 
            self.img_embeddings = nn.Linear(embed_size, 2*hidden_size)
            self.msg_fc = nn.Linear(hidden_size, 2*hidden_size)
            
            self.img_clas_fc = nn.Linear(hidden_size, 2*hidden_size)
            self.target_clas_fc = nn.Linear(hidden_size, num_classes)
        else:
            self.embed_symbol = nn.Embedding(vocab_size, embed_size)

    def forward(self, hidden_state, imgs):
        if self.game_type == "SymbolGameReinforce":
            # hidden_state is the message passed by the sender and is of shape (batch_size, 1)
            # imgs is of shape (batch_size, num_imgs, embed_size)
            
            # message = self.embed_symbol(hidden_state) # (batch_size, embed_size)
            
            # (batch_size, num_imgs, embed_size) x (batch_size, embed_size, 1) = (batch_size, num_imgs, 1)
            # because (num_imgs, embed_size) x (embed_size, 1) = (num_imgs, 1)
            sg_energies = torch.matmul(imgs, torch.unsqueeze(hidden_state, dim=-1))
            sg_energies = sg_energies.squeeze() # (batch_size, num_imgs)
        else: 
            # hidden_state is the message passed by the sender and is of shape (batch_size, hidden_size)
            # imgs is of shape (batch_size, num_imgs, embed_size)
            
            embed_imgs = self.img_embeddings(imgs).tanh() # (batch_size, num_imgs, 2*hidden_size)
            
            embed_msg = self.msg_fc(hidden_state) # (batch_size, 2*hidden_size, 1)
            embed_msg = torch.unsqueeze(embed_msg, dim=-1) # (batch_size, 2*hidden_size, 1)

            # (batch_size, num_imgs, 2*hidden_size) x (batch_size, 2*hidden_size, 1) = (batch_size, num_imgs, 1)
            # because (num_imgs, 2*hidden_size) x (2*hidden_size, 1) = (num_imgs, 1)
            sg_energies = torch.matmul(embed_imgs, embed_msg) # (batch_size, num_imgs, 1)
            sg_energies = sg_energies.squeeze() # (batch_size, num_imgs)
        
        if self.task=="standard":
            output = sg_energies 
        elif self.task=="img_clas":
            img_clas_msg = self.img_clas_fc(hidden_state) # (batch_size, 2*hidden_size)
            img_clas_msg = torch.unsqueeze(img_clas_msg, dim=-1) # (batch_size, 2*hidden_size, 1)
            
            img_clas_energies = torch.matmul(embed_imgs, img_clas_msg) # (batch_size, num_imgs, 1)
            img_clas_energies = img_clas_energies.squeeze() # (batch_size, num_imgs)
            output = torch.cat((sg_energies, img_clas_energies), dim=1) # (batch_size, num_imgs + num_imgs)
        elif self.task=="target_clas":
            target_clas_msg = self.target_clas_fc(hidden_state) # (batch_size, num_classes)
            output = torch.cat((sg_energies, target_clas_msg), dim=1) # (batch_size, num_imgs + num_classes)
        else:
            assert False, "Wrong task."
            
            
        return output # (batch_size, num_imgs)
               
