import torch.nn as nn
import torch.nn.functional as F

class Sender(nn.Module):
    def __init__(self, embed_size, num_imgs, hidden_sender, game_type="SenderReceiverRnnGS", vocab_size=100):
        super(Sender, self).__init__()
        """
            Note: embed_size is also the size of image features extracted from the vision model,
            so the shape of a batch from the SignalGameDataset() is going to be (batch_size, embed_size * num_imgs)
        """
        self.embed_size = embed_size
        self.num_imgs = num_imgs
        self.game_type = game_type
        
        if game_type != "SymbolGameReinforce": 
            self.fc = nn.Linear(embed_size, hidden_sender)
        else:
            self.fc = nn.Linear(embed_size, vocab_size)
        
    def forward(self, imgs):
        """
        In our setup, the sender only sees the target image. If we wanted to give both target image and distractors to the sender,
        we would replace `imgs.reshape(-1, self.embed_size)` with `imgs.reshape(-1, self.num_imgs*self.embed_size)`
        """
        imgs = imgs.reshape(-1, self.embed_size)
        x = self.fc(imgs) 
        if self.game_type == "SymbolGameReinforce":
            x = F.log_softmax(x, dim=1)
        else:
            x = x.tanh()
        return x
