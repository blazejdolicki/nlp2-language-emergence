import torch.nn as nn

class Sender(nn.Module):
    def __init__(self, embed_size, num_imgs, hidden_sender):
        super(Sender, self).__init__()
        """
            Note: embed_size is also the size of image features extracted from the vision model,
            so the shape of a batch from the SignalGameDataset() is going to be (batch_size, embed_size * num_imgs)
        """
        self.embed_size = embed_size
        self.num_imgs = num_imgs
        self.fc = nn.Linear(embed_size, hidden_sender)
        
    def forward(self, imgs):
        """
        In our setup, the sender only sees the target image. If we wanted to give both target image and distractors to the sender,
        we would replace `imgs.reshape(-1, self.embed_size)` with `imgs.reshape(-1, self.num_imgs*self.embed_size)`
        """
        imgs = imgs.reshape(-1, self.embed_size)
        x = self.fc(imgs).tanh()
        return x
