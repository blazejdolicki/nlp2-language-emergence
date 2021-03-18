import torch
import numpy as np
from tqdm import tqdm

class SignalGameDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_imgs, vision, task, classes=None, seed=1):
        self.dataset = dataset
        self.num_imgs = num_imgs
        self.vision = vision
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        np.random.seed(seed=seed)
        
        self.img_features, self.img_class_labels = self._extract_img_features() # (dataset_size, embed_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # get random images
        random_idxs = np.random.randint(low=0, high=self.__len__(), size=self.num_imgs)
        random_imgs = self.img_features[random_idxs]
        
        # get a random permutation of integers from 0 to num_imgs-1
        permutation = torch.randperm(self.num_imgs)

        # set the target image as the first random image
        sender_imgs = random_imgs[0].unsqueeze(dim=0)
        
        # permute random images for the receiver
        receiver_imgs = random_imgs[permutation]
        
        # set the label
        sg_target = permutation.argmin()
        
        if self.task == "standard":
            target = sg_target
        elif self.task=="img_clas":
            random_img_labels = self.img_class_labels[random_idxs]
            # image class label of the target image
            target_class = random_img_labels[0].squeeze()
            # array of booleans that are true if the class of a given image is the same as the class of the target image
            is_target_class = (random_img_labels[permutation] == target_class).float().squeeze()
            # concatenate the signaling game label with the multilabel binary image classification label
            target = torch.cat((sg_target.unsqueeze(dim=0), is_target_class)) 
        elif self.task=="target_clas":
            random_img_labels = self.img_class_labels[random_idxs]
            # image class label of the target image
            target_class = random_img_labels[0]
            # concatenate the signaling game label with the multiclass target image classification label 
            target = torch.cat((sg_target.unsqueeze(dim=0), target_class))
        else:
            assert False, "Wrong task name"
            

        
        return sender_imgs, target, receiver_imgs

    def _extract_img_features(self):
        """
            We have to have to extract image features by making a forward pass through the pretrained vision model.
            We can't do it for all images at once as there's too many of them and we would run out of memory,
            so we do it in batches.
        """

        # read images and labels from dataset into separate arrays
        imgs, img_class_labels = zip(*self.dataset)
        imgs = torch.stack(list(imgs)) # (dataset_size, 3, 32, 32)
        img_class_labels = torch.Tensor(img_class_labels).reshape(-1, 1) # (dataset_size, 1)
        
        # if you run out of memory or your laptop freezes, decrease this number and try again
        VISION_BATCH_SIZE = 1000
        vision_loader = torch.utils.data.DataLoader(imgs, shuffle=False, batch_size=VISION_BATCH_SIZE, num_workers=0)

        # extract features from images with a vision model
        img_features = []
        for img in tqdm(vision_loader, desc="Extracting features in batches"):
            img = img.to(self.device)
            with torch.no_grad():
                img_features.append(self.vision(img))
        img_features = torch.cat(img_features) # (dataset_size, 64)
        return img_features, img_class_labels

