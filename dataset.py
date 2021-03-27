import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

class SignalGameDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_imgs, vision, task, classes=None, same_class_prob=0.0, seed=1):
        self.dataset = dataset
        self.num_imgs = num_imgs
        self.vision = vision
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.same_class_prob = same_class_prob
        np.random.seed(seed=seed)

        self.img_features, self.img_class_labels = self._extract_img_features() # (dataset_size, embed_size)
        self._create_class_index()

    def __len__(self):
        return len(self.dataset)

    def randint_exclude(self, target_idx):
        """
        Get distractor indices which are different than target index.
        """
        random_idxs = []

        # sample distractors until you get enough
        while len(random_idxs) != self.num_imgs-1:
            random_float = np.random.random_sample()

            if random_float < self.same_class_prob: # sample distractor with the same class as target
                target_class_label = self.img_class_labels[target_idx]
                # sample an index from all indices from a given class
                random_idx = np.random.choice(self.class_idxs[target_class_label.item()], size=None)
            else: # sample distractor with any class
                random_idx = np.random.randint(low=0, high=self.__len__())

            # if the distractor index is different than target, add it to the list of indices
            if random_idx != target_idx:
                random_idxs.append(random_idx)

        return random_idxs


    def __getitem__(self, item):

        # get random images
        random_idxs = [item] + self.randint_exclude(item)
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

    def _create_class_index(self):
        self.class_idxs = defaultdict(list)
        for i, label in enumerate(self.img_class_labels):
            self.class_idxs[label.item()].append(i)


class GaussianNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, num_imgs, vision, task, dataset_size, classes=None, seed=1, im_size=(3, 32, 32), mean=0.0, std_dev=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = mean
        self.std_dev = std_dev
        self.num_imgs = num_imgs
        self.im_size = im_size
        self.dataset_size = dataset_size
        self.task = task
        self.dataset = self._init_dataset()


        self.vision = vision

        np.random.seed(seed=seed)
        self.img_features = self._extract_img_features() # (dataset_size, embed_size)

    def __len__(self):
        return len(self.dataset)
        
    def randint_exclude(self, target_idx):
        """
        Get distractor indices which are different than target index.
        """
        random_idxs = []

        # sample distractors until you get enough
        while len(random_idxs) != self.num_imgs-1:
            random_float = np.random.random_sample()

            random_idx = np.random.randint(low=0, high=self.__len__())

            # if the distractor index is different than target, add it to the list of indices
            if random_idx != target_idx:
                random_idxs.append(random_idx)

        return random_idxs


    def __getitem__(self, item):
        # get random images
        random_idxs = [item] + self.randint_exclude(item)
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
            is_target_class = torch.zeros(2)
            target = torch.cat((sg_target.unsqueeze(dim=0), is_target_class))
        elif self.task=="target_clas":
            target_class = torch.zeros(1).float()
            target = torch.cat((sg_target.unsqueeze(dim=0), target_class))
        else:
            assert False, "Wrong task name"

        return sender_imgs, target, receiver_imgs

    def _init_dataset(self):
        samples = []
        for i in range(self.dataset_size):
            noise_vec = np.random.normal(loc=self.mean, scale=self.std_dev, size=self.im_size)
            # What label to give to these noise vectors?
            samples.append((torch.from_numpy(noise_vec).float(), 0))
        return samples

    def _extract_img_features(self):
        """
            We have to extract image features by making a forward pass through the pretrained vision model.
            We can't do it for all images at once as there's too many of them and we would run out of memory,
            so we do it in batches.
        """

        # read images from dataset into a single array
        imgs = [img for img, label in self.dataset]
        imgs = torch.stack(imgs)

        # if you run out of memory or your laptop freezes, decrease this number and try again
        VISION_BATCH_SIZE = 1000
        vision_loader = torch.utils.data.DataLoader(imgs, shuffle=False, batch_size=VISION_BATCH_SIZE, num_workers=0)

        # extract features from images with a vision model
        img_features = []
        for img in tqdm(vision_loader, desc="Extracting features in batches"):
            img = img.to(self.device)
            with torch.no_grad():
                img_features.append(self.vision(img))
        img_features = torch.cat(img_features)
        return img_features
