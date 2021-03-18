from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import os
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy.spatial.distance as distance
import scipy.stats
import scipy
import egg.core as core
import egg.zoo as zoo

import types
import json

from vision import Vision, BasicBlock
from dataset import SignalGameDataset
from sender import Sender
from receiver import Receiver
from loss import signal_game_loss, ImageClasLoss, TargetClasLoss

import argparse
from argparse import Namespace



# Parse command line arguments that vary between runs
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, choices=["standard", "img_clas", "target_clas"], default="standard",
                    help="""Choose which tasks we optimize for: 
                            `standard` is the vanilla signaling game
                            `img_clas` means the standard task and the task of predicting for each image 
                                       if it is the same class as the target image
                            `target_clas` means the standard task and the task of predicting class of the target image""")

parser.add_argument('--ic_loss_weight', type=float, default=1.0, 
                    help='Weight assigned to the image classification loss.')
parser.add_argument('--num_imgs', type=int, default=2, 
                    help='Number of images used in the game (number of distractors + 1)')

cmd_args = parser.parse_args()

_args_dict = {
    "architecture" : {
        "embed_size"      : 64,
        "hidden_sender"   : 200,
        "hidden_receiver" : 200,
        "cell_type"       : 'gru',
    },
    "training" : {
        "temperature"     : 1,
        "decay"           : 0.9,
        "early_stop_accuracy" : 0.97,
    },
}

# A trick for having a hierarchical argument namespace from the above dict
fixed_args = json.loads(json.dumps(_args_dict), object_hook=lambda item: types.SimpleNamespace(**item))

args = Namespace(**vars(cmd_args), **vars(fixed_args))
    

# For convenience and reproducibility, we set some EGG-level command line arguments here
opts = core.init(params=['--random_seed=7', # will initialize numpy, torch, and python RNGs
                                   '--lr=1e-3', # sets the learning rate for the selected optimizer 
                                   '--batch_size=64',
                                   '--vocab_size=100',
                                   '--max_len=10',
                                   '--n_epochs=15',
                                   '--tensorboard',
                                   ]) 

print("Parameters specified in the command line:") 
print("Image classification task:", args.task)
print("Image classification loss weight:", args.ic_loss_weight)
print("Number of images in the game:", args.num_imgs)
print()


print("Cell type of the agents:", args.architecture.cell_type)

# TODO: other configurations?

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# load pre-trained parameters
num_classes = 100
restnet_location = "https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet56-2f147f26.pth"
vision = Vision(BasicBlock, [9, 9, 9], num_classes=num_classes).to(device)
vision.load_state_dict(model_zoo.load_url(restnet_location))

vision.eval()

transform = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4865, 0.4409],
        std=[0.2009, 0.1984, 0.2023]
    ),
])

cifar_train_set = datasets.CIFAR100('./data', train=True, download=True, transform=transform)

cifar_test_set = datasets.CIFAR100('./data', train=False, transform=transform)

print("Extract image features from train set")
trainset = SignalGameDataset(cifar_train_set, args.num_imgs, vision, task=args.task)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,
                                          batch_size=opts.batch_size, num_workers=0)

print("Extract image features from test set")
testset = SignalGameDataset(cifar_test_set, args.num_imgs, vision, task=args.task)
testloader = torch.utils.data.DataLoader(testset, shuffle=False,
                                         batch_size=opts.batch_size, num_workers=0)


sender = Sender(args.architecture.embed_size, args.num_imgs, args.architecture.hidden_sender)
sender = core.RnnSenderGS(sender, opts.vocab_size, args.architecture.embed_size, 
                          args.architecture.hidden_sender, cell=args.architecture.cell_type, max_len=opts.max_len, 
                          temperature=args.training.temperature, straight_through=True)

receiver = Receiver(args.architecture.hidden_receiver, args.architecture.embed_size, args.task, num_classes)
receiver = core.RnnReceiverGS(receiver, opts.vocab_size, args.architecture.embed_size, 
                              args.architecture.hidden_receiver, cell=args.architecture.cell_type)


    
model_prefix = f"maxlen_{opts.max_len}" # Example
models_path = "/content/drive/My Drive/SignalGame/models" # location where we store trained models

checkpointer = core.callbacks.CheckpointSaver(checkpoint_path=models_path, checkpoint_freq=0, prefix=model_prefix)

if args.task=="standard":
    loss = signal_game_loss
elif args.task=="img_clas":
    loss = ImageClasLoss(args.ic_loss_weight, args.num_imgs).get_loss
elif args.task=="target_clas":
    loss = TargetClasLoss(args.ic_loss_weight, args.num_imgs).get_loss
else:
    assert False, "Wrong task"
    
game = core.SenderReceiverRnnGS(sender, receiver, loss)
optimizer = torch.optim.Adam(game.parameters())

callbacks = [core.TemperatureUpdater(agent=game.sender, decay=args.training.decay, minimum=0.1),
             core.ConsoleLogger(as_json=True, print_train_loss=True),
             core.TensorboardLogger(),
             core.EarlyStopperAccuracy(args.training.early_stop_accuracy),
             checkpointer]

trainer = core.Trainer(game=game, optimizer=optimizer, train_data=trainloader,
                       validation_data=testloader, callbacks=callbacks)

trainer.train(n_epochs=opts.n_epochs)