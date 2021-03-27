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
from dataset import SignalGameDataset, GaussianNoiseDataset
from sender import Sender
from receiver import Receiver
from loss import signal_game_loss, ImageClasLoss, TargetClasLoss

import argparse
from argparse import Namespace
from best_epoch_checkpoint import BestEpochCheckpointSaver


# Parse command line arguments that vary between runs
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, choices=["standard", "img_clas", "target_clas"], default="standard",
                    help="""Choose which tasks we optimize for:
                            `standard` is the vanilla signaling game
                            `img_clas` means the standard task and the task of predicting for each image
                                       if it is the same class as the target image
                            `target_clas` means the standard task and the task of predicting class of the target image.""")

parser.add_argument('--ic_loss_weight', type=float, default=1.0,
                    help='Weight assigned to the image classification loss.')
parser.add_argument('--num_imgs', type=int, default=2,
                    help='Number of images used in the game (number of distractors + 1).')
parser.add_argument("--same_class_prob", type=float, default=0.0,
                    help="Probability that a distractor will have the same image class as the target image.")
parser.add_argument("--seed", type=int, default=7, help="Random seed.")
parser.add_argument("--eval_noise", action='store_true',
                    help="Set this flag if you want to evaluate the trained model on Gaussian noise")
parser.add_argument("--game_type", type=str, choices=["SenderReceiverRnnGS", "SenderReceiverRnnReinforce", "SymbolGameReinforce"], default="SenderReceiverRnnGS")

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

from datetime import datetime

now = datetime.now()
date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
log_path = f"{date_time}_task_{args.task}_seed_{args.seed}"

print()
print("Log path:", log_path)

# For convenience and reproducibility, we set some EGG-level command line arguments here
opts = core.init(params=[f'--random_seed={args.seed}', # will initialize numpy, torch, and python RNGs
                                   '--lr=1e-3', # sets the learning rate for the selected optimizer
                                   '--batch_size=64',
                                   '--vocab_size=100',
                                   '--max_len=10',
                                   '--n_epochs=10',
                                   '--tensorboard',
                                   f'--tensorboard_dir=runs/{log_path}'
                                   ])

# save
if not os.path.exists("args/"):
    os.makedirs("args/")

with open(f'args/args_{log_path}.json', 'w') as fp:
    json.dump(vars(cmd_args), fp)

print("Parameters specified in the command line:")
print("Image classification task:", args.task)
print("Game type: ", args.game_type)
print("Image classification loss weight: ", args.ic_loss_weight)
print("Number of images in the game: ", args.num_imgs)
print("Same class probability: ", args.same_class_prob)
print("Evaluate on Gaussian noise images?", args.eval_noise)
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
trainset = SignalGameDataset(cifar_train_set, args.num_imgs, vision, task=args.task, same_class_prob=args.same_class_prob)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,
                                          batch_size=opts.batch_size, num_workers=0)

print("Extract image features from test set Cifar100")
testset = SignalGameDataset(cifar_test_set, args.num_imgs, vision, task=args.task, same_class_prob=args.same_class_prob)
testloader = torch.utils.data.DataLoader(testset, shuffle=False,
                                         batch_size=opts.batch_size, num_workers=0)

if args.eval_noise:
    print("Extract image features from gaussian noise dataset")
    noiseset = GaussianNoiseDataset(num_imgs=args.num_imgs, vision=vision, task=args.task, dataset_size=10000)
    noiseloader = torch.utils.data.DataLoader(noiseset, shuffle=False,
                                         batch_size=opts.batch_size, num_workers=0)

# reset seed because otherwise generating the noise dataset might effect training performance
core.util._set_seed(args.seed)

if args.game_type == "SenderReceiverRnnGS":
    sender = Sender(args.architecture.embed_size, args.num_imgs, args.architecture.hidden_sender)
    sender = core.RnnSenderGS(sender, opts.vocab_size, args.architecture.embed_size,
                              args.architecture.hidden_sender, cell=args.architecture.cell_type, max_len=opts.max_len,
                              temperature=args.training.temperature, straight_through=True)
    receiver = Receiver(args.architecture.hidden_receiver, args.architecture.embed_size, args.task, num_classes)
    receiver = core.RnnReceiverGS(receiver, opts.vocab_size, args.architecture.embed_size,
                                  args.architecture.hidden_receiver, cell=args.architecture.cell_type)
elif args.game_type == "SenderReceiverRnnReinforce":
    sender = Sender(args.architecture.embed_size, args.num_imgs, args.architecture.hidden_sender)
    receiver = Receiver(args.architecture.hidden_receiver, args.architecture.embed_size, args.task, num_classes)
    
    sender = core.RnnSenderReinforce(sender, opts.vocab_size, args.architecture.embed_size,
                              args.architecture.hidden_sender, cell=args.architecture.cell_type, max_len=opts.max_len)
    receiver = core.RnnReceiverDeterministic(receiver, opts.vocab_size, args.architecture.embed_size,
                                  args.architecture.hidden_receiver, cell=args.architecture.cell_type)
elif args.game_type == "SymbolGameReinforce":
    sender = Sender(args.architecture.embed_size, args.num_imgs, None, game_type=args.game_type)
    receiver = Receiver(None, args.architecture.embed_size, args.task, num_classes,
                        game_type=args.game_type)

    sender = core.ReinforceWrapper(sender)
    receiver = core.SymbolReceiverWrapper(receiver, opts.vocab_size, args.architecture.embed_size)
    receiver = core.ReinforceDeterministicWrapper(receiver)


model_prefix = f"model" # Example
models_path = f"checkpoints/{log_path}" # location where we store trained models

best_epoch_checkpointer = BestEpochCheckpointSaver(checkpoint_path=models_path, checkpoint_freq=0, prefix=model_prefix)

if args.task=="standard":
    loss = signal_game_loss
elif args.task=="img_clas":
    loss = ImageClasLoss(args.ic_loss_weight, args.num_imgs).get_loss
elif args.task=="target_clas":
    loss = TargetClasLoss(args.ic_loss_weight, args.num_imgs).get_loss
else:
    assert False, "Wrong task"

if args.game_type == "SenderReceiverRnnGS":
    game = core.SenderReceiverRnnGS(sender, receiver, loss)
elif args.game_type == "SenderReceiverRnnReinforce":
    game = core.SenderReceiverRnnReinforce(sender, receiver, loss)
elif args.game_type == "SymbolGameReinforce":
    game = core.SymbolGameReinforce(sender, receiver, loss, sender_entropy_coeff=0.05,
                                    receiver_entropy_coeff=0.0)

optimizer = torch.optim.Adam(game.parameters())

epoch_range = list(range(1,opts.n_epochs+1))
callbacks = [core.ConsoleLogger(as_json=True, print_train_loss=True),
             core.TensorboardLogger(),
             core.EarlyStopperAccuracy(args.training.early_stop_accuracy),
             best_epoch_checkpointer,
             # save interactions after every epoch
             core.InteractionSaver(train_epochs=epoch_range, test_epochs=epoch_range, folder_path=f"interactions/{log_path}")] 

if args.game_type == "SenderReceiverRnnGS":
    temp = core.TemperatureUpdater(agent=game.sender, decay=args.training.decay, minimum=0.1)
    callbacks.append(temp) 
            

trainer = core.Trainer(game=game, optimizer=optimizer, train_data=trainloader,
                       validation_data=testloader, callbacks=callbacks)

print("Start training")
trainer.train(n_epochs=opts.n_epochs)


if args.eval_noise:
    print("Evaluate trained model on noise images")
    # load model from the best epoch
    trainer.load_from_checkpoint(f"{models_path}/best_model.tar")
    # set the noise dataset as validation set
    trainer.validation_data = noiseloader
    # evaluate on the noise dataset
    validation_loss, validation_interaction = trainer.eval()
    epoch = opts.n_epochs+1 # placeholder integer value
    
    for callback in trainer.callbacks:
        callback.on_test_end(validation_loss, validation_interaction, epoch)
