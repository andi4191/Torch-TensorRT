import os
import torch
import numpy as np

from model import VGG16Unet
from torchvision.models import segmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim

from PIL import Image
import argparse
from tqdm import tqdm

from utils import get_loaders, load_checkpoint, save_checkpoint, check_accuracy

IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_WORKERS = 2
PIN_MEMORY = True
TRAIN_IMG_DIR = "ADEChallengeData2016/images/training"
TRAIN_MASK_DIR = "ADEChallengeData2016/annotations/training"

VAL_IMG_DIR = "ADEChallengeData2016/images/validation"
VAL_MASK_DIR = "ADEChallengeData2016/annotations/validation"
CHECKPOINT_PREFIX = "seg_model"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(loader, model, optimizer, loss_fn, scaler):
    pr_loop = tqdm(loader)

    for data, target in pr_loop:
        data = data.to(device=DEVICE)
        target = target.to(device=DEVICE)

        preds = model(data)

        target = target.sigmoid() * 149

        loss = loss_fn(preds, target.long())
        loss = loss.mean()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pr_loop.set_postfix(loss=loss.item())


def main():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--data', type=str, default='', help='Path to dataset root dir')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for training optimizer')
    parser.add_argument("--weight-decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument('--start-from', type=int, default=0, help='Load the checkpoint epoch')
    parser.add_argument('--export', type=str, default='segmentation_model.jit.pt', help='Export as a Torch Script')
    parser.add_argument('--load-model', type=bool, default=True, help='Load a checkpoint')

    args = parser.parse_args()
    train_transform = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = VGG16Unet().to(device=DEVICE)

    # Multi-class semantic segmentation
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Pick dataset root directory according as per user input if required
    train_loader, val_loader  = get_loaders(
        TRAIN_IMG_DIR if args.data == '' else os.path.join(args.data, TRAIN_IMG_DIR),
        TRAIN_MASK_DIR if args.data == '' else os.path.join(args.data, TRAIN_MASK_DIR),
        VAL_IMG_DIR if args.data == '' else os.path.join(args.data, VAL_IMG_DIR),
        VAL_MASK_DIR if args.data == '' else os.path.join(args.data, VAL_MASK_DIR),
        args.batch,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    epochs = 0
    ckpt_file = None
    if args.load_model == True:
        for ckpt in os.listdir():
            if ckpt.startswith(CHECKPOINT_PREFIX):
                names = ckpt.split('_')
                if names:
                    names = names[-1].split('.pth.tar')[0]
                
                epochs = max(epochs, int(names))
                ckpt_file = ckpt
        
        if ckpt_file is not None and os.path.exists(ckpt_file):
            load_checkpoint(torch.load(ckpt_file), model)
        
    check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()    
    for epoch in range(epochs, args.epochs):
        print(f"Epoch: {epoch}/{args.epochs}...")
        train(train_loader, model, optimizer, loss_fn, scaler)

        if epoch % 5 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, CHECKPOINT_PREFIX + "_epoch_" + str(epoch) + ".pth.tar")
    save_checkpoint(checkpoint, CHECKPOINT_PREFIX + "_epoch_" + str(args.epochs) + ".pth.tar")
    
    # Check accuracy
    check_accuracy(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()