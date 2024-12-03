# PyTorch model training script

import torch
import click
import albumentations as A
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from dataset import RailSem19Dataset


ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
PATIENCE = 5
LR_INITIAL = 1e-3
LR_DECAY = 0.9
LR_ON_PLATEAU_DECAY = 0.1
LR_MINIMAL = 1e-5
MIN_SCORE_CHANGE = 1e-3
APPLY_LR_DECAY_EPOCH = 30


def get_device():
    if torch.backends.mps.is_available():
        print('Using MPS device')
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def move_to_device(data, device):
    if isinstance(data, (tuple, list)):
        return tuple(move_to_device(d, device) for d in data)
    return data.to(device)


def get_training_augmentation():
    transform = [    
        A.RandomCrop(height=16*23, width=16*40, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ]
    return A.Compose(transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    transform = []
    if preprocessing_fn:
        transform.append(A.Lambda(image=preprocessing_fn, always_apply=True))
    transform.append(A.Lambda(image=to_tensor, mask=to_tensor, always_apply=True))
    return A.Compose(transform)

@click.command
@click.option('-n', '--epochs', type=click.IntRange(1, 10000), default=10,
              help='Number of epoch to train')
@click.option('-i', '--image_count',type=click.IntRange(10, 10000), default=80,
              help='Maximum number of images to be processed (will be randomly selected)')
@click.option('--use_cpu', is_flag=True,
              help='Run training on CPU (will be much slower)')
@click.option('-d', '--data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), default='./data/rs19_val/',
              help='Location of the RS19 dataset')
@click.option('-v', '--val_split',type=click.FloatRange(0.1, 1.0), default=0.3,
              help='Validation-to-train split ratio')
@click.option('-s', '--save_dir',
              type=click.Path(exists=True, file_okay=False, dir_okay=True), default='./weights/',
              help='Weights directory')
def main(epochs, image_count, use_cpu, data_dir, val_split, save_dir):
    """ Training script """

    device = "cpu" if use_cpu else get_device()
    images_dir = str(Path(data_dir).joinpath('jpgs', 'rs19_val'))
    masks_dir = str(Path(data_dir).joinpath('uint8', 'rs19_val'))
    config_json_path = str(Path(data_dir).joinpath('rs19-config.json'))
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    dataset = RailSem19Dataset(
        images_dir, 
        masks_dir, 
        config_json_path,
        image_count,
        augmentation = get_training_augmentation(),
        preprocessing = get_preprocessing(preprocessing_fn),
    )
    model = smp.DeepLabV3Plus(
        encoder_name = ENCODER,
        encoder_weights = ENCODER_WEIGHTS,
        activation = ACTIVATION,
        classes=len(dataset.classes),
    )

    model.to(device)

    n_train = int(len(dataset)*(1-val_split))
    train_dataset, val_dataset = random_split(dataset, [n_train, len(dataset)-n_train])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    loss = smp_utils.losses.CrossEntropyLoss()
    # loss = smp_utils.losses.DiceLoss()
    metrics = [
        smp_utils.metrics.IoU(threshold=0.5),
        #smp_utils.metrics.Accuracy(threshold=0.5),
    ]

    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.AdamW([
        {'params' : model.decoder.parameters(), 'lr': LR_INITIAL},
    ])

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5, min_lr=LR_MINIMAL
    )

    train_epoch = smp_utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )
    val_epoch = smp_utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )

    Path(save_dir).mkdir(exist_ok=True)
    save_name = str(Path(save_dir).joinpath('rs19_deeplabv3plus.pth'))
    save_name_best = str(Path(save_dir).joinpath('rs19_deeplabv3plus_best.pth'))

    min_score = 100
    min_score_epoch = epochs
    for epoch in range(0, epochs):
        lr = optimizer.param_groups[0]['lr']
        print(
            f'\nEpoch #{epoch+1} (learning rate - {optimizer.param_groups[0]["lr"]:.2e})'
        )

        train_logs = train_epoch.run(
            (
                move_to_device(batch, device) for batch in train_loader
            )  # Move training data to device
        )
        valid_logs = val_epoch.run(
            (
                move_to_device(batch, device) for batch in val_loader
            )  # Move validation data to device
        )

        score = valid_logs['cross_entropy_loss']
        lr_scheduler.step(score)  # Update learning rate based on validation loss
        if score < min_score - MIN_SCORE_CHANGE:
            min_score = score
            min_score_epoch = epoch
            patience_counter = 0  # Reset patience counter
            torch.save(model.state_dict(), save_name_best)
        else:
            patience_counter += 1  # Increment counter if no improvement

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        torch.save(model.state_dict(), save_name)

if __name__ == '__main__':
    main()
