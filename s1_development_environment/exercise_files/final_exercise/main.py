import click
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model import ImageClassifier
from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="Learning rate to use for training.")
@click.option("--model_name", default='efficientnet_es.ra_in1k', help="Name of the model.")
@click.option("--num_classes", default=10, help="Number of classes.")
@click.option("--drop_rate", default=0.5, help="Dropout rate.")
@click.option("--pretrained", is_flag=True, help="Use pretrained model.")
@click.option("--lr_backbone", default=3e-6, help="Learning rate for the backbone.")
@click.option("--lr_head", default=3e-4, help="Learning rate for the head.")
@click.option("--optimizer", default="AdamW", help="Optimizer to use.")
@click.option("--criterion", default="cross_entropy", help="Criterion for training.")
@click.option("--batch_size", default=64, help="Batch size for training.")
@click.option("--num_workers", default=4, help="Number of workers for data loading.")
@click.option("--num_epochs", default=100, help="Number of epochs for training.")
@click.option("--log_every_n_steps", default=10, help="Logging frequency.")
@click.option("--wandb_name", default="training_run", help="Wandb run name.")
@click.option("--file_name", default="model", help="Model file name for checkpoints.")
def train(lr, model_name ,num_classes, drop_rate, pretrained, lr_backbone, lr_head, optimizer, 
          criterion, batch_size, num_workers, num_epochs, log_every_n_steps, 
          wandb_name, file_name):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    checkpoint_callback = ModelCheckpoint(
        dirpath="model_checkpoints/corrupted_mnist/",
        filename=file_name + '-{epoch:02d}-{val_accuracy:.2f}',
        save_top_k=1,
        verbose=True,
        monitor="val_accuracy",
        mode="max"
    )

    wandb_logger = WandbLogger(name=wandb_name, project="MLOps-mnist")

    # TODO: Implement training loop here
    model = ImageClassifier(
        model_name=model_name,
        num_classes=num_classes,
        drop_rate=drop_rate,
        pretrained=pretrained,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        optimizer=optimizer,
        criterion=criterion,
    )

    train_loader, val_loader, _ = mnist(batch_size = batch_size, num_workers=num_workers)

    trainer = Trainer(
        max_epochs=num_epochs, 
        logger=wandb_logger, 
        log_every_n_steps=log_every_n_steps, 
        callbacks=[checkpoint_callback]
        )

    trainer.fit(model, train_loader, val_loader)

    
@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, _, test_loader = mnist()


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
