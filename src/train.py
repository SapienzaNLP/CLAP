import os
import omegaconf
import hydra
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pl_modules import BasePLModule
from pl_data_modules import BasePLDataModule

# Disable tokenizers parallelism to prevent warnings in multi-threaded environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(conf: omegaconf.DictConfig) -> None:
    """
    Train a model using PyTorch Lightning, with configurations managed by Hydra.

    Args:
    conf (omegaconf.DictConfig): Configuration containing all setup and hyperparameters.
    """
    # Ensuring reproducibility
    pl.seed_everything(conf.train.seed)

    # Initialize main module and data module
    pl_module = BasePLModule(conf)
    pl_data_module = BasePLDataModule(conf, pl_module)

    # Save initial pretrained model and tokenizer if specified
    if conf.model.save_model_path:
        pl_module.model.save_pretrained(conf.model.save_model_path)
        pl_module.tokenizer.save_pretrained(conf.model.save_model_path)

    # Set up Weights & Biases logging if enabled
    wandb_logger = WandbLogger(project=conf.train.wandb.project, name=conf.train.wandb.run, entity=conf.train.wandb.entity) if conf.train.report_wandb else None

    # Prepare callbacks for training
    callbacks = [
        hydra.utils.instantiate(conf.train.model_checkpoint_callback)
        if conf.train.model_checkpoint_callback else None,
        LearningRateMonitor(logging_interval='step') if conf.train.report_wandb else None
    ]
    # Filter out any `None` callbacks
    callbacks = [cb for cb in callbacks if cb]

    # Initialize the trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        num_sanity_val_steps=0
    )

    # Start training or resume from checkpoint
    if conf.model.resume_from_checkpoint:
        pl_module = pl_module.load_from_checkpoint(conf.model.resume_from_checkpoint)
        pl_module.save_hyperparameters(conf)
        pl_module.tokenizer = pl_data_module.tokenizer

        pl_module = pl_module.load_from_checkpoint(conf.model.resume_from_checkpoint)
        trainer.fit(pl_module, datamodule=pl_data_module, ckpt_path=conf.model.resume_from_checkpoint)
    else:
        trainer.fit(pl_module, datamodule=pl_data_module)

    # Load the best model and save it post-training
    pl_module = pl_module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    pl_module.model.save_pretrained(conf.model.save_model_path)
    pl_module.tokenizer.save_pretrained(conf.model.save_model_path)

@hydra.main(version_base="1.2", config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    """
    Main function to initiate training.
    """
    train(conf)

if __name__ == "__main__":
    main()
