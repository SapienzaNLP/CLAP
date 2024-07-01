import os
import omegaconf
import hydra
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pl_data_modules import BasePLDataModule
from pl_modules import BasePLModule

def predict(conf: omegaconf.DictConfig) -> None:
    """
    Runs the prediction pipeline for a given PyTorch Lightning model using configurations provided by Hydra.

    Parameters:
    conf (omegaconf.DictConfig): Configuration object containing paths, model settings, and trainer options.
    """

    # Ensure reproducibility by setting a seed
    pl.seed_everything(conf.train.seed)

    # Load model from checkpoint
    pl_module = BasePLModule.load_from_checkpoint(checkpoint_path=conf.model.checkpoint_path)

    # Update hyperparameters from config
    pl_module.hparams["data"].update({"prediction_path": conf.data.prediction_path})
    pl_module.hparams["data"].update({"gold_path": conf.data.gold_path})
    pl_module.hparams["data"].update({"test_file": conf.data.test_file})
    pl_module.hparams["train"].update({"decoder_start_token_id": conf.train.decoder_start_token_id})

    # Clear previous predictions
    if os.path.exists(conf.data.prediction_path):
        os.system(f"rm -rf {conf.data.prediction_path}")

    # Initialize data module
    pl_data_module = BasePLDataModule(conf, pl_module)

    # Instantiate trainer
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer)

    # Prepare and sort test data
    pl_data_module.prepare_test_data()
    pl_data_module.sort_test_data()
    
    # Set tokenizer for the model
    pl_module.tokenizer = pl_data_module.tokenizer

    # Execute testing phase
    trainer.test(pl_module, dataloaders=pl_data_module.test_dataloader())

@hydra.main(version_base="1.2", config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    """
    Main function that gets called with Hydra to inject configuration.
    """
    predict(conf)

if __name__ == "__main__":
    main()
