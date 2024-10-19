import omegaconf
import hydra
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pl_data_modules import BasePLDataModule
from pl_modules import AlignerPLModule

def predict(conf: omegaconf.DictConfig) -> None:
    """
    Executes the prediction process using a trained model loaded from a checkpoint.

    Args:
    conf (omegaconf.DictConfig): Configuration object loaded from Hydra, containing all the required settings for the model, data, and trainer.
    """
    # Ensure reproducibility by fixing the random seed
    pl.seed_everything(conf.train.seed)

    # Load the model from the specified checkpoint
    pl_module = AlignerPLModule.load_from_checkpoint(checkpoint_path=conf.model.checkpoint_path)

    # Update model hyperparameters based on the configuration
    pl_module.hparams["data"].update({"prediction_path" : conf.data.prediction_path})
    pl_module.hparams["data"].update({"gold_path" : conf.data.gold_path})
    pl_module.hparams["data"].update({"test_file" : conf.data.test_file})
                                      
    # Initialize the data module with configurations
    pl_data_module = BasePLDataModule(conf, pl_module)

    # Instantiate the PyTorch Lightning Trainer with the configurations
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer)

    # Prepare test data and execute the test phase
    pl_data_module.prepare_test_data()
    trainer.test(pl_module, dataloaders=pl_data_module.test_dataloader())

@hydra.main(version_base="1.2", config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    """
    Main function called when the script is executed. It loads the configuration and runs the prediction function.

    Args:
    conf (omegaconf.DictConfig): Configuration object loaded from Hydra.
    """
    predict(conf)

if __name__ == "__main__":
    main()
