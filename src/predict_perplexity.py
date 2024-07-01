import omegaconf
import hydra
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pl_data_modules import BasePLDataModule
from pl_modules import ProbabilityPLModule

def predict(conf: omegaconf.DictConfig) -> None:
    """
    Perform a prediction using a trained model loaded from a checkpoint based on the provided configuration.

    Args:
    conf (omegaconf.DictConfig): Configuration containing all necessary settings including paths and training parameters.
    """
    # Set the random seed for reproducibility across runs
    pl.seed_everything(conf.train.seed)

    # Load the trained module from the checkpoint path specified in the configuration
    pl_module = ProbabilityPLModule.load_from_checkpoint(checkpoint_path=conf.model.checkpoint_path)

    # Update model hyperparameters based on the configuration
    pl_module.hparams["data"].update({"prediction_path" : conf.data.prediction_path})
    pl_module.hparams["data"].update({"gold_path" : conf.data.gold_path})
    pl_module.hparams["data"].update({"test_file" : conf.data.test_file})
                                      

    # Output the prediction path to confirm the correct path is being used
    print(f"Prediction path: {pl_module.hparams.data.prediction_path}")

    # Initialize the data module with the current configuration and model
    pl_data_module = BasePLDataModule(conf, pl_module)

    # Create the trainer from configuration using Hydra
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer)

    # Prepare the dataset for testing
    pl_data_module.prepare_test_data()

    # Run the testing phase using the loaded model and prepared data
    trainer.test(pl_module, dataloaders=pl_data_module.test_dataloader())

@hydra.main(version_base="1.2", config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    """
    Main function invoked by Hydra, loading the configuration and executing the prediction function.
    """
    predict(conf)

if __name__ == "__main__":
    main()
