import hydra
import omegaconf

import erc


logger = erc.utils.get_logger()


@hydra.main(config_path="config", config_name="train.yaml")
def main(config: omegaconf.DictConfig):
    logger.info("Start Training")
    erc.trainer.train(config)

if __name__=="__main__":
    main()