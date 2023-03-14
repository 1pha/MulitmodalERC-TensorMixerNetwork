import hydra
import omegaconf

import erc


logger = erc.utils.get_logger(name=__name__)


@hydra.main(config_path="config", config_name="train.yaml", version_base="1.1")
def main(config: omegaconf.DictConfig):
    logger.info("Start Training")
    erc.trainer.train(config)

if __name__=="__main__":
    main()