import hydra
import omegaconf

import erc


@hydra.main(config_path="config", config_name="train.yaml")
def main(config: omegaconf.DictConfig):
    erc.trainer.train(config)

if __name__=="__main__":
    main()