import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH
from src.utils.utils import get_text_from_dir

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.seed)

    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    if config.dir_path is None and config.text is None:
        raise ValueError(
            "Either dir_path or text have to be not None, but both are None"
        )
    elif config.dir_path is None:
        text = [{"text": config.text, "name": "predict"}]
    else:
        text = get_text_from_dir(config.dir_path)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    # print(model)

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        save_path=save_path,
        text=text,
        skip_model_load=False,
    )

    inferencer.run_inference()
    print("The audio were successfully generated and saved to {}".format(save_path))


if __name__ == "__main__":
    main()
