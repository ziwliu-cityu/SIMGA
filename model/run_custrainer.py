import sys
import logging
from logging import getLogger
from recbole.utils import init_logger, init_seed
#from recbole.trainer import Trainer
#from mamba4rec import Mamba4Rec
from gated_mamba import Mamba4Rec
#from simple4rec import SMLPREC
#from gated_mamba_s import Mamba4Rec
#from bert4rec import BERT4Rec
#from gru4rec import GRU4Rec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
from custom_trainer import CustomTrainer  # 导入自定义的Trainer

if __name__ == '__main__':

    config = Config(model=Mamba4Rec, config_file_list=['config.yaml'])
    init_seed(config['seed'], config['reproducibility'])
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = Mamba4Rec(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = CustomTrainer(config, model)  # 使用自定义的Trainer

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )
    # 进行评估
    grouped_results = trainer.evaluate(test_data, show_progress=config["show_progress"])

    
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {grouped_results}")
