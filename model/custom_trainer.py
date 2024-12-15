import os
from logging import getLogger
from time import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import torch.cuda.amp as amp

from recbole.data.interaction import Interaction
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator, Collector
from recbole.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    EvaluatorType,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger,
)
from torch.nn.parallel import DistributedDataParallel
from recbole.trainer import Trainer
from recbole.data.dataset import Dataset

class CustomTrainer(Trainer):
    def __init__(self, config, model):
        super(CustomTrainer, self).__init__(config, model)
        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.wandblogger = WandbLogger(config)
        self.device = config["device"]

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            self.logger.info(f"Loading model structure and parameters from {checkpoint_file}")

        self.model.eval()
        results = {}

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = tqdm(eval_data, total=len(eval_data), ncols=100, desc=set_color(f"Evaluate   ", "pink")) if show_progress else eval_data

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        self.logger.info(f"Overall results: {result}")  # 输出评估结果

        if 'ndcg@10' not in result:
            raise KeyError("Evaluation result does not contain 'ndcg@10'")

        results['overall'] = result
      

        # 分组评估
        item_lengths = eval_data.dataset.inter_feat['item_length'].cpu().numpy()
        #groups = [(0, 5), (5, 10),(10,15),(15,20),(20, float('inf'))]
        #groups = [(0, 20), (20, 50),(50, float('inf'))]
        groups = [(0, 5), (5, 20),(20, float('inf'))]
        for start, end in groups:
            group_mask = (item_lengths >= start) & (item_lengths < end)
            group_indices = np.where(group_mask)[0]
            group_result, group_sample_count = self.evaluate_group(eval_data, group_indices)
            results[f'group_{start}_{end}'] = group_result
            self.logger.info(f"Group Len:{start} - {end} | Samples: {group_sample_count}: {group_result}")
        self.wandblogger.log_eval_metrics(result, head="eval")
        return results

    def evaluate_group(self, eval_data, group_indices):
        # Create a subset of the dataset
        subset_inter_feat = eval_data.dataset.inter_feat[group_indices]
        subset_dataset = Dataset(config=eval_data.dataset.config)
        subset_dataset.inter_feat = subset_inter_feat
        # Create a new FullSortEvalDataLoader for the subset
        subset_eval_data = FullSortEvalDataLoader(
            config=eval_data.config, dataset=subset_dataset, sampler=eval_data.sampler, shuffle=False
        )

        # Reinitialize the collector
        self.eval_collector = Collector(config=self.config)
        
        iter_data = tqdm(subset_eval_data, total=len(subset_eval_data), ncols=100, desc=set_color(f"Evaluate Subset", "pink"))
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = self._full_sort_batch_eval(batched_data)
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        return result , len(group_indices)
    
    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(
            valid_data, load_best_model=False, show_progress=show_progress
        )
        valid_result = valid_result['overall']
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result