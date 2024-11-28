import hydra
import numpy as np
import torch
# from small_config.small_config import TrainConfig
from recommenders.models.newsrec.models.nrms import NRMSModel
from const.path import LOG_OUTPUT_DIR, MIND_SMALL_TRAIN_DATASET_DIR, MIND_SMALL_VAL_DATASET_DIR, MODEL_OUTPUT_DIR
from evaluation.RecEvaluator import RecEvaluator, RecMetrics
from mind.dataframe import read_behavior_df, read_news_df
from mind.MINDDataset import MINDTrainDataset, MINDValDataset
from recommendation.nrms import NRMS, EnhancedPLMBasedNewsEncoder, UserEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from transformers.modeling_outputs import ModelOutput
from utils.logger import logging
from utils.path import generate_folder_name_with_timestamp
from utils.random_seed import set_random_seed
from utils.text import create_transform_fn_from_pretrained_tokenizer
import time
import logging

class AUCCallback(TrainerCallback):
    def __init__(self, eval_dataset, device):
        self.eval_dataset = eval_dataset
        self.device = device
        self.best_auc = 0.0

    def on_epoch_end(self, args, state, control, **kwargs):
        logging.info("Evaluating at the end of epoch")
        metrics = evaluate(kwargs['model'], self.eval_dataset, self.device)
        current_auc = metrics.auc
        logging.info(f"Current AUC: {current_auc:.4f}")
        if current_auc > self.best_auc:
            self.best_auc = current_auc
            logging.info(f"New best AUC: {self.best_auc:.4f}")

def evaluate(net: torch.nn.Module, eval_mind_dataset: MINDValDataset, device: torch.device) -> RecMetrics:
    net.eval()
    EVAL_BATCH_SIZE = 1  # 增大批量大小以提高评估速度
    eval_dataloader = DataLoader(
        eval_mind_dataset, 
        batch_size=EVAL_BATCH_SIZE, 
        pin_memory=True,
        num_workers=4  # 添加多线程加载
    )

    all_y_true = []
    all_y_score = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch["news_histories"] = batch["news_histories"].to(device)
        batch["candidate_news"] = batch["candidate_news"].to(device)
        batch["target"] = batch["target"].to(device)
        
        with torch.no_grad():
            model_output: ModelOutput = net(**batch)
            
        y_score = model_output.logits.flatten().cpu().to(torch.float64).numpy()
        y_true = batch["target"].flatten().cpu().to(torch.int).numpy()
        
        all_y_true.append(y_true)
        all_y_score.append(y_score)
    
    # 合并所有批次的结果
    y_true_concat = np.concatenate(all_y_true)
    y_score_concat = np.concatenate(all_y_score)
    
    # 只计算 AUC
    auc = RecEvaluator.evaluate_all(y_true_concat, y_score_concat)
    logging.info(f"Evaluation AUC: {auc.auc:.4f}")
    return auc

def train(
    pretrained: str,
    npratio: int,
    history_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_len: int,
    device: torch.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu"),
    # device: torch.device = torch.device("cpu"),
) -> None:
    try:
        # 记录开始时间
        total_start_time = time.time()

        # 初始化模型和数据集
        hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size
        loss_fn = nn.CrossEntropyLoss()
        transform_fn = create_transform_fn_from_pretrained_tokenizer(
            AutoTokenizer.from_pretrained(pretrained), 
            max_len
        )
        model_save_dir = generate_folder_name_with_timestamp(MODEL_OUTPUT_DIR)
        model_save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化模型
        logging.info("Initializing Model")
        model_start_time = time.time()
        news_encoder = EnhancedPLMBasedNewsEncoder(pretrained)
        user_encoder = UserEncoder(hidden_size=hidden_size)
        nrms_net = NRMS(
            news_encoder=news_encoder, 
            user_encoder=user_encoder, 
            hidden_size=hidden_size, 
            loss_fn=loss_fn
        ).to(device, dtype=torch.bfloat16)
        model_end_time = time.time()
        logging.info(f"Model initialized in {model_end_time - model_start_time:.2f} seconds")

        # 加载数据集
        logging.info("Loading Datasets")
        data_start_time = time.time()
        train_news_df = read_news_df(MIND_SMALL_TRAIN_DATASET_DIR / "news.tsv")
        train_behavior_df = read_behavior_df(MIND_SMALL_TRAIN_DATASET_DIR / "behaviors.tsv")
        train_dataset = MINDTrainDataset(
            train_behavior_df, 
            train_news_df, 
            transform_fn, 
            npratio, 
            history_size, 
            device
        )

        val_news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
        val_behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
        eval_dataset = MINDValDataset(
            val_behavior_df, 
            val_news_df, 
            transform_fn, 
            history_size
        )
        data_end_time = time.time()
        logging.info(f"Datasets loaded in {data_end_time - data_start_time:.2f} seconds")

        # 训练配置
        training_args = TrainingArguments(
            output_dir=model_save_dir,
            logging_strategy="epoch",  # 每个epoch记录一次
            save_strategy="epoch",
            save_total_limit=3,  # 只保存最近3个检查点
            lr_scheduler_type="linear",  # 使用线性学习率调度
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs,
            remove_unused_columns=False,
            report_to="none",
        )

        # 创建Trainer
        trainer = Trainer(
            model=nrms_net,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[AUCCallback(eval_dataset, device)]  # 添加AUC回调
        )

        # 开始训练
        logging.info("Starting Training")
        train_start_time = time.time()
        trainer.train()
        train_end_time = time.time()
        logging.info(f"Training completed in {train_end_time - train_start_time:.2f} seconds")

        # 最终评估
        logging.info("Final Evaluation")
        eval_start_time = time.time()
        final_metrics = evaluate(trainer.model, eval_dataset, device)
        eval_end_time = time.time()
        logging.info(f"Final evaluation completed in {eval_end_time - eval_start_time:.2f} seconds")
        logging.info(f"Final AUC: {final_metrics.auc:.4f}")

        # 记录总时间
        total_end_time = time.time()
        logging.info(f"Total training time: {total_end_time - total_start_time:.2f} seconds")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

@hydra.main(version_base=None, config_name="train_config")
def main(cfg: TrainConfig) -> None:
    try:
        set_random_seed(cfg.random_seed)
        train(
            cfg.pretrained,
            cfg.npratio,
            cfg.history_size,
            cfg.batch_size,
            cfg.gradient_accumulation_steps,
            cfg.epochs,
            cfg.learning_rate,
            cfg.weight_decay,
            cfg.max_len,
        )
    except Exception as e:
        logging.error(e)

if __name__ == "__main__":
    main()