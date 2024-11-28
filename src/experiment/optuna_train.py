import hydra
import numpy as np
import torch
from my_config.my_config import TrainConfig
from const.path import LOG_OUTPUT_DIR, MIND_SMALL_TRAIN_DATASET_DIR, MIND_SMALL_VAL_DATASET_DIR, MODEL_OUTPUT_DIR
from evaluation.RecEvaluator import RecEvaluator, RecMetrics
from mind.dataframe import read_behavior_df, read_news_df
from mind.MINDDataset import MINDTrainDataset, MINDValDataset
from recommendation.nrms import NRMS, EnhancedPLMBasedNewsEncoder, UserEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, BertConfig, EarlyStoppingCallback
from transformers.modeling_outputs import ModelOutput
from utils.logger import logging
from utils.path import generate_folder_name_with_timestamp
from utils.random_seed import set_random_seed
from utils.text import create_transform_fn_from_pretrained_tokenizer
import time
import logging
import optuna  # Import Optuna
# from optuna.integration import HuggingFaceTrainerPruningCallback

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

def objective(trial):
    set_random_seed(42)

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

def objective(trial):
    try:
        # Define hyperparameter search space
        pretrained = 'bert-base-uncased'
        npratio = trial.suggest_int('npratio', 2, 10, step=2)
        history_size = trial.suggest_int('history_size', 20, 50, step=10)
        batch_size = 8
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        max_len = trial.suggest_int('max_len', 30, 126, step=32)
        num_epochs = 1
        num_attention_heads = trial.suggest_categorical('num_attention_heads', [8, 12, 16])
        num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 3)
        intermediate_size = trial.suggest_int('intermediate_size', 512, 2048, step=512)
        
        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize model and data as before
        hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size
        loss_fn = nn.CrossEntropyLoss()
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        transform_fn = create_transform_fn_from_pretrained_tokenizer(tokenizer, max_len)
        model_save_dir = generate_folder_name_with_timestamp(MODEL_OUTPUT_DIR)
        model_save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        logging.info("Initializing Model")
        model_start_time = time.time()
        news_encoder = EnhancedPLMBasedNewsEncoder(pretrained)

        # Obtain hidden_size from the news encoder's pretrained model
        hidden_size = news_encoder.plm.config.hidden_size

        # Create the Fastformer configuration
        fastformer_config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=history_size + 1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pooler_type='weightpooler'
        )

        user_encoder = UserEncoder(fastformer_config)
        nrms_net = NRMS(
            news_encoder=news_encoder,
            user_encoder=user_encoder,
            hidden_size=hidden_size,
            loss_fn=loss_fn
        ).to(device)
        model_end_time = time.time()
        logging.info(f"Model initialized in {model_end_time - model_start_time:.2f} seconds")

        # Load datasets
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

        # Training configuration
        training_args = TrainingArguments(
            output_dir=model_save_dir,
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            lr_scheduler_type="linear",
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            remove_unused_columns=False,
            report_to="none",
        )

        # pruning_callback = HuggingFaceTrainerPruningCallback(trial, "eval_loss")  # Monitor evaluation loss


        # Create Trainer
        trainer = Trainer(
            model=nrms_net,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[AUCCallback(eval_dataset, device)]  # 添加AUC回调
        )

        # Start training
        logging.info("Starting Training")
        train_start_time = time.time()
        trainer.train()
        train_end_time = time.time()
        logging.info(f"Training completed in {train_end_time - train_start_time:.2f} seconds")

        # Evaluation
        logging.info("Evaluating")
        eval_start_time = time.time()
        metrics = evaluate(trainer.model, eval_dataset, device)
        eval_end_time = time.time()
        logging.info(f"Evaluation completed in {eval_end_time - eval_start_time:.2f} seconds")
        auc_score = metrics.auc
        logging.info(f"AUC: {auc_score:.4f}")

        # Return negative AUC because Optuna minimizes the objective function
        return -auc_score
    finally:
        # Clean up memory
        del nrms_net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)  # Adjust n_trials as needed

    print("Best trial:")
    trial = study.best_trial

    print(f"  AUC: {-trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
