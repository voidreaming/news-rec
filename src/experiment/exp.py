import torch
import os
import numpy as np
import pandas as pd
from my_config.my_config import TrainConfig
from recommendation.nrms import NRMS, EnhancedPLMBasedNewsEncoder, UserEncoder
from transformers import AutoTokenizer, BertConfig
from mind.dataframe import read_behavior_df, read_news_df
from utils.text import create_transform_fn_from_pretrained_tokenizer
from utils.random_seed import set_random_seed
from utils.logger import logging
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file
import zipfile
import json

def load_model(config, model_path, device):
    logging.info("Initializing model...")
    news_encoder = EnhancedPLMBasedNewsEncoder(config.pretrained)
    hidden_size = news_encoder.plm.config.hidden_size
    fastformer_config = BertConfig(
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=hidden_size,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=config.history_size + 1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pooler_type="weightpooler",
    )
    user_encoder = UserEncoder(fastformer_config)
    nrms_net = NRMS(
        news_encoder=news_encoder,
        user_encoder=user_encoder,
        hidden_size=hidden_size,
        loss_fn=None,
    ).to(device)
    nrms_net.hidden_size = hidden_size
    logging.info(f"Loading model weights from {model_path}")
    state_dict = load_file(model_path)
    nrms_net.load_state_dict(state_dict, strict=False)
    nrms_net.eval()
    logging.info("Model loaded successfully.")
    return nrms_net

def prepare_data(config, news_path, behavior_path, transform_fn):
    logging.info("Loading validation data...")
    news_df = read_news_df(news_path)
    behaviors_df = read_behavior_df(behavior_path)
    logging.info("Validation data loaded successfully.")
    return news_df, behaviors_df

def generate_news_embeddings(model, news_df, transform_fn, device):
    logging.info("Generating news embeddings...")
    news_embeddings = {}
    column_names = news_df.columns  # Get column names

    for idx, row in enumerate(tqdm(news_df.iter_rows(named=False), total=len(news_df), desc="Generating News Embeddings")):
        row_dict = dict(zip(column_names, row))  # Convert row to a dictionary
        news_id = row_dict['news_id']
        title = row_dict['title']
        title_tensor = transform_fn([title]).to(device)
        with torch.no_grad():
            news_vector = model.news_encoder(title_tensor).cpu().numpy()
        news_embeddings[news_id] = news_vector.squeeze(0)
    logging.info("News embeddings generated.")
    return news_embeddings


def generate_predictions(model, news_embeddings, behaviors_df, transform_fn, device, output_file, config, data_path):
    logging.info("Generating recommendations...")
    
    # Lists to store impression indexes, prediction scores, and rankings
    all_impr_indexes = []
    all_preds = []

    # Convert Polars DataFrame to Pandas DataFrame
    behaviors_df_pandas = behaviors_df.to_pandas()
    
    # Iterate through the behavior dataset (validation dataset)
    for idx in tqdm(range(len(behaviors_df_pandas)), desc="Generating Predictions"):
        behavior_data = behaviors_df_pandas.iloc[idx]

        # Ensure 'news_histories' exists and is not missing
        if "news_histories" not in behavior_data or pd.isna(behavior_data["news_histories"]):
            logging.warning(f"Missing news_histories for index {idx}, skipping this entry.")
            continue
        
        user_history = torch.tensor(behavior_data["news_histories"]).unsqueeze(0).to(device)  # Prepare user history tensor
        candidate_news = torch.tensor(behavior_data["candidate_news"]).unsqueeze(0).to(device)  # Prepare candidate news tensor
        candidate_news_id = behavior_data["candidate_news_id"]  # List of candidate news IDs
        impression_id = behavior_data["impression_id"]  # Impression ID

        # Generate predictions using the model
        with torch.no_grad():
            # Obtain prediction scores
            preds = model(user_history, candidate_news)
        
        # Rank the predictions
        ranked_indices = np.argsort(np.argsort(preds.cpu().numpy()[0])[::-1]) + 1  # Ranking starts from 1
        all_impr_indexes.append(impression_id)
        all_preds.append(ranked_indices.tolist())

    # Write predictions to a text file
    logging.info("Writing predictions to file...")
    with open(os.path.join(data_path, 'prediction.txt'), 'w') as f:
        for impr_index, preds in zip(all_impr_indexes, all_preds):
            pred_rank = '[' + ','.join([str(i) for i in preds]) + ']'
            f.write(f'{impr_index} {pred_rank}\n')

    # Zip the prediction file
    with zipfile.ZipFile(os.path.join(data_path, 'prediction.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(data_path, 'prediction.txt'), arcname='prediction.txt')

    logging.info("Prediction generation completed and zipped successfully.")




def zip_predictions(output_file, zip_file):
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, arcname='prediction.txt')
    logging.info(f"Predictions zipped to {zip_file}")

def main():
    config = TrainConfig(
        pretrained="bert-base-uncased",
        npratio=10,
        history_size=50,
        batch_size=8,
        gradient_accumulation_steps=4,
        epochs=10,
        learning_rate=3.0e-05,
        weight_decay=0.01,
        max_len=128,
        random_seed=42,
    )
    set_random_seed(config.random_seed)
    model_path = "output/model/2024-11-25/20-12-01/checkpoint-44147/model.safetensors"
    news_path = Path("dataset/mind/large/val/news.tsv")
    behavior_path = Path("dataset/mind/large/val/behaviors.tsv")
    output_file = "prediction.txt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained)
    transform_fn = create_transform_fn_from_pretrained_tokenizer(tokenizer, config.max_len)
    model = load_model(config, model_path, device)
    news_df, behaviors_df = prepare_data(config, news_path, behavior_path, transform_fn)
    news_embeddings = generate_news_embeddings(model, news_df, transform_fn, device)
    generate_predictions(model, news_embeddings, behaviors_df, transform_fn, device, output_file, config)
    zip_predictions(output_file, 'prediction.zip')

if __name__ == "__main__":
    main()
