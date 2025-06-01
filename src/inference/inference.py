import os
import sys
import glob
import pickle


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.utils import model_dir
from model.disease_predictor import DiseasePredictor
from dataset.watch_log import WatchLogDataset, get_datasets
from dataset.dataloader import DataLoader
from src.evaluate.evaluate import evaluate
from postprocess.postprocess import write_db

def recommend_to_df(recommend):
    return pd.DataFrame(
        data=recommend,
        columns="recommend_content_id".split()
    )


def make_inference_df(data):
    columns = "user_id content_id watch_seconds rating popularity".split()
    return pd.DataFrame(
        data=[data],
        columns=columns,
    )


def init_model(checkpoint):
    model = DiseasePredictor(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = checkpoint.get("scaler", None)
    label_encoder = checkpoint.get("label_encoder", None)
    return model, scaler, label_encoder


def load_checkpoint():
    target_dir = model_dir(DiseasePredictor.name)
    models_path = os.path.join(target_dir, "*.pkl")
    latest_model = glob.glob(models_path)[-1]

    with open(latest_model, "rb") as f:
        checkpoint = pickle.load(f)
        
    return checkpoint


def inference(model, scaler, label_encoder, data: np.array, batch_size=1):
    if data.size > 0:  # 실시간
        df = make_inference_df(data)
        dataset = WatchLogDataset(df, scaler=scaler, label_encoder=label_encoder)
    else:  # 배치
        _, _, dataset = get_datasets(scaler=scaler, label_encoder=label_encoder)

    dataloader = SimpleDataLoader(
	dataset.features, dataset.labels, batch_size=batch_size, shuffle=False
    )
    loss, predictions = evaluate(model, dataloader)
    print(loss, predictions)
    return [dataset.decode_content_id(idx) for idx in predictions]


if __name__ == '__main__':  # python inference/inference.py
    load_dotenv()
    checkpoint = load_checkpoint()
    model, scaler, label_encoder = init_model(checkpoint)
    # data = np.array([1, 1092073, 4508, 7.577, 1204.764])
    recommend = inference(model, scaler, label_encoder, data=np.array([]), batch_size=64)
    print(recommend)

    recommend_df = recommend_to_df(recommend)
    write_db(recommend_df, "mlops", "recommend")