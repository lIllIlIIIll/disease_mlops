
import numpy as np
import pickle
import os
import datetime

from utils.utils import model_dir
from sklearn.ensemble import RandomForestClassifier

class DiseasePredictor:
    name = "disease_predictor"  # utils.py(model_dir) : /opt/mlops/models/disease_predictor/model.pth

    def __init__(self, input_dim, hidden_dim, num_classes):
        # RandomForest 파라미터 (기존 파라미터는 무시하고 RF 설정)
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=hidden_dim,  # hidden_dim을 max_depth로 활용
            random_state=42,
            n_jobs=-1
        )
        self.num_classes = num_classes
        self.is_fitted = False

    def forward(self, x):
        if not self.is_fitted:
            # 첫 번째 호출시에는 더미 결과 반환
            return np.random.rand(x.shape[0], self.num_classes)
        
        # 예측 확률 반환
        return self.model.predict_proba(x)

    def backward(self, x, y, output, lr=0.001):
        # y가 one-hot이면 argmax로 변환
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y.flatten()
        
        # RandomForest 학습
        self.model.fit(x, y_labels)
        self.is_fitted = True

    def load_state_dict(self, state_dict):
        self.model = state_dict["model"]
        self.is_fitted = state_dict.get("is_fitted", True)


def model_save(model, model_params, epoch, loss, scaler, label_encoder):
    save_dir = model_dir(model.name)  # disease_predictor
    os.makedirs(save_dir, exist_ok=True)
    
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")  # 250521142130
    dst = os.path.join(save_dir, f"E{epoch}_T{current_time}.pkl")

    save_data = {
        "epoch": epoch,
        "model_params": model_params,
        "model_state_dict": {
            "model": model.model,
            "is_fitted": model.is_fitted,
        },
        "loss": loss,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }

    with open(dst, "wb") as f:
        pickle.dump(save_data, f)

    return dst
