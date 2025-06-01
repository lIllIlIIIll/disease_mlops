import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import fire
from icecream import ic
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import numpy as np

from src.dataset.watch_log import get_datasets
from src.dataset.dataloader import DataLoader
from src.model.disease_predictor import DiseasePredictor, model_save
from src.utils.utils import init_seed, auto_increment_run_suffix
from src.train.train import train
from src.evaluate.evaluate import evaluate
from src.utils.constant import Models
from src.inference.inference import (
    load_checkpoint, init_model, inference, recommend_to_df
)
from src.postprocess.postprocess import write_db


init_seed()
load_dotenv()


def setup_mlflow(experiment_name):
    # MLflow 추적 서버 URI 설정 (환경변수에서 가져오거나 기본값 사용)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8550")
    mlflow.set_tracking_uri(tracking_uri)
    
    # 실험 설정 (없으면 생성)
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id


def get_runs(experiment_name):
    """실험의 모든 실행 가져오기"""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    return runs


def get_latest_run(experiment_name):
    """최신 실행 이름 가져오기"""
    runs = get_runs(experiment_name)
    if not runs:
        return f"{experiment_name}-000"
    
    return runs[0].info.run_name or f"{experiment_name}-000"


def run_train(model_name, batch_size=64, num_epochs=10):
    Models.validation(model_name)

    # MLflow 실험 설정
    experiment_name = model_name.replace("_", "-")  # disease_predictor -> disease-predictor
    setup_mlflow(experiment_name)
    
    # 실행 이름 생성
    run_name = get_latest_run(experiment_name)
    next_run_name = auto_increment_run_suffix(run_name)

    # MLflow 실행 시작
    with mlflow.start_run(run_name=next_run_name) as run:
        # 하이퍼파라미터 로깅
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        
        # 태그 설정
        mlflow.set_tag("model_type", "content-based")
        mlflow.set_tag("task", "disease-predict")
        mlflow.set_tag("description", "content-based disease predict model")

        # 데이터 로드
        train_dataset, val_dataset, test_dataset = get_datasets()
        train_loader = DataLoader(train_dataset.features, train_dataset.labels, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset.features, val_dataset.labels, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset.features, test_dataset.labels, batch_size=batch_size, shuffle=False)

        # 데이터셋 정보 로깅
        mlflow.log_param("train_size", len(train_dataset.features))
        mlflow.log_param("val_size", len(val_dataset.features))
        mlflow.log_param("test_size", len(test_dataset.features))
        mlflow.log_param("features_dim", train_dataset.features_dim)
        mlflow.log_param("num_classes", train_dataset.num_classes)

        model_params = {
            "input_dim": train_dataset.features_dim,
            "num_classes": train_dataset.num_classes,
            "hidden_dim": 64,
        }
        
        # 모델 파라미터 로깅
        for key, value in model_params.items():
            mlflow.log_param(f"model_{key}", value)
        
        # 모델 초기화
        model_class = Models[model_name.upper()].value  # Models -> DISEASE_PREDICTOR = DiseasePredictor
        model = model_class(**model_params)

        # 훈련 루프
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader)
            val_loss, _ = evaluate(model, val_loader)
            
            ic(f"{epoch + 1}/{num_epochs}")
            ic(train_loss)
            ic(val_loss)
            
            # 메트릭 로깅
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            # 최고 성능 모델 추적
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                mlflow.log_metric("best_val_loss", best_val_loss)
        
        # 테스트 평가
        test_loss, predictions = evaluate(model, test_loader)
        ic(test_loss)
        ic([train_dataset.decode_prognosis(idx) for idx in predictions])
        
        # 최종 메트릭 로깅
        mlflow.log_metric("final_test_loss", test_loss)
        mlflow.log_metric("final_train_loss", train_loss)
        
        # 모델 저장
        model_save(
            model=model,
            model_params=model_params,
            epoch=num_epochs,
            loss=train_loss,
            scaler=train_dataset.scaler,
            label_encoder=train_dataset.label_encoder,
        )
        
        # MLflow에 모델 등록
        try:
            # PyTorch 모델로 로깅 (또는 sklearn이면 mlflow.sklearn.log_model 사용)
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=f"{model_name}_model"
            )
            
            # 추가 아티팩트 저장 (scaler, label_encoder 등)
            import joblib
            import tempfile
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # scaler 저장
                scaler_path = os.path.join(tmpdir, "scaler.pkl")
                joblib.dump(train_dataset.scaler, scaler_path)
                mlflow.log_artifact(scaler_path, "preprocessors")
                
                # label_encoder 저장
                encoder_path = os.path.join(tmpdir, "label_encoder.pkl")
                joblib.dump(train_dataset.label_encoder, encoder_path)
                mlflow.log_artifact(encoder_path, "preprocessors")
            
        except Exception as e:
            ic(f"MLflow 모델 저장 실패: {e}")
        
        print(f"MLflow 실행 완료! Run ID: {run.info.run_id}")
        print(f"실험: {experiment_name}, 실행명: {next_run_name}")


def run_inference(data=None, batch_size=64, model_run_id=None, model_name=None):
    """
    추론 실행
    
    Args:
        data: 입력 데이터
        batch_size: 배치 크기
        model_run_id: 특정 실행 ID의 모델 사용
        model_name: 등록된 모델 이름 사용
    """
    with mlflow.start_run(run_name=f"inference_{model_name or 'latest'}") as run:
        # 모델 로드 방법 선택
        if model_run_id:
            # 특정 실행의 모델 로드
            model_uri = f"runs:/{model_run_id}/model"
            mlflow.log_param("model_source", "run_id")
            mlflow.log_param("source_run_id", model_run_id)
        elif model_name:
            # 등록된 모델의 최신 버전 로드
            model_uri = f"models:/{model_name}_model/latest"
            mlflow.log_param("model_source", "registered_model")
            mlflow.log_param("model_name", model_name)
        else:
            # 기존 체크포인트 방식 사용
            checkpoint = load_checkpoint()
            model, scaler, label_encoder = init_model(checkpoint)
            mlflow.log_param("model_source", "checkpoint")
        
        # MLflow에서 모델 로드하는 경우
        if 'model_uri' in locals():
            try:
                # MLflow에서 모델 로드 (실제 구현은 모델 타입에 따라 다름)
                # model = mlflow.pytorch.load_model(model_uri)
                # 현재는 기존 방식 유지
                checkpoint = load_checkpoint()
                model, scaler, label_encoder = init_model(checkpoint)
                mlflow.log_param("model_uri", model_uri)
            except Exception as e:
                ic(f"MLflow 모델 로드 실패, 체크포인트 사용: {e}")
                checkpoint = load_checkpoint()
                model, scaler, label_encoder = init_model(checkpoint)

        if data is None:
            data = []

        data = np.array(data)
        mlflow.log_param("input_data_shape", data.shape)
        mlflow.log_param("batch_size", batch_size)

        # 추론 실행
        recommend = inference(model, scaler, label_encoder, data, batch_size)
        print(recommend)
        
        # 추론 결과 메트릭
        mlflow.log_metric("num_recommendations", len(recommend))
        
        # 결과를 데이터베이스에 저장
        df_result = recommend_to_df(recommend)
        write_db(df_result, "mlops", "recommend")
        
        # 추론 결과를 아티팩트로 저장
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_result.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "predictions")
        
        print(f"추론 완료! Run ID: {run.info.run_id}")


if __name__ == '__main__':  # python main.py
    fire.Fire({
        "train": run_train,  # python main.py train --model_name disease_predictor
        "inference": run_inference,  # python main.py inference --model_name disease_predictor
    })