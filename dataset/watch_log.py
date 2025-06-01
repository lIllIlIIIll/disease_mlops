import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils.utils import project_path
from minio import Minio


class WatchLogDataset:
    def __init__(self, df, scaler=None, label_encoder=None):
        self.df = df
        self.features = None
        self.labels = None
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.contents_id_map = None
        self._preprocessing()

    def _preprocessing(self):
        # 결측값 처리
        self.df = self.df.drop(columns=['Unnamed: 133'])
        
        # content_id를 정수형으로 변환
        if self.label_encoder:
            self.df["prognosis"] = self.label_encoder.transform(self.df["prognosis"])
        else:
            self.label_encoder = LabelEncoder()
            self.df["prognosis"] = self.label_encoder.fit_transform(self.df["prognosis"])
        
        # content_id 디코딩 맵 생성
        self.prognosis_map = dict(enumerate(self.label_encoder.classes_))
        # 타겟 및 피처 정의
        target_columns = [col for col in self.df.columns if col != "prognosis"]
        self.labels = self.df["prognosis"].values
        features = self.df[target_columns].values

        # 피처 스케일링
        if self.scaler:
            self.features = self.scaler.transform(features)
        else:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features)

    @property
    def features_dim(self):
        return self.features.shape[1]

    @property
    def num_classes(self):
        return len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def decode_prognosis(self, idx):
        return self.prognosis_map[idx]

def read_dataset(use_minio=True, bucket_name="disease", object_name="disease/Training.csv"):
    if use_minio:
        # MinIO 클라이언트 생성
        client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9002"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minio1234"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minio1234"),
            secure=False
        )
        
        # CSV 데이터 로드
        try:
            response = client.get_object(bucket_name, object_name)
            df = pd.read_csv(response)
            response.close()
            response.release_conn()
            return df
        except Exception as e:
            print(f"MinIO 로드 실패: {e}")
            # 폴백: 로컬 파일
            watch_log_path = os.path.join(project_path(), "dataset", "Training.csv")
            return pd.read_csv(watch_log_path)
    else:
        # 기존 로컬 방식
        watch_log_path = os.path.join(project_path(), "dataset", "Training.csv")
        return pd.read_csv(watch_log_path)


def split_dataset(df):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
    return train_df, val_df, test_df


def get_datasets(scaler=None, label_encoder=None):
    df = read_dataset()
    train_df, val_df, test_df = split_dataset(df)
    train_dataset = WatchLogDataset(train_df, scaler, label_encoder)
    val_dataset = WatchLogDataset(val_df, scaler=train_dataset.scaler, label_encoder=train_dataset.label_encoder)
    test_dataset = WatchLogDataset(test_df, scaler=train_dataset.scaler, label_encoder=train_dataset.label_encoder)
    return train_dataset, val_dataset, test_dataset

