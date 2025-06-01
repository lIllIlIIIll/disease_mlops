from enum import Enum

from model.disease_predictor import DiseasePredictor


class CustomEnum(Enum):
    @classmethod
    def names(cls):
        return [member.name for member in list(cls)]

    @classmethod
    def validation(cls, name: str):
        names = [name.lower() for name in cls.names()]
        if name.lower() in names:
            return True
        else:
            raise ValueError(f"Invalid argument. Must be one of {cls.names()}")


class Models(CustomEnum):
    DISEASE_PREDICTOR = DiseasePredictor  # python main.py train --model_name disease_predictor
