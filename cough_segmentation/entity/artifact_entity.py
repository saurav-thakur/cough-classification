from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifacts:
    data_ingestion_dir: Path
    source_data_dir: Path
    source_manual_data: Path

@dataclass
class DataTransformationArtifacts:
    data_transformation_dir: Path
    audio_files: Path
    manual_labels: Path
    transformed_data_file: Path

@dataclass
class ModelTrainerArtifacts:
    model_trainer_dir: Path
    model_name: Path
