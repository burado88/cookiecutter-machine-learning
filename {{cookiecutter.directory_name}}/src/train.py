import xgboost
import yaml
from pathlib import Path

project_path = Path(__file__).resolve().parents[1]

with open(project_path / "xgboost_params.yaml", "r") as f:
    xgboost_params = yaml.full_load(f)