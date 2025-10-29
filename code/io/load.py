from code.config import DATA_DIR
from pathlib import Path
import pandas as pd

def load(path: Path = DATA_DIR):
    print(path)
    file = path / 'test.csv'
    df = pd.read_csv(file)
    print(df.head())

if __name__ == "__main__":
    load()