from src.schema.dataset import Dataset

def get_dataset_stats(dataset: Dataset):
    dataset_stats = {}
    dataset_stats['length'] = len(dataset)
    return dataset_stats