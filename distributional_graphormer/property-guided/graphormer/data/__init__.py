DATASET_REGISTRY = {}

def register_dataset(name: str):
    def register_dataset_func(func):
        DATASET_REGISTRY[name] = func()
    return register_dataset_func
