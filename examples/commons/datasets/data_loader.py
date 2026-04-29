import torch
from torch.utils.data import DataLoader


def get_data_loader(
    dataset: torch.utils.data.Dataset,
    pin_memory: bool = False,
) -> DataLoader:
    loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=pin_memory,
        collate_fn=lambda x: x,
    )
    return loader

# 多进程加载数据，解决单进程加载数据过慢的问题
def get_data_loader_multi_worker(
    dataset: torch.utils.data.Dataset,
    pin_memory: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 3,
) -> DataLoader:
    def worker_init_fn(worker_id):
        if hasattr(dataset, 'set_worker_id'):
            dataset.set_worker_id(worker_id, num_workers)
            if hasattr(dataset, '_shuffle_batch'):
                dataset._shuffle_batch(worker_id=worker_id)

    loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=pin_memory,
        collate_fn=lambda x: x,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    return loader