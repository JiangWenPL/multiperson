from functools import partial

from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler

from .sampler import GroupSampler, DistributedGroupSampler, DistributedSampler

# https://github.com/pytorch/pytorch/issues/973
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     **kwargs):
    shuffle = kwargs.get('shuffle', True)
    if dist:
        rank, world_size = get_dist_info()
        if shuffle:
            sampler = DistributedGroupSampler(dataset, imgs_per_gpu,
                                              world_size, rank)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        **kwargs)

    return data_loader


def build_dataloader_fuse(dataset, imgs_per_gpu, workers_per_gpu, num_gpus=1, drop_last=True, shuffle=True, dist=False,
                          **kwargs):
    batch_size = num_gpus * imgs_per_gpu
    num_workers = num_gpus * workers_per_gpu
    sampler = WeightedRandomSampler(dataset.density, len(dataset))

    print(f"Building dataloader with batch_size {batch_size}")
    if shuffle:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
            pin_memory=False,
            drop_last=drop_last,
            **kwargs)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
            pin_memory=False,
            drop_last=drop_last,
            **kwargs)

    return data_loader
