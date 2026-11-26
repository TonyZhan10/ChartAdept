"""Retain the final partial question batch of the training dataloader."""


def keep_tail(loader):
    return type(loader)(dataset=loader.dataset, batch_size=loader.batch_size,
        num_workers=loader.num_workers, drop_last=False, collate_fn=loader.collate_fn,
        sampler=loader.sampler)
