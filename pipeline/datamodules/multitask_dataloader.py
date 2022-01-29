import random
from torch.utils.data import DataLoader


class MultitaskDataLoader(DataLoader):

    def __init__(self, task_names, datasets, batch_size, num_workers, pin_memory, collate_fn, shuffle):
        self.task_names = task_names
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.lengths = [len(d) for d in datasets]
        indices = [[i] * v for i, v in enumerate(self.lengths)]
        self.task_indices = sum(indices, [])

    def _reset(self):
        random.shuffle(self.task_indices)
        self.current_index = 0

    def __iter__(self):
        self.iterators = [iter(d) for d in self.datasets]
        self._reset()
        return self

    def __len__(self):
        return sum(self.lengths)

    def __next__(self):
        if self.current_index < len(self.task_indices):
            task_index = self.task_indices[self.current_index]
            task_name = self.task_names[task_index]
            batch = next(self.iterators[task_index])
            new_batch = (batch, task_name)
            self.current_index += 1
            return new_batch
        else:
            raise StopIteration
