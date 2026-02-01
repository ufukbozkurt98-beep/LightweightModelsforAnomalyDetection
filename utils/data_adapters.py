import torch


class TupleLoader:
    """
    Wraps a loader and always yields (image, label, mask).
    If the loader returns a dict, it is converted to this tuple format.
    """

    def __init__(self, loader):
        self.loader = loader  # Store the original DataLoader

    def __len__(self):
        return len(self.loader)  # Return the number of batches in the original loader.

    def __iter__(self):
        for batch in self.loader:
            # If the batch is a dict, convert it to (image, label, mask)
            if isinstance(batch, dict):
                image = batch["image"]
                label = batch.get("label", torch.zeros(image.size(0), dtype=torch.long))
                mask = batch.get("mask", torch.zeros(image.size(0), 1, image.size(2), image.size(3)))
                yield image, label, mask
            # If it is already a tuple, return it as-is
            else:
                yield batch
