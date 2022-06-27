import torchvision.transforms as tt
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
import torch
import config
import matplotlib.pyplot as plt
from torchvision.utils import make_grid



def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def show_batch(dl, stats):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        denorm_images = denormalize(images, *stats)
        ax.imshow(make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0,1))
        break

class ImageNormalization:
    """
    We will normalize the image tensors by subtracting the mean
    and dividing by the standard deviation across each channel.
    As a result, the mean of the data across each channel is 0,
    and standard deviation is 1. Normalizing the data prevents the values
    from any one channel from disproportionately affecting the
    losses and gradients while training, simply by having a higher
    or wider range of values that others
    """

    def __init__(self, train_image_folder, test_image_folder):
        self.train_image_folder = train_image_folder
        self.test_image_folder = test_image_folder

    def read_data(self):
        data = ImageFolder(
            self.train_image_folder,
            transform=tt.Compose(
                [
                    tt.Resize(config.CROP_SIZE),
                    tt.RandomCrop(config.CROP_SIZE),
                    tt.ToTensor(),
                ]
            ),
        )
        data_dl = DataLoader(data, config.CROP_SIZE, shuffle=True, num_workers=3, pin_memory=True)
        stats = self._get_mean_and_std(dataloader=data_dl)
        return data_dl, stats

    def normalize(self):
        """
        We will apply randomly chosen transformations
        while loading images from the training dataset.
        Specifically, we will pad each image by 4 pixels,
        and then take a random crop of size 64 x 64 pixels,
        and then flip the image horizontally with a 50% probability.
        Since the transformation will be applied
        randomly and dynamically each time a particular
        image is loaded, the model sees slightly different
        images in each epoch of training, which allows it generalize better.
        """
        _, stats = self.read_data()
        train_transform = tt.Compose(
            [
                tt.Resize(config.CROP_SIZE),
                tt.RandomCrop(config.CROP_SIZE),
                tt.RandomHorizontalFlip(),
                tt.ToTensor(),
                tt.Normalize(*stats, inplace=True),
            ]
        )
        test_transform = tt.Compose(
            [
                tt.Resize(config.CROP_SIZE),
                tt.RandomCrop(config.CROP_SIZE),
                tt.ToTensor(),
                tt.Normalize(*stats, inplace=True),
            ]
        )
        train = ImageFolder(self.train_image_folder, transform=train_transform)
        test = ImageFolder(self.test_image_folder, transform=test_transform)
        train_dl = DataLoader(train, config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        test_dl = DataLoader(test, config.BATCH_SIZE * 2, num_workers=2, pin_memory=True)
        return {"train_dataloader":train_dl, "test_dataloader":test_dl,"train":train,"test":test}

    def _get_mean_and_std(self, dataloader):
        sum_, squared_sum, batches = 0, 0, 0
        for data, _ in dataloader:
            sum_ += torch.mean(data, dim=([0, 2, 3]))
            squared_sum += torch.mean(data**2, dim=([0, 2, 3]))
            batches += 1

        mean = sum_ / batches
        std = (squared_sum / batches - mean**2) ** 0.5
        mean = tuple(mean.numpy())
        std = tuple(std.numpy())
        return (mean, std)


if __name__ == "__main__":
    imnorm = ImageNormalization(
        train_image_folder="../data/intel/seg_train/seg_train/",
        test_image_folder="../data/intel/seg_test/seg_test/",
    )
    cache = imnorm.normalize()

