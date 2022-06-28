import config
import model
import datasetup
import torch
import torch.nn as nn
from tqdm import tqdm


class ImageClassifier():

    def __init__(self, grad_clip=None):
        no_of_classes = len(cache["train"].classes)
        device = config.get_default_device()
        self.resnet = config.to_device(model.ResNet9(3, no_of_classes), device)
        self.opt_func = config.OPT_FUNC
        self.grad_clip = grad_clip
        self.criterion = nn.CrossEntropyLoss()

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def fit(self, train_dataloader):
        torch.cuda.empty_cache()
        optimizer = self.opt_func(self.resnet.parameters(), config.MAX_LR, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, config.MAX_LR, epochs=config.EPOCHS,
                                                    steps_per_epoch=len(train_dataloader))

        for epoch in range(config.EPOCHS):
            self.resnet.train()
            train_losses = []
            lrs = []
            for ix, batch in tqdm(
                    enumerate(train_dataloader), total=len(train_dataloader)
            ):

                data, target = batch
                output = self.resnet(data)
                loss = self.criterion(output, target)
                train_losses.append(loss)
                loss.backward()

                if self.grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), self.grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                # Record & update learning rate
                lrs.append(self.get_lr(optimizer))
                scheduler.step()
        torch.save(self.resnet.state_dict(), 'checkpoints/model_state.pt')
    def predict(self, test_dataloader):
        self.resnet.eval()
        final_outputs = []
        for ix, batch in tqdm(
                enumerate(test_dataloader), total=len(test_dataloader)
        ):
            data, target = batch
            output = self.resnet(data)
            probabilities = torch.exp(output.cpu()).detach.numpy()
            final_outputs.append(probabilities)

        return final_outputs


if __name__ == "__main__":
    imnorm = datasetup.ImageNormalization(
        train_image_folder="../data/intel/seg_train/seg_train/",
        test_image_folder="../data/intel/seg_test/seg_test/",
    )
    cache = imnorm.normalize()
    train_dl = cache['train_dataloader']
    clf = ImageClassifier()
    clf.fit(train_dl)
    out = clf.predict(cache['test_dataloader'])
    print(out)