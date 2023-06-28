import copy
import pytorch_lightning as pl
import torch
from torch.optim import Adam
import torchvision
from torch import nn
import torchvision.transforms as transforms
from lightly.loss import DINOLoss
from lightly.data import LightlyDataset
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum, activate_requires_grad
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
import wandb


class DINO(pl.LightningModule):
    def __init__(self):
        super().__init__()
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        input_dim = backbone.embed_dim

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        wandb.log({"pretraining_loss": loss})
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optim


class Supervised_trainer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        wandb.log({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        wandb.log({"val_loss": loss, "val_acc": acc})
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-5)
        return optimizer


def pretrain():
    print("starting pretraining")
    wandb.init(project='unsup pretraining')

    model = DINO()
    transform = DINOTransform()

    #is this important to have?: target_transform=lambda t: 0 (to ignore object detection)
    cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    crop_transform = DINOTransform(global_crop_size=196, local_crop_size=64)
    dataset = LightlyDataset.from_torch_dataset(cifar10, transform=crop_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(max_epochs=10, devices=1, accelerator=accelerator)
    trainer.fit(model=model, train_dataloaders=dataloader)

    wandb.finish()

    # we need to reactivate requires grad to perform supervised backpropagation later
    activate_requires_grad(model.teacher_backbone)
    return model.teacher_backbone

def create_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    print("train size: ", len(train_dataset), "validation size: ", len(val_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader

def supervised_train(model):
    print("starting sup training")
    wandb.init(project='sup training')

    train_loader, val_loader = create_datasets()
    sup_trainer = Supervised_trainer(model)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=10, devices=1, accelerator=accelerator, val_dataloaders=val_loader)

    # Train the model
    trainer.fit(sup_trainer, train_loader)

    wandb.finish()

if __name__ == '__main__':

    pretrained_model = pretrain()
    supervised_train(pretrained_model)
