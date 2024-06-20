# coding: utf-8

import logging
import os
from os import listdir

import hydra
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.nn.modules import loss
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

from dataset import GlyphData
from model import Glyphnet
import swanlab


def train(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
          loss_function: nn.Module, current_epoch_number: int = 0,
          device: torch.device = None, batch_reports_interval: int = 100):
    """ Training a provided model using provided data etc. """
    model.train()
    loss_accum = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        # throwing away the gradients
        optimizer.zero_grad()

        # predicting scores
        output = model(data.to(device))

        # computing the error
        loss = loss_function(output, target.to(device))

        # saving loss for stats
        loss_accum += loss.item() / len(data)

        # computing gradients
        loss.backward()

        # updating the model's weights
        optimizer.step()

        if batch_idx % batch_reports_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tAveraged Epoch Loss: {:.6f}'.format(
                current_epoch_number + 1,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_accum / (batch_idx + 1)))
            # swanlab.log({"loss_accum":loss_accum})


def softmax2predictions(output: torch.Tensor) -> torch.Tensor:
    """ model.predict(X) based on softmax scores """
    return torch.topk(output, k=1, dim=-1).indices.flatten()


def test(model: nn.Module, test_loader: DataLoader, loss_function: nn.Module, device):
    """ Testing an already trained model using the provided data from `test_loader` """
    global MAX_Accuracy
    model.eval()
    test_loss, correct = 0, 0
    all_predictions, all_gold = [], []

    with torch.no_grad():
        for data, target in test_loader:

            # getting y_true
            target = target.to(device)

            # getting y_pred
            output = model(data.to(device))
            pred = softmax2predictions(output)

            # accumulating loss and accuracy
            test_loss += loss_function(output, target).sum().item()
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_predictions.append(pred.cpu().numpy())
            all_gold.append(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    # swanlab.log({"test_loss":test_loss})
    # swanlab.log({"Accuracy_number":correct})

    logging.info('    Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_gold)

    logging.info("    Acc.: %2.2f%%; F-macro: %2.2f%%\n" % (accuracy_score(y_true, y_pred) * 100,
                                                         f1_score(y_true, y_pred, average="macro") * 100))
    
    return accuracy_score(y_true, y_pred) * 100
    # swanlab.log({"F-macro":f1_score(y_true, y_pred, average="macro") * 100})
    # swanlab.log({"Acc":accuracy_score(y_true, y_pred) * 100})

@hydra.main("configs", "config")
def main(cfg):
    MAX_Accuracy = 0

    # preparing data directories for processing
    train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
    test_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.test_path)

    train_labels = {l: i for i, l in enumerate(sorted([p.strip("/") for p in listdir(train_path)]))}

    train_set = GlyphData(root=train_path, class_to_idx=train_labels,
                          transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                        transforms.ToTensor()]))

    logging.info("Splitting data...")

    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_set)),
        train_set.targets,
        # stratify=train_set.targets,
        test_size=cfg.data.val_fraction,
        shuffle=True,
        random_state=cfg.model.seed
    )

    train_loader = torch.utils.data.DataLoader(Subset(train_set, train_indices),
                                               batch_size=cfg.model.batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(Subset(train_set, val_indices),
                                             shuffle=False,
                                             batch_size=128)

    logging.info(f"CUDA available? {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Setting up a model...")
    model = Glyphnet(num_classes=len(train_labels),
                     first_conv_out=cfg.model.first_convolution_filters,
                     last_sconv_out=cfg.model.last_separable_convolution_filters,
                     sconv_seq_outs=cfg.model.inner_separable_convolution_filters_seq,
                     dropout_rate=cfg.model.dropout
                     ).to(device)

    if cfg.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(model.parameters())
    elif cfg.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)
    else:
        raise Exception(f"Unknown optimizer [{cfg.optimizer.name}]!")

    loss_function = loss.CrossEntropyLoss()
    logging.info("Starting training...")

    for epoch in range(cfg.model.epochs):
        train(model, train_loader, optimizer, loss_function, epoch, device)
        logging.info("Evaluation on development set:")
        test(model, val_loader, loss_function, device)

        logging.info("Goodness of fit (evaluation on train):")
        Acc = test(model, train_loader, loss_function, device)
        if Acc> MAX_Accuracy:
            MAX_Accuracy=Acc
            torch.save(model.state_dict(), 'C:/Users/94427/kashiwa/glyphnet-pytorch/checkpoint/best_checkpoint.pth')
            print("最优权重已更新")


        # print(model.state_dict())
   

    #  FINAL EVALUATION

    test_labels_set = {l for l in [p.strip("/") for p in listdir(test_path)]}
    test_labels = {k: v for k, v in train_labels.items() if k in test_labels_set}

    test_set = GlyphData(root=test_path, class_to_idx=test_labels,
                         transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                       transforms.ToTensor()]))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    model.eval()

    logging.info("Checking quality on test set:")
    test(model, test_loader, loss_function, device)
    
    # if not os.path.exists("checkpoint"):
    #     os.makedirs("checkpoint")
    #     print("1")
    # torch.save(model.state_dict(), 'checkpoint/latest_checkpoint.pth')
    # print("1")
    # torch.save(model, "checkpoint.pth")

    logging.info("Saving the trained model.")
    


if __name__ == "__main__":
    
    # run = swanlab.init(
    #     project="Ancient-Egyptian-Hieroglyphs-Classification",
    #     experiment_name="101",
    #     config={
    #         "model": "Glyphnet",
    #         "optim": "Adam",
    #         "lr": 1e-4,
    #         "batch_size": 256,
    #         "num_epochs": 10,
    #     },
    # )

    main()
    
