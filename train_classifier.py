import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchmetrics
from Dataset import MAMIDataset, collate
from params import *
from torch.utils.data import DataLoader
from torchvision import models, transforms
from metrics import *


def train(model, dataloader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model = model.to(device)
    for epoch in range(num_epochs):
        # train_acc = 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            all_batchs_loss = 0
            all_batchs_corrects = 0
            all_batchs_acc = 0
            all_batchs_prec = 0
            all_batchs_recall = 0
            all_batchs_fscore = 0
            counter = 0

            losses = AverageMeter('Loss', ':.2e')
            short_answer_acc = AverageMeter('Acc@Short', ':4.2f')
            precision_class = Precision_class(num_classes=54, average=True)
            recall_class = Recall_class(num_classes=54, average=True)

            progress = ProgressMeter(
                len(dataloader[phase]),
                [
                    losses,
                    short_answer_acc, precision_class, recall_class  #
                ],
                prefix="Epoch: [{}]".format(epoch))

            i = 0
            for inputs, _, _, _, labels, _, img_id in dataloader[phase]:

                this_batch_size = labels.size(0)
                i += 1

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                all_batchs_loss += loss.item() * inputs.size(0)
                all_batchs_corrects += torch.sum(preds == labels.data)

                losses.update(loss.item(), this_batch_size)
                this_short_answer_acc1 = accuracy_topk(outputs, labels, topk=(1,))
                short_answer_acc.update(this_short_answer_acc1[0].item(), this_batch_size)
                precision_class.update(outputs, labels)
                recall_class.update(img_id, outputs, labels)

                accuracy = torchmetrics.Accuracy()
                # train_acc += (outputs.argmax(1) == labels).cpu().numpy().mean()
                acc_score = accuracy(preds.cpu().detach(), labels.cpu().detach())
                # print('acc: {}'.format(acc_score))
                all_batchs_acc += acc_score
                # all_batchs_acc += train_acc
                # print(f'epoch_acc={train_acc}')

                f1 = torchmetrics.F1(num_classes=num_classes)
                fscore = f1(preds.cpu().detach(), labels.cpu().detach())
                # print('F1: {}'.format(f1(preds.cpu().detach(), labels.cpu().detach())))

                all_batchs_fscore += fscore

                counter += 1

            if phase == 'train':
                scheduler.step()

            progress.display(dataloader[phase].batch_size)

            epoch_loss = all_batchs_loss / counter
            # print(f"Testing accuracy: {train_acc/nb_batches}")
            print(f'epoch_loss={epoch_loss}')
            epoch_corrects = all_batchs_corrects.double() / counter
            print(f'epoch_corrects={epoch_corrects}')
            epoch_acc = all_batchs_acc / counter
            print(f'epoch_acc={epoch_acc}')
            # epoch_prec=all_batchs_prec / counter
            # print(f'epoch_precision={epoch_prec}')
            # epoch_recall=all_batchs_recall/ counter
            # print(f'epoch_recall={epoch_recall}')
            epoch_fscore = all_batchs_fscore / counter
            print(f'epoch_fscore={epoch_fscore}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'best_model_weight.pth')


if __name__ == '__main__':
    num_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    train_dataset = MAMIDataset(MAX_LEN, MAX_VOCAB, split='train')
    val_dataset = MAMIDataset(MAX_LEN, MAX_VOCAB, split='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,collate_fn=collate)

    dataloader = {
        'train': train_loader,
        'val': val_loader
    }

    num_classes = 6
    model = models.vgg16(pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    train(model, dataloader, num_epochs, device)

