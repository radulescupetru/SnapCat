from data.snapcat_dataset import SnapCatDataset
from models.f_model import PtrNET
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.util_func import plot_image
import matplotlib.pyplot as plt



def validation(net, validation_loader, criterion):
    net.eval()

    valid_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(validation_loader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        valid_loss += loss.item()
        total = targets.size(0) * targets.size(1) * targets.size(2)
        _, predicted = torch.max(outputs, 1)
        correct = torch.sum(torch.eq(predicted, targets))

    print('Validation Loss: %.3f | Validation Acc: %.3f%%'
          % (valid_loss / (batch_idx + 1),
             100 * float(correct) / total))
    return valid_loss

def main():
    train_set = SnapCatDataset(inputs='inputs', targets='targets', img_dir="")

    # val_set = VocDataSet(filename="test_imgs", label_dir="data/raw/labels",
    #                      img_dir="data/raw/images")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10,
                                               shuffle=True, num_workers=4)

    # validation_loader = torch.utils.data.DataLoader(val_set, batch_size=10,
    #                                                 shuffle=False,
    #                                                 num_workers=4)
    # plot_image(train_set[2][0],train_set[2][1])
    if torch.cuda.is_available():
        net = PtrNET().cuda()
    else:
        net = PtrNET()
    if torch.cuda.device_count() > 1:
        device_ids = range(torch.cuda.device_count())
        net = nn.DataParallel(net, device_ids=device_ids)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    num_epochs = 500
    train_losses = []
    # validation_losses=[]
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0
        total = 0
        correct = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # loss = Variable(loss, requires_grad=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # total = targets.size(0)*targets.size(1)*targets.size(2)
            # _, predicted = torch.max(outputs, 1)
            # correct = torch.sum(torch.eq(predicted,targets))

        train_losses.append(train_loss)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        print('Results after epoch %d,' % (epoch + 1))

        print('Training Loss: %.3f '
              % (train_loss / (batch_idx + 1)))
        # val_loss = validation(net, validation_loader, criterion)
        # validation_losses.append(val_loss)
        # scheduler.step(val_loss)
    torch.save(net, 'models/saved_models/{}'.format("TEST_{}".format(1)))
    plt.figure()
    plt.plot(list(range(num_epochs)), train_losses)
    # plt.plot(list(range(num_epochs)),validation_losses)

if __name__ == "__main__":
    main()
    plt.show()
    # net = torch.load('models/saved_models/{}'.format("TEST_1"))
    # if torch.cuda.is_available():
    #     net = net.cuda()
    #
    # val_set = VocDataSet(filename="test_imgs", label_dir="data/raw/labels",
    #                      img_dir="data/raw/images")
    # image = val_set[10][0].cuda()
    # net.eval()
    # with torch.no_grad():
    #     image.unsqueeze_(0)
    #     outputs = net(image)
    # _, predicted = torch.max(outputs.data, 1)
    # outputs = predicted.view(predicted.size(1),predicted.size(2))
    # plt.imshow(outputs.cpu().numpy())
    # plt.show()