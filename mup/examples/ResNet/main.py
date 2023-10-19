'''Train CIFAR10 with PyTorch.'''
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from mup.coord_check import get_coord_data, plot_coord_data
from mup import MuAdam, MuSGD, get_shapes, make_base_shapes, set_base_shapes

import resnet


def coord_check(mup, lr, optimizer, nsteps, arch, base_shapes, nseeds, device='cuda', plotdir='', legend=False):

    optimizer = optimizer.replace('mu', '')

    def gen(w, standparam=False):
        def f():
            model = getattr(resnet, arch)(wm=w).to(device)
            if standparam:
                set_base_shapes(model, None)
            else:
                set_base_shapes(model, base_shapes)
            return model
        return f

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='../dataset', train=True, download=True, transform=transform_train)
    dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=False)

    widths = 2**np.arange(-2., 2)
    models = {w: gen(w, standparam=not mup) for w in widths}
    df = get_coord_data(models, dataloader, mup=mup, lr=lr, optimizer=optimizer, nseeds=nseeds, nsteps=nsteps)

    prm = 'μP' if mup else 'SP'
    plot_coord_data(df, legend=legend,
        save_to=os.path.join(plotdir, f'{prm.lower()}_{arch}_{optimizer}_coord.png'),
        suptitle=f'{prm} {arch} {optimizer} lr={lr} nseeds={nseeds}',
        face_color='xkcd:light grey' if not mup else None)


# Training
def train(epoch, net):
    from utils import progress_bar
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, net):
    from utils import progress_bar
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=''
    '''
    PyTorch CIFAR10 Training, with μP.

    To save base shapes info, run e.g.

        python main.py --save_base_shapes resnet18.bsh --width_mult 1

    To train using MuAdam (or MuSGD), run

        python main.py --width_mult 2 --load_base_shapes resnet18.bsh --optimizer {muadam,musgd}

    To test coords, run

        python main.py --load_base_shapes resnet18.bsh --optimizer sgd --lr 0.1 --coord_check

        python main.py --load_base_shapes resnet18.bsh --optimizer adam --lr 0.001 --coord_check

    If you don't specify a base shape file, then you are using standard parametrization, e.g.

        python main.py --width_mult 2 --optimizer {muadam,musgd}

    Here muadam (resp. musgd) would have the same result as adam (resp. sgd).

    Note that models of different depths need separate `.bsh` files.
    ''', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--optimizer', default='musgd', choices=['sgd', 'adam', 'musgd', 'muadam'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--width_mult', type=float, default=1)
    parser.add_argument('--save_base_shapes', type=str, default='',
                        help='file location to save base shapes at')
    parser.add_argument('--load_base_shapes', type=str, default='',
                        help='file location to load base shapes from')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--test_num_workers', type=int, default=2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--coord_check', action='store_true',
                        help='test μ parametrization is correctly implemented by collecting statistics on coordinate distributions for a few steps of training.')
    parser.add_argument('--coord_check_nsteps', type=int, default=3,
                        help='Do coord check with this many steps.')
    parser.add_argument('--coord_check_nseeds', type=int, default=1,
                        help='number of seeds for coord check')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    # Data
    if not args.save_base_shapes:
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='../dataset', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR10(
            root='../dataset', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_num_workers)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')

    if args.coord_check:
        print('testing parametrization')
        import os
        os.makedirs('coord_checks', exist_ok=True)
        plotdir = 'coord_checks'
        coord_check(mup=True,
            lr=args.lr, optimizer=args.optimizer, nsteps=args.coord_check_nsteps, arch=args.arch, base_shapes=args.load_base_shapes, nseeds=args.coord_check_nseeds, device=device, plotdir=plotdir, legend=False)
        coord_check(mup=False,
            lr=args.lr, optimizer=args.optimizer, nsteps=args.coord_check_nsteps, arch=args.arch, base_shapes=args.load_base_shapes, nseeds=args.coord_check_nseeds, device=device,plotdir=plotdir, legend=False)
        import sys; sys.exit()


    # Model
    print('==> Building model..')
    net = getattr(resnet, args.arch)(wm=args.width_mult)
    if args.save_base_shapes:
        print(f'saving base shapes at {args.save_base_shapes}')
        base_shapes = get_shapes(net)
        delta_shapes = get_shapes(getattr(resnet, args.arch)(wm=args.width_mult/2))
        make_base_shapes(base_shapes, delta_shapes, savefile=args.save_base_shapes)
        # save_shapes(net, args.save_base_shapes)
        print('done and exit')
        import sys; sys.exit()

    net = net.to(device)

    if args.load_base_shapes:
        print(f'loading base shapes from {args.load_base_shapes}')
        set_base_shapes(net, args.load_base_shapes)
        print('done')
    else:
        print(f'using standard parametrization')
        set_base_shapes(net, None)
        print('done')

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'musgd':
        optimizer = MuSGD(net.parameters(), lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    elif args.optimizer == 'muadam':
        optimizer = MuAdam(net.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise ValueError()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch, net)
        test(epoch, net)
        scheduler.step()