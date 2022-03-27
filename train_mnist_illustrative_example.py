
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR

logger = logging.getLogger('trainer')

file_log_handler = logging.FileHandler( './logs/logfile-mnist-illustrative-diffusion.log')
logger.addHandler(file_log_handler)

stderr_log_handler = logging.StreamHandler()
logger.addHandler(stderr_log_handler)

# nice output format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_log_handler.setFormatter(formatter)
stderr_log_handler.setFormatter(formatter)

logger.setLevel( 'DEBUG' )



#%pylab inline
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False, K=1, backbone='residual'):
        super(ConvBlock, self).__init__()
        
        self.f = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.g = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.K = K
        self.backbone = backbone

        self.act = nn.ReLU(inplace=True)   # nn.SiLU

        self.bn_out = nn.BatchNorm2d(c_out)
        self.bn_f1  = nn.BatchNorm2d(c_in)
        self.bn_g   = nn.BatchNorm2d(c_out)

    def forward(self, x):
        f = self.f(F.relu(self.bn_f1(x)))
        h = f 
        
        K = self.K 
        #f = F.elu( self.bn_r(self.r(h)) * F.relu( self.bn_g(self.g(h)) ) )  

        if self.backbone == 'cnn':
            h = self.g( F.relu( self.bn_g(h) ) )
        elif self.backbone == 'residual' :
            for k in range(K):
                h = h + self.g( F.relu( self.bn_g(h) ) )
            h = self.bn_out(h)
            h = self.act(h)
        else:        
            h0 = h #f #h
            #f = h
    
            g  = self.g( F.relu( self.bn_g(h) ) )
            g1 = g
            #g  = self.act( self.bng(self.convg(h)) )
            #g1 = self.act( self.bng1(self.convg1(h)) )
                
            dt = 0.2
            dx = 1.
            dy = 1.
    
            Dx = 1.
            Dy = 1.
            #Dx  = self.act( self.bnDx(self.convDx(h)) )
            #Dy  = self.act( self.bnDy(self.convDy(h)) )
    
            ux = (1. / (2*dx)) * ( torch.roll(g, 1, dims=2) - torch.roll(g, -1, dims=2) )
            vy = (1. / (2*dy)) * ( torch.roll(g1, 1, dims=3) - torch.roll(g1, -1, dims=3) )
    
            Ax = g  * (dt / dx)
            Ay = g1 * (dt / dy)
            Bx = Dx * (dt / (dx*dx))
            By = Dy * (dt / (dy*dy))
            E  = (ux + vy) * dt
    
            D = (1. / (1 + 2*Bx + 2*By))
    
            for k in range(self.K):
                prev_h = h
                    
                h = D  *   (   (1 - 2*Bx - 2*By) * h0 - 2 * E * h 
                             + (-Ax  + 2*Bx) * torch.roll(h, 1, dims=2) 
                             + ( Ax  + 2*Bx) * torch.roll(h, -1, dims=2) 
                             + (-Ay  + 2*By) * torch.roll(h, 1, dims=3)  
                             + ( Ay  + 2*By) * torch.roll(h, -1, dims=3)  
                             + 2 * dt * f )
                h0 = prev_h
            
            h = self.bn_out(h)
            h = self.act(h)

            #for k in range(K):
            #    #h = 0.5 * ( torch.roll(h, 1, dims=2) + torch.roll(h, 1, dims=3)  + f +  self.g( F.relu( self.bn_g(h) ) ))
            #    #h = 0.5 * ( torch.roll(h, 1, dims=2) + torch.roll(h, 1, dims=3)  + self.g( F.relu( self.bn_g(h) ) ))

        return h 


class NetworkMNIST(nn.Module):
    def __init__(self, backbone='residual', K=5):
        super(NetworkMNIST, self).__init__()
        self.conv = ConvBlock( 1, 1, 3, K=K, backbone=backbone )
        self.fc = nn.Linear(49, 10)

    def forward(self, x): 
        debug = False
        if debug: print('x = ', x.size())
            
        x = self.conv(x)
        f1 = x
        if debug: print('conv1 x = ', x.size())
        x = F.relu(x)

        x = F.avg_pool2d(x, 4)
        f2 = x
        if debug: print('avg_pool x = ', x.size())    
            
        x = torch.flatten(x, 1)
        if debug: print('x = ', x.size())
        
        x = self.fc(x)
        if debug: print('x = ', x.size())
        if debug: assert(1==2)
        
        output = F.log_softmax(x, dim=1)
        return output, f1, f2

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, f1, f2 = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())

            logger.info( msg )
            if args.dry_run:
                break
                
        #plt.figure()
        #plt.imshow(  f1[0, 0].detach().cpu() )
        #assert(1==2)


def test(model, device, test_loader, show=False):
    model.eval()
    test_loss = 0
    correct = 0
    
    predictions = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, f1, f2 = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            eql = pred.eq(target.view_as(pred))
            correct += eql.sum().item()
            
            if show:
                plt.figure()
                plt.imshow(  f1[0,0].detach().cpu() )
                plt.figure()
                plt.imshow(  f2[0,0].detach().cpu() )
                
            predictions.append( (data.detach().cpu(), target.detach().cpu(), 
                                 f1.detach().cpu(), 
                                 f2.detach().cpu(),
                                 eql.detach().cpu() ) )

    test_loss /= len(test_loader.dataset)

    msg = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    logger.info( msg )

    return predictions

def main(model_name = 'cnn'):
    msg = '\n\n Model = ' +  model_name
    logger.info( msg )
    start_time = time.time()
    
    args = parser.parse_args("")
    logger.info( args )
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    #root = '/home/anilkag/code/compact-vision-nets-PDE-Feature-Generator/data/'
    #root = '/media/anilkag/drive/anilkag/code/ODE-RNN/Decoupled-RNN/FloatingRNNs/data/mnist/'
    root = '/media/anilkag/9e70dcae-9db4-48cf-b1e6-844e3c424102/anilkag/code/ODE-RNN/Decoupled-RNN/FloatingRNNs/data/mnist/'
    dataset1 = datasets.MNIST(root, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(root, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if model_name == 'cnn':
        model = NetworkMNIST(backbone='cnn').to(device)
    elif model_name == 'residual':
        model = NetworkMNIST(backbone='residual', K=1).to(device)
    else:
        model = NetworkMNIST(backbone='pde').to(device)
        
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    msg = 'Total params = ' + str(total_params)
    logger.info( msg )
    
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
        
        msg = 'lr = ' + str(optimizer.param_groups[0]['lr'])
        logger.info( msg )

    predictions = test(model, device, test_loader, show=False)
    #if args.save_model:
    torch.save(model.state_dict(), "./models/mnist_"+ model_name +"-v2.pt")
        
    msg = "--- {}s seconds ---".format(time.time() - start_time)
    logger.info( msg )
    return predictions

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                    help='input batch size for training (default: 64)')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')

#parser.add_argument('--epochs', type=int, default=200, metavar='N',
parser.add_argument('--epochs', type=int, default=30, metavar='N',
#parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 14)')


#parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
#                    help='learning rate (default: 1.0)')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 1.0)')

parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
   
 
#cnn_predictions = main(model_name = 'cnn')
#residual_predictions = main(model_name = 'residual')
pde_predictions =  main(model_name = 'pde')
