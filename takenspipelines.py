import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from topologylayer.nn import BarcodePolyFeature, PartialSumBarcodeLengths
from persim import plot_diagrams
from takenslayers import *



class RipsRegularizedAutoencoderCNN(nn.Module):
    def __init__(self, device, x, depth, dim, win, lam=1, lr=1e-2):
        """
        Parameters
        ----------
        x: tensor(n_frames, 1, imgres, imgres)
            Original video
        depth: int
            Depth of the CNN
        dim: int
            Dimension of the latent space
        win: int
            Window length of sliding window
        lam: float
            Weight of topological regularization
        lr: float
            Learning rate
        """
        super(RipsRegularizedAutoencoderCNN, self).__init__()
        self.device = device
        self.x = x
        imgres = x.shape[-1]
        self.depth = depth
        self.dim = dim
        self.win = win
        self.dim = dim
        self.lam = lam
        
        ## Step 1: Create Convolutional Down Network
        self.convdown = nn.ModuleList()
        lastchannels = 1
        channels = 16
        for i in range(depth):
            self.convdown.append(nn.Conv2d(lastchannels, channels, 3, stride=2, padding=1))
            self.convdown.append(nn.ReLU())
            lastchannels = channels
            if i < depth-1:
                channels *= 2
        if dim > 0:
            flatten = nn.Flatten()
            res_down = int(imgres/(2**depth))
            flatten_dim = channels*res_down**2
            latentdown = nn.Linear(flatten_dim, dim)
            latentdown_relu = nn.ReLU()
            self.convdown += [flatten, latentdown, latentdown_relu]
        
        ## Step 2: Create Convolutional up layer
        self.convup = nn.ModuleList()
        if dim > 0:
            latentup = nn.Linear(dim, flatten_dim)
            latentup_relu = nn.ReLU()
            reshape = nn.Unflatten(1, (channels, res_down, res_down))
            self.convup += [latentup, latentup_relu, reshape]
        for i in range(depth):
            nextchannels = channels//2
            if i == depth-1:
                nextchannels = 1
            self.convup.append(nn.ConvTranspose2d(channels, nextchannels, 3, stride=2, padding=1, output_padding=1))
            channels = channels//2
            if i < depth-1:
                self.convup.append(nn.ReLU())
            else:
                self.convup.append(nn.Sigmoid())
        
        ## Step 3: Create layers for sliding window
        self.vid_dist = SlidingVideoDistanceMatrixLayer(win, device)
        self.rips = RipsPersistenceDistance([1])
        
        # Optimizer / Loss functions
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        # NOTE: These are actually H1, I just don't compute H0
        self.sum_h1_loss = PartialSumBarcodeLengths(dim=0, skip=0) 
        self.skip1_sum_h1_loss = PartialSumBarcodeLengths(dim=0, skip=1)
        self.losses = []
    
    def forward(self):
        y = self.x
        for layer in self.convdown:
            y = layer(y)
        for layer in self.convup:
            y = layer(y)
        D = self.vid_dist(y)
        dgms = self.rips(D.cpu())
        return y, D, dgms
    
    def train_step(self):
        self.optimizer.zero_grad()
        self.train()
        y, D, dgms = self.forward()
        mse_loss = self.mse_loss(self.x, y)
        sum_h1_loss = self.sum_h1_loss((dgms, True))
        skip1_sum_h1_loss = self.skip1_sum_h1_loss((dgms, True))
        largest_pers = sum_h1_loss-skip1_sum_h1_loss
        dgm_score = largest_pers-skip1_sum_h1_loss
        loss = mse_loss - self.lam*dgm_score
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        return y, D, dgms, mse_loss.item(), self.lam*largest_pers.item(), self.lam*skip1_sum_h1_loss.item()

    def train_epochs(self, num_epochs, do_plot=True):
        self.losses = []
        plot_idx = 0
        res = 4
        plt.figure(figsize=(res*5, res*2))
        y = None
        mse_losses = []
        largest_perses = []
        sum_other_persistences = []
        for epoch in range(num_epochs):
            y, D, dgms, mse_loss, largest_pers, sum_other_persistence = self.train_step()
            
            ## Everything below this point is just for plotting
            mse_losses.append(mse_loss)
            largest_perses.append(largest_pers)
            sum_other_persistences.append(sum_other_persistence)
            dgm_score = largest_pers-sum_other_persistence
            
            if do_plot and (plot_idx%40 == 0 or plot_idx == num_epochs-1):
                plt.clf()
                plt.subplot2grid((2, 5), (0, 0), colspan=3, rowspan=2)
                plt.title("Epoch {}: MSE Loss {:.3f}, DGM Score: {:.3f}, Total Loss: {:.3f}".format(epoch, mse_loss, dgm_score, self.losses[-1]))
                plt.plot(mse_losses)
                plt.plot(largest_perses)
                plt.plot(sum_other_persistences)
                plt.legend(["MSE Loss", "Largest Persistence", "Sum of Other Persistences"])
                plt.scatter([epoch], [mse_loss], c='C0')
                plt.scatter([epoch], [largest_pers], c='C1')
                plt.scatter([epoch], [sum_other_persistence], c='C2')
                #plt.ylim([0, np.quantile(np.concatenate((mse_losses, dgm_losses, birth_losses)), 0.99)])
                plt.xlabel("Epoch")
                plt.ylabel("Loss Term")


                plt.subplot2grid((2, 5), (0, 3))
                dgms = dgms[0].detach().numpy()
                if dgms.size > 0:
                    plot_diagrams(dgms, labels=["H1"])

                plt.subplot2grid((2, 5), (0, 4))
                plt.imshow(D.cpu().detach().numpy())

                plt.subplot2grid((2, 5), (1, 4))
                I = y.cpu().detach().numpy()[0, 0, :, :]
                plt.imshow(I, cmap='gray', vmin=0, vmax=1)
                plt.colorbar()
                plt.title("First frame")


                plt.savefig("Iter{}.png".format(plot_idx), facecolor='white')
            plot_idx += 1
        return y
