import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from topologylayer.nn import AlphaLayer, BarcodePolyFeature, PartialSumBarcodeLengths
from persim import plot_diagrams
from takenslayers import *

class DGMLoss:
    def __init__(self):
        # NOTE: These are actually H1, I just don't compute H0
        self.sum_h1_loss = PartialSumBarcodeLengths(dim=0, skip=0) 
        self.skip1_sum_h1_loss = PartialSumBarcodeLengths(dim=0, skip=1)
    
    def get_loss(self, dgms):
        sum_h1_loss = self.sum_h1_loss((dgms, True))
        skip1_sum_h1_loss = self.skip1_sum_h1_loss((dgms, True))
        largest_pers = sum_h1_loss-skip1_sum_h1_loss
        return largest_pers-skip1_sum_h1_loss, largest_pers, skip1_sum_h1_loss



class VideoAutoencoderCNN(nn.Module):
    def __init__(self, device, x, depth, dim, lr=1e-2, last_layer = nn.Sigmoid):
        """
        Same as the above class, except there's an additional loss term for persistent
        homology within overlapping blocks

        Parameters
        ----------
        x: tensor(n_frames, 1, imgres, imgres)
            Original video
        depth: int
            Depth of the CNN
        dim: int
            Dimension of the latent space
        lr: float
            Learning rate
        """
        super(VideoAutoencoderCNN, self).__init__()
        self.device = device
        self.x = x
        imgres = x.shape[-1]
        self.depth = depth
        self.dim = dim
        self.dim = dim
        
        ## Step 1: Create Convolutional Down Network
        self.convdown = nn.ModuleList()
        lastchannels = x.shape[1]
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
                nextchannels = x.shape[1]
            self.convup.append(nn.ConvTranspose2d(channels, nextchannels, 3, stride=2, padding=1, output_padding=1))
            channels = channels//2
            if i < depth-1:
                self.convup.append(nn.ReLU())
            else:
                self.convup.append(last_layer())
        
        # Optimizer / Loss functions
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        self.losses = []
    
    def forward(self):
        ## Step 1: Send video through the convolutional autoencoder
        y = self.x
        for layer in self.convdown:
            y = layer(y)
        for layer in self.convup:
            y = layer(y)
        return y
    
    def train_step(self):
        self.optimizer.zero_grad()
        self.train()
        y = self.forward()
        mse_loss = self.mse_loss(self.x, y)
        mse_loss.backward()
        self.optimizer.step()
        self.losses.append(mse_loss.item())
        return y, mse_loss.item()

    def train_epochs(self, num_epochs, plot_interval=0):
        self.losses = []
        plot_idx = 0
        res = 4
        plt.figure(figsize=(res*2, res*2))
        y = None
        mse_losses = []
        for epoch in range(num_epochs):
            y, mse_loss = self.train_step()
            ## Everything below this point is just for plotting
            mse_losses.append(mse_loss)
            
            if plot_interval > 0 and (plot_idx%plot_interval == 0 or plot_idx == num_epochs-1):
                plt.clf()
                plt.subplot2grid((2, 2), (0, 0), colspan=2)
                plt.title("Epoch {}: MSE Loss {:.6f}".format(epoch, mse_loss))
                plt.plot(mse_losses)
                plt.legend(["MSE Loss"])
                plt.scatter([epoch], [mse_loss], c='C0')
                plt.xlabel("Epoch")
                plt.ylabel("Loss Term")
                plt.gca().set_yscale('log')

                I = np.array(y.cpu().detach().numpy())
                Ix = I.shape[3]//2
                Iy = I.shape[2]//3

                plt.subplot(223)
                I0 = I[0, :, :, :]
                I0 = np.moveaxis(I0, (0, 1, 2), (2, 0, 1))
                plt.imshow(I0)
                plt.scatter([Ix], [Iy], c='r')
                plt.title("First Frame")

                plt.subplot(224)
                x = I[:, :, Iy, Ix]
                plt.plot(x[:, 0], c='r')

                plt.savefig("AutoencoderIter{}.png".format(plot_idx), facecolor='white')
            plot_idx += 1
        return y


class RipsRegularizedAutoencoderCNN(nn.Module):
    def __init__(self, device, x, depth, dim, win, lam=1, lr=1e-2, last_layer = nn.Sigmoid, bandpass_params = (0.5, 2, 30)):
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
        lastchannels = x.shape[1]
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
                nextchannels = x.shape[1]
            self.convup.append(nn.ConvTranspose2d(channels, nextchannels, 3, stride=2, padding=1, output_padding=1))
            channels = channels//2
            if i < depth-1:
                self.convup.append(nn.ReLU())
            else:
                self.convup.append(last_layer())
        
        ## Step 3: Create layers for sliding window
        self.bandpass = BandpassLayer(*bandpass_params)
        self.batch_norm = nn.BatchNorm2d(3)
        self.vid_dist = SlidingVideoDistanceMatrixLayer(win, device)
        self.rips = RipsPersistenceDistance([1])
        
        # Optimizer / Loss functions
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        self.dgm_loss = DGMLoss()
        self.losses = []
    
    def forward(self):
        y = self.x
        for layer in self.convdown:
            y = layer(y)
        for layer in self.convup:
            y = layer(y)
        yb = self.bandpass(y)
        z = self.batch_norm(yb)
        D = self.vid_dist(z)
        dgms = self.rips(D.cpu())
        return y, D, dgms
    
    def train_step(self):
        self.optimizer.zero_grad()
        self.train()
        y, D, dgms = self.forward()
        mse_loss = self.mse_loss(self.x, y)
        dgm_score, largest_pers, skip1_sum_h1_loss = self.dgm_loss.get_loss(dgms)
        loss = mse_loss - self.lam*dgm_score
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        return y, D, dgms, mse_loss.item(), self.lam*largest_pers.item(), self.lam*skip1_sum_h1_loss.item()

    def train_epochs(self, num_epochs, plot_interval=0):
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
            
            if plot_interval > 0 and (plot_idx%plot_interval == 0 or plot_idx == num_epochs-1):
                plt.clf()
                plt.subplot2grid((2, 5), (0, 0), colspan=3, rowspan=2)
                plt.title("Epoch {}: MSE Loss {:.6f}, DGM Score: {:.6f}, Total Loss: {:.6f}".format(epoch, mse_loss, dgm_score, self.losses[-1]))
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
                plt.gca().set_yscale('log')


                plt.subplot2grid((2, 5), (0, 3))
                dgms = dgms[0].detach().numpy()
                if dgms.size > 0:
                    plot_diagrams(dgms, labels=["H1"])

                plt.subplot2grid((2, 5), (0, 4))
                plt.imshow(D.cpu().detach().numpy())

                plt.subplot2grid((2, 5), (1, 3))
                I = y.cpu().detach().numpy()
                x = I[:, :, I.shape[2]//2, I.shape[3]//2]
                plt.plot(x[:, 0], c='r')
                #plt.plot(x[:, 1], c='g')
                #plt.plot(x[:, 2], c='b')
                print(x[:, 0])

                plt.subplot2grid((2, 5), (1, 4))
                if y.shape[1] == 1:
                    I = y.cpu().detach().numpy()[0, 0, :, :]
                    plt.imshow(I, cmap='gray', vmin=0, vmax=1)
                    plt.colorbar()
                else:
                    I = np.array(y.cpu().detach().numpy()[0, :, :, :])
                    I = np.moveaxis(I, [0, 1, 2], [2, 0, 1])
                    I -= np.min(I)
                    I /= np.max(I)
                    plt.imshow(I)
                plt.title("First frame")


                plt.savefig("Iter{}.png".format(plot_idx), facecolor='white')
            plot_idx += 1
        return y




class BlockRipsRegularizedAutoencoderCNN(nn.Module):
    def __init__(self, device, x, depth, dim, win, block_win, block_hop, lam_block=1, lam_var=1, lr=1e-2, last_layer = nn.Sigmoid, bandpass_params = (0.5, 2, 30)):
        """
        Same as the above class, except there's an additional loss term for persistent
        homology within overlapping blocks

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
        block_win: int
            Number of frames in a block
        block_hop: int
            Number of frames to jump between blocks
        lam_block: float
            Weight of block topological regularization
        lam_var: float
            Lambda in front of block variance term
        lr: float
            Learning rate
        """
        super(BlockRipsRegularizedAutoencoderCNN, self).__init__()
        self.device = device
        self.x = x
        imgres = x.shape[-1]
        self.depth = depth
        self.dim = dim
        self.win = win
        self.block_win = block_win
        self.block_hop = block_hop
        self.dim = dim
        self.lam_block = lam_block
        self.lam_var = lam_var
        
        ## Step 1: Create Convolutional Down Network
        self.convdown = nn.ModuleList()
        lastchannels = x.shape[1]
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
                nextchannels = x.shape[1]
            self.convup.append(nn.ConvTranspose2d(channels, nextchannels, 3, stride=2, padding=1, output_padding=1))
            channels = channels//2
            if i < depth-1:
                self.convup.append(nn.ReLU())
            else:
                self.convup.append(last_layer())

        ## Step 3: Create layers for sliding window of blocks
        self.blocks = []
        self.block_losses = []
        n_frames = x.shape[0] - bandpass_params[-1]
        i = 0
        while i + block_win <= n_frames:
            batch_norm = nn.BatchNorm2d(3)
            vid_dist =  SlidingVideoDistanceMatrixLayer(win, device)
            rips = RipsPersistenceDistance([1])
            self.blocks.append((batch_norm, vid_dist, rips))
            self.block_losses.append(DGMLoss())
            i += block_hop
        
        ## Step 4: Create layers for sliding window of overall video
        self.bandpass = BandpassLayer(*bandpass_params)
        self.batch_norm = nn.BatchNorm2d(3)
        self.vid_dist = SlidingVideoDistanceMatrixLayer(win, device)
        self.rips = RipsPersistenceDistance([1])
        self.dgm_loss = DGMLoss()

        print("Number of blocks: ", len(self.blocks))
        # Optimizer / Loss functions
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        self.losses = []
    
    def forward(self):
        ## Step 1: Send video through the convolutional autoencoder
        y = self.x
        for layer in self.convdown:
            y = layer(y)
        for layer in self.convup:
            y = layer(y)
        ## Step 2: Compute block persistence
        block_Ds = []
        block_dgms = []
        i = 0
        yb = self.bandpass(y)
        for (batch_norm, vid_dist, rips) in self.blocks:
            yi = yb[i:i+self.block_win, :, :, :]
            zi = batch_norm(yi)
            Di = vid_dist(zi)
            block_Ds.append(Di)
            dgmsi = rips(Di.cpu())
            block_dgms.append(dgmsi)
            i += self.block_hop
        return y, block_Ds, block_dgms
    
    def train_step(self):
        self.optimizer.zero_grad()
        self.train()
        y, block_Ds, block_dgms = self.forward()
        mse_loss = self.mse_loss(self.x, y)

        block_losses = torch.zeros((len(self.blocks)), device=self.device)
        for i, (dgmsi, block_lossi) in enumerate(zip(block_dgms, self.block_losses)):
            dgm_scorei, _, _ = block_lossi.get_loss(dgmsi)
            block_losses[i] = dgm_scorei
        mu = torch.mean(block_losses)
        block_stdev = torch.sqrt(torch.sqrt(torch.sum(torch.pow(block_losses-mu, 4))))

        loss = mse_loss + self.lam_var*block_stdev - self.lam_block*torch.mean(block_losses)

        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        return y, block_Ds, mse_loss.item(), block_losses, block_stdev.item()

    def train_epochs(self, num_epochs, plot_interval=0):
        self.losses = []
        plot_idx = 0
        res = 2.5
        plt.figure(figsize=(res*5, res*3.7))
        y = None
        mse_losses = []
        block_means = []
        for epoch in range(num_epochs):
            y, block_Ds, mse_loss, block_losses, block_stdev = self.train_step()
            block_losses = block_losses.cpu().detach().numpy()
            ## Everything below this point is just for plotting
            mse_losses.append(mse_loss)
            block_means.append(np.mean(block_losses)*self.lam_block)
            
            if plot_interval > 0 and (plot_idx%plot_interval == 0 or plot_idx == num_epochs-1):
                plt.clf()
                plt.subplot2grid((3, 5), (0, 0), colspan=3, rowspan=2)
                plt.title("Epoch {}: MSE Loss {:.6f}, Block STDev: {:.6f}\nBlock Means: {:.6g}, Total Loss: {:.6f}".format(epoch, mse_loss, block_stdev, block_means[-1], self.losses[-1]))
                plt.plot(mse_losses)
                plt.plot(block_means)
                plt.legend(["MSE Loss", "Block Mean Scores"])
                plt.scatter([epoch], [mse_loss], c='C0')
                plt.scatter([epoch], [block_means[-1]], c='C1')

                #plt.ylim([0, np.quantile(np.concatenate((mse_losses, dgm_losses, birth_losses)), 0.99)])
                plt.xlabel("Epoch")
                plt.ylabel("Loss Term")
                plt.gca().set_yscale('log')


                D = self.vid_dist(self.batch_norm(y))
                dgms = self.rips(D)[0].detach().numpy()

                plt.subplot2grid((3, 5), (0, 3))
                if dgms.size > 0:
                    plot_diagrams(dgms, labels=["H1"])
                plt.title("Persistence Diagrams")

                plt.subplot2grid((3, 5), (0, 4))
                plt.imshow(D.cpu().detach().numpy(), cmap='magma_r')
                plt.title("Self-Similarity Matrix")

                plt.subplot2grid((3, 5), (1, 3))
                plt.plot(self.block_hop*np.arange(len(self.blocks)), block_losses)
                plt.xlabel("Frame Index")
                plt.title("Block DGM Scores")

                plt.subplot2grid((3, 5), (2, 2), colspan=3)
                I = np.array(y.cpu().detach().numpy())
                Ix = I.shape[3]//2
                Iy = I.shape[2]//3
                x = I[:, :, Iy, Ix]
                plt.plot(x[:, 0], c='r')

                plt.subplot2grid((3, 5), (1, 4))
                idx = np.argmax(block_losses)*self.block_hop
                if y.shape[1] == 1:
                    I = y.cpu().detach().numpy()[idx, 0, :, :]
                    plt.imshow(I, cmap='gray', vmin=0, vmax=1)
                    plt.colorbar()
                else:
                    I = np.array(y.cpu().detach().numpy()[idx, :, :, :])
                    I = np.moveaxis(I, [0, 1, 2], [2, 0, 1])
                    I -= np.min(I)
                    I /= np.max(I)
                    plt.imshow(I)
                    plt.scatter([Ix], [Iy], c='r')
                plt.title("Frame Index {}".format(idx))
                plt.savefig("Iter{}.png".format(plot_idx), facecolor='white')
            plot_idx += 1
        return y


class SlidingAutoencoder(nn.Module):
    def __init__(self, x, dim, win, lam=1, lr=1e-2):
        """
        Parameters
        ----------
        x: ndarray(N)
            Time series
        dim: int
            Dimension of embedding
        win: int
            Window size
        lam: float
            Weight of topological regularization
        lr: float
            Learning rate
        """
        super(SlidingAutoencoder, self).__init__()
        self.x_orig = x
        self.x = torch.from_numpy(x.reshape(1, 1, -1)).float()
        self.win = win
        self.dim = dim
        self.lam = lam
        ## Part 1: Autoencoder
        ## TODO: This could be an RNN, and there could also be more layers
        self.linear1 = nn.Conv1d(1, 1, win+1, stride=1, padding='same')
        self.linear1_tanh = nn.Tanh()
        self.linear2 = nn.Conv1d(1, 1, win+1, stride=1, padding='same')
        self.linear2_tanh = nn.Tanh()
        self.batchnorm = nn.BatchNorm1d(1)
        ## Part 2: Sliding Window
        self.swlayer = SlidingWindowLayer(win=win, dim=dim, dT=2)# Stride 2 for speed
        self.flatten = nn.Flatten(start_dim=0, end_dim=1)
        self.transpose = Transpose()
        #self.znorm = ZNormalize()
        ## Part 3: TDA
        self.alpha = AlphaLayer(maxdim=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Loss functions and losses
        self.mse_loss = nn.MSELoss()
        self.sum_h1_loss = PartialSumBarcodeLengths(dim=1, skip=0)
        self.skip1_sum_h1_loss = PartialSumBarcodeLengths(dim=1, skip=1)
        self.losses = []
    
    def forward(self):
        """
        Returns
        -------
        y: The warped time series
        dgms: The persistence diagrams of the sliding widow embedding of the
              warped time series
        """
        y = self.linear1(self.x)
        y = self.linear1_tanh(y)
        y = self.linear2(y)
        y = self.linear2_tanh(y)
        sw = self.swlayer(self.batchnorm(y))
        sw = self.flatten(sw)
        sw = self.transpose(sw)
        #sw = self.znorm(sw)
        dgms = self.alpha(sw)
        return y, dgms
    
    def train_step(self):
        self.optimizer.zero_grad()
        self.train()
        y, dgms = self.forward()
        mse_loss = self.mse_loss(self.x, y)
        sum_h1_loss = self.sum_h1_loss(dgms)
        skip1_sum_h1_loss = self.skip1_sum_h1_loss(dgms)
        largest_pers = sum_h1_loss-skip1_sum_h1_loss
        dgm_score = largest_pers-skip1_sum_h1_loss
        loss = mse_loss - self.lam*dgm_score
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        return y, dgms, mse_loss.item(), self.lam*largest_pers.item(), self.lam*skip1_sum_h1_loss.item()
    
    def train_epochs(self, num_epochs, plot_interval=0):
        self.losses = []
        plot_idx = 0
        res = 4
        plt.figure(figsize=(res*5, res*2))
        y = None
        mse_losses = []
        largest_perses = []
        sum_other_persistences = []
        for epoch in range(num_epochs):
            y, dgms, mse_loss, largest_pers, sum_other_persistence = self.train_step()
            
            ## Everything below this point is just for plotting
            mse_losses.append(mse_loss)
            largest_perses.append(largest_pers)
            sum_other_persistences.append(sum_other_persistence)
            dgm_score = largest_pers-sum_other_persistence

            if plot_interval > 0 and (plot_idx%plot_interval == 0 or plot_idx == num_epochs-1):
                y = y.detach().numpy().flatten()
                plt.clf()
                plt.subplot2grid((2, 5), (0, 0), colspan=4)
                plt.plot(self.x_orig)
                plt.plot(y)
                plt.title("Epoch {}: MSE Loss {:.3f}, DGM Score: {:.3f}, Total Loss: {:.3f}".format(epoch, mse_loss, dgm_score, self.losses[-1]))
                plt.subplot2grid((2, 5), (1, 0), colspan=2)
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
                plt.subplot2grid((2, 5), (1, 2), colspan=1)
                layer1 = list(self.parameters())[0].detach().numpy().flatten()
                plt.plot(layer1)
                plt.title("Autoencoder Layer 1")
                plt.subplot2grid((2, 5), (1, 3), colspan=1)
                layer1 = list(self.parameters())[2].detach().numpy().flatten()
                plt.plot(layer1)
                plt.title("Autoencoder Layer 2")
                
                plt.subplot2grid((2, 5), (0, 4), colspan=1, rowspan=1)
                plot_diagrams(dgms[0][1].detach().numpy())
                plt.subplot2grid((2, 5), (1, 4), colspan=1, rowspan=1)
                plt.plot(self.losses)
                plt.scatter([epoch], [self.losses[-1]])
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Total Loss")
                plt.savefig("Iter{}.png".format(plot_idx))
            plot_idx += 1
        return y