import torch
import numpy as np
from torchsummary import summary
from models import Encoder, Decoder
from data_loader import get_loader
from data_utils import show_imgs
import torch.optim.lr_scheduler as lr_scheduler
from losses import get_kl_loss, get_rec_loss, get_rot_loss
torch.cuda.empty_cache()


def train(x_paths, y_labels, image_shape, noise_shape, lr, bs, epochs, save_encoder, save_decoder):
    # set init 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_ch, img_size, _ = image_shape
    noise_ch = noise_shape[0]

    # model 
    encoder = Encoder(img_ch, noise_ch).to(device)
    decoder = Decoder(noise_ch).to(device)
    
    # data
    loader = get_loader(x_paths,
                       y_labels,
                       version='02',
                       batch_size=bs,
                       img_size=img_size,
                       hole_ratio=0.2,
                       rmbkg='rnd',
                       rotate='rnd',
                       shuffle=True)
    # opt
    optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=lr)
    
    # train loop
    mn_loss = np.inf
    encoder.train()
    decoder.train()
    for i in range(epochs):
        total_loss = 0.
        for bi, (x, x_true, y) in enumerate(loader):
            x = x.to(device)
            x_true = x_true.to(device)
            y = y.to(device)

            # Forward pass
            noise = torch.randn(noise_shape).to(device)
            z, mu, log_var = encoder(x, noise)
            x_recon = decoder(z)
            
            # Set loss        
            kl_loss = get_kl_loss(mu, log_var, divide=img_size*bs)
            rec_loss = get_rec_loss(x_true, x_recon, divide=img_size*bs)
            #rot_loss = get_rot_loss(x_recon, y, divide=img_size*bs)
            loss = (kl_loss*0.1)  + rec_loss 
            total_loss+=loss
            
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

            if bi%10==0:
                print(f"Iteration {i}, batch {bi},  Loss: {loss.item():.4f}", end=' ')
                print(f"rec : {rec_loss.item():.4f}", end=' ')
                print(f"kl  : {kl_loss.item():.4f}", end=' ')
                print()
            if bi%100==0:
                x_cp = x.detach().cpu().numpy()
                x_recon_cp = x_recon.detach().cpu().numpy()
                x_true_cp = x_true.detach().cpu().numpy()
                show_imgs(x_recon_cp[:8,0])
                show_imgs(x_cp[:8,0])
                show_imgs(x_true_cp[:8,0])

        if rec_loss < mn_loss:
            torch.save(encoder.state_dict(), save_encoder)
            torch.save(decoder.state_dict(), save_decoder)
            mn_loss = total_loss
            print('* save model! ')
        print(total_loss, mn_loss)    
