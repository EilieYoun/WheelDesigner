import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_utils import *
from models import Encoder, Decoder

class WheelPreprocessV01(Dataset):
  def __init__(self, x_paths, y_labels,
               img_size=128, 
               hole_ratio=0.2, 
               use_rmbkg=True, 
               use_rotate=True):
    
    self.x_paths = x_paths
    self.y_labels = y_labels
    self.img_size = img_size
    self.hole_ratio = hole_ratio
    self.use_rmbkg = use_rmbkg
    self.use_rotate = use_rotate

  def __len__(self):
    return len(self.x_paths)

  def __getitem__(self, idx):
    # preprocess
    x = path2arr(self.x_paths[idx])
    x = tobinary(x)
    x = fill_hole(x, hole_ratio=self.hole_ratio)
    # aug - random zoom
    if self.use_rmbkg:
      crop_ratios = get_rnd_beta(0.55, 1.0) # follow beta fn.
      crop_ratio  = np.random.choice(crop_ratios)
      x = remove_bkg(x, crop_ratio=crop_ratio)
      x = zoom_and_resize(x, crop_ratio=crop_ratio)
    # aug - random rotate
    if self.use_rotate:
      angle = np.random.choice(range(5,350,5))
      x = rotate(x, angle=angle)
    x = torch.tensor(x, dtype=torch.float32)
    x = x.unsqueeze(0)
    y = self.y_labels[idx]
    
    return x, y

class WheelPreprocessV02(Dataset):
    def __init__(self, x_paths, y_labels,
               img_size=128,
               hole_ratio=0.2,
               rmbkg=1.0,
               rotate=0, 
               noise_param = None,
               z_noise_param = None):

        self.x_paths = x_paths
        self.y_labels = y_labels
        self.img_size = img_size
        self.hole_ratio = hole_ratio
        self.rmbkg = rmbkg
        self.rotate = rotate
        self.noise_param = noise_param
        self.z_noise_param = z_noise_param

        self.noise_shape = (2, 16, 16)
        self.encoder = Encoder(1, 2)
        self.decoder = Decoder(2)
        encoder_path = 'weights/encoder_v1_c2.pth'
        decoder_path = 'weights/decoder_v1_c2.pth'
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.encoder.eval()
        self.decoder.eval()
        
    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        x = path2arr(self.x_paths[idx])
        y = self.y_labels[idx]
        
        # img preprocess
        x = tobinary(x)
        x = fill_hole(x, hole_ratio=self.hole_ratio)
        
        # aug - random zoom
        if self.rmbkg == 'rnd':
            crop_ratios = get_rnd_beta(0.55, 1.0) # follow beta fn.
            crop_ratio  = np.random.choice(crop_ratios)
            x = remove_bkg(x, crop_ratio=crop_ratio)
            x = zoom_and_resize(x, crop_ratio=crop_ratio)
        elif self.rmbkg<=1.:
            x = remove_bkg(x, crop_ratio=self.rmbkg)
            x = zoom_and_resize(x, crop_ratio=self.rmbkg)
        else: pass
        
        # aug - random rotate
        if self.rotate == 'rnd':
            angle = np.random.choice(range(5,350,5))
            x = rotate(x, angle=angle, thres=.5)
        elif self.rotate>0: 
            x = rotate(x, angle=self.rotate, thres=.5)
        else: pass
        
        # get groundtruth image
        x_true = get_norm_img(x, y)

        # get volume image by groundtruth
        vol = np.mean(x_true)
        x_vol = get_volume_img(vol)

        # get slice image by slice y 
        x_sli = slice_dict[y]
        
        # noise
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        noise = torch.randn(self.noise_shape)
        
        #noise_param = np.clip(np.random.normal(3, 3), 1.0, 10.0))
        #z_param = np.clip(np.random.normal(3, 3), 1.0, 10.0))
         
        with torch.no_grad():
            if self.noise_param == 'rnd':
                z, _, _ = self.encoder(x, noise * np.clip(np.random.normal(3, 3), 1.0, 10.0))
            elif type(self.noise_param) == float:
                z, _, _ = self.encoder(x, noise * self.noise_param)
            else:
                z, _, _ = self.encoder(x, noise)
            if self.z_noise_param == 'rnd':
                x_noise = self.decoder(z*np.clip(np.random.normal(3, 3), 1.0, 10.0))
            elif type(self.z_noise_param) == float:
                x_noise = self.decoder(z*self.z_noise_param)
            else:
                x_noise = self.decoder(z)

        x_noise = torch.where(x_noise > .5, 1., 0.)
        
        # to tensor
        x_true = torch.tensor(x_true, dtype=torch.float32).unsqueeze(0)
        x_vol = torch.tensor(x_vol, dtype=torch.float32).unsqueeze(0)
        x_sli = torch.tensor(x_sli, dtype=torch.float32).unsqueeze(0)
        x = torch.cat((x_noise.squeeze(0), x_vol, x_sli), dim=0)
        return x, x_true, y


def get_loader(
               x_paths,
               y_labels,
               version='02',
               batch_size=32,
               img_size=128,
               hole_ratio=0.2,
               rmbkg='rnd',
               rotate='rnd',
               shuffle=False,
               noise_param=1.0,
               z_noise_param=1.0):
   
    if version=='01':
        pp = WheelPreprocessV01(x_paths, y_labels, img_size, hole_ratio, rmbkg, rotate)
    elif version=='02':
        pp = WheelPreprocessV02(x_paths, y_labels, img_size, hole_ratio, rmbkg, rotate, noise_param, z_noise_param)
    else: pass
    
    return  DataLoader(pp, batch_size=batch_size, shuffle=shuffle)
