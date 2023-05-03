from modules import *

class Encoder(nn.Sequential):
    def __init__(self, img_channel, noise_channel):
        self.img_channel = img_channel
        self.noise_channel=noise_channel
        super().__init__(
            nn.Conv2d(self.img_channel, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, int(self.noise_channel*2), kernel_size=3, padding=1),
            nn.Conv2d(int(self.noise_channel*2), int(self.noise_channel*2), kernel_size=1, padding=0),
        )

    def forward(self, x, noise):
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        variance = log_var.exp()
        stdev = variance.sqrt()
        x = mean + stdev * noise

        x *= 0.18215
        return x, mean, log_var

class Classifier():
    def forard(self, z_slices, z_volumes):
        x 

class Decoder(nn.Sequential):
    def __init__(self, noise_channel):
        self.noise_channel = noise_channel
        super().__init__(
            nn.Conv2d(self.noise_channel, self.noise_channel, kernel_size=1, padding=0),
            nn.Conv2d(self.noise_channel, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x /= 0.18215
        for module in self:
            x = module(x)
        return x

def get_model(img_dims, noise_dims, load_encoder=None, load_decoder=None):
    encoder = Encoder(img_dims, noise_dims)
    decoder = Decoder(noise_dims)
    
    if load_encoder is not None:
        encoder.load_state_dict(torch.load(load_encoder))
    if load_decoder is not None:
        decoder.load_state_dict(torch.load(load_decoder))

    return encoder, decoder
