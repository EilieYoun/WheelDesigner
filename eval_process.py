class WheelEvalProcess():
    def __init__(self, 
                path,
                encoder_path, 
                umap_path, 
                cand_path,
                cand_piece,
                cand_emb,
                device='cpu'
                ):
       
        self.path = path
        self.img_shape = (3, 128, 128)
        self.noise_shape = (4,16,16)
        self.img_size=self.img_shape[-1]
        self.encoder, _ = get_model(self.img_shape[0], self.noise_shape[0], device, encoder_path)
        self.encoder.eval()
      
        self.device=device
        with open(umap_path, 'rb') as f:
            self.umap_model = pickle.load(f)

        self.cand_path  = np.load(cand_path)
        self.cand_piece = np.load(cand_piece)
        self.cand_emb   = np.load(cand_emb)

    def path2process( self, path, 
                      hole_ratio=0.2, 
                      crop_ratio=0.83, 
                    ):

        x = path2arr(path)
        x = tobinary(x)
        x = fill_hole(x, hole_ratio)
        x = remove_bkg(x, crop_ratio)
        x = zoom_and_resize(x, resize=self.img_size, crop_ratio=crop_ratio)
        return x 
            
    
    def make_input(self, x, n_pieces):
        # get groundtruth image
        x_normed = get_norm_img(x, n_pieces)
        # get volume image by groundtruth
        vol = np.mean(x_normed)
        x_vol = get_volume_img(vol)
        # get slice image
        x_sli = slice_dict[n_pieces]

        # to tensor
        x_img = torch.tensor(x_normed, dtype=torch.float32).unsqueeze(0)
        x_vol = torch.tensor(x_vol, dtype=torch.float32).unsqueeze(0)
        x_sli = torch.tensor(x_sli, dtype=torch.float32).unsqueeze(0)
        x = torch.cat((x_img, x_vol, x_sli), dim=0).unsqueeze(0)
        x = x.to(self.device)
        return x 

    def eval(self, x_input):
        noise_input = torch.randn(self.noise_shape)
        
        with torch.no_grad():
            x_lat, _, _ = self.encoder(x_input, noise_input)
        
        x_lat = x_lat.numpy()
        x_emb = self.umap_model.transform(x_lat.reshape(1, -1))
        
        return x_emb

    def get_diff(self, a, b):
        return np.mean(np.sqrt((a - b)**2))

    def find_sim(self, x_img, x_emb, n_piece, n, img_w = 0.1):
        
        # fileter : same group, same piece
        piece_mask = self.cand_piece == n_piece
        cand_filtered_path = self.cand_path[piece_mask]
        cand_filtered_emb  = self.cand_emb[piece_mask]
        emb_diff = list(map(lambda cand_emb: self.get_diff(cand_emb, x_emb), cand_filtered_emb))
        sorted_idx = sorted(range(len(emb_diff)), key=lambda i: (emb_diff[i]), reverse=False)
        cand_sorted_path = [self.path+'/'+cand_filtered_path[idx] for idx in sorted_idx][:1000]
       
        cand_imgs = list(map(lambda x: self.path2process(x), cand_sorted_path))
        img_diff = list(map(lambda cand_img: self.get_diff(cand_img, x_img), cand_imgs))
        sorted_idx = sorted(range(len(img_diff)), key=lambda i: (img_diff[i]), reverse=False)
        cand_sorted_imgs = [cand_imgs[idx] for idx in sorted_idx][:n]

        return cand_sorted_imgs#, img_diff, emb_diff
