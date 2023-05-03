import matplotlib.pyplot as plt
import numpy as np
import cv2

def preprocess(path, resize=128, hole_ratio=0.2, crop_ratio=1.0):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = tobinary(x)
    x = zoom_and_resize(x, resize, crop_ratio)
    x = remove_bkg(x, crop_ratio)
    x = fill_hole(x, hole_ratio)
    return x

def show_imgs(imgs=[], titles=[], r=1, cmap='gray', axis=False, vrange=[0.,1.]):  
    c = len(imgs) // r
    plt.figure(figsize=(c*3, r*3))
    for i in range(len(imgs)):
        plt.subplot(r, c, i+1)
        if bool(vrange): plt.imshow(imgs[i], cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        else: plt.imshow(imgs[i], cmap=cmap)
        if len(titles) == len(imgs):
            plt.title(titles[i])
        if not axis:
            plt.axis('off')
    plt.show()

def path2arr(path):
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  img_arr = np.array(img)
  return img_arr

def tobinary(arr):
  x = np.copy(arr)
  x = x.astype(np.float32) / 255.0
  mean = np.mean(x)
  std = np.std(x)
  # z-score normalization
  x = (x - mean) / std
  x = np.where(x > 0, 0.0, 1.0) 
  return x

def fill_hole(arr, hole_ratio=0.2):
  x = np.copy(arr)
  img_size = x.shape[1]
  hole_mask = get_circle(hole_ratio, img_size=img_size) 
  x +=hole_mask
  x = np.where(x > 0., 1.0, 0.0) 
  return x

def cal_img_diff(img1, img2):
  diff = np.sum(np.abs(img1 - img2)) / img1.size
  return diff
    
def get_circle(r, img_size=128, center=(0,0), isbool=False):
  x = np.linspace(-1, 1, img_size)[:,None]
  y = np.linspace(-1, 1, img_size)[None,:]
  d = np.sqrt((x-center[0])**2 + (y-center[1])**2) 
  if not isbool:
    return np.where(d <= r, 1, 0)
  else: 
    return d<=r

def remove_bkg(arr, crop_ratio=0.8):
  x = np.copy(arr)
  img_size=x.shape[1]
  crop_mask = get_circle(crop_ratio, img_size=img_size, isbool=True)
  x[crop_mask != True] = 0.
  return x

def zoom_and_resize(arr, resize=128, crop_ratio=0.8):
  x = np.copy(arr)
  img_size = x.shape[1]
  center = int(img_size//2)
  crop_size = int(img_size * crop_ratio)
  start = center - crop_size // 2
  crop_slice = np.s_[start:start+crop_size, start:start+crop_size]
  cropped = x[crop_slice]
  resized = cv2.resize(cropped, dsize=(resize, resize), interpolation=cv2.INTER_AREA)
  resized = np.where(resized > 0., 1.0, 0.0)
  return resized
  
get_angle = lambda x : 360 // x

def rotate(arr, angle, thres=None):
  (h, w) = arr.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  img_rotated = cv2.warpAffine(arr, M, (w, h), borderValue=(0.))
  img_rotated = cv2.GaussianBlur(img_rotated, (5, 5), 0)
  if thres is not None:
    img_rotated = np.where(img_rotated > thres, 1.0, 0.0)
  return img_rotated


def get_rnd_beta(mn, mx, a=2., b=1.2, size=1000):
    z = (mx-mn)/mx
    xs = (np.random.beta(a, b, size=size))*z+mn
    return xs

def get_slice_label(ds_paths):
  labels = []
  for path in ds_paths:
    diffs = {}
    slices = range(2,16)
    for s in slices:
      img = path2arr(path)
      img = tobinary(img)
      angle = 360  / s
      rot = rotate(img, angle)      
      diff = cal_img_diff(img, rot)
      diffs[s]=diff
    label = min(diffs, key=diffs.get)
    labels.append(label)
  return labels

def get_norm_img(img, slices, thres=0.65):
    norm_img = np.zeros(shape=img.shape)
    for i in range(1, 1+slices):
        angle = 360 * i / slices
        rot_img = rotate(img, angle)
        norm_img+=rot_img
    norm_img /=slices
    norm_img = np.where(norm_img > thres, 1.0, 0.0)
    return norm_img

def get_volume_img(volume, radius_ratio=1., img_size=128):
    image = np.full((img_size, img_size), 0.)
    center = img_size / 2.
    y, x = np.indices((img_size, img_size))
    distances = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    image[distances < center * radius_ratio] = volume/0.78
    return image

def get_slice_img(slices, std, image_size = (128, 128), radius_ratio=1.0):
    center = (image_size[0] // 2, image_size[1] // 2)
    angle_weights = [1.0/slices for x in range(slices)]  

    image = np.zeros(image_size)

    y, x = np.indices(image_size)
    distances = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    angles = np.arctan2(y - center[0], x - center[1]) * 180 / np.pi

    for i in range(len(angle_weights)):
        angle = i * (360 / slices)
        angle_diff = np.abs(angles - angle)
        angle_diff = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)
        weight = np.exp(-((angle_diff ** 2) / (2 * std ** 2)))  
        weight = weight * angle_weights[i]
        image = np.maximum(image, weight)  
    image[distances >= center[0] * radius_ratio] = 0
    return image

slice_dict = {x : get_slice_img(x, 360/x/4) for x in range(2,20)}

def get_wheel_img(img, img_size, new_img_size, hole_ratio):
    x = np.zeros((new_img_size, new_img_size))
    s = (new_img_size-img_size)//2
    x[s:new_img_size-s, s:new_img_size-s] = img
    o1 = get_circle(1., img_size=new_img_size)
    o2 = get_circle((img_size/new_img_size)*0.95, img_size=new_img_size)
    ring = o1-o2
    hole = get_circle(hole_ratio, img_size=new_img_size)
    wheel = x+ring-hole
    wheel = np.where(wheel>0., 1., 0.)
    return wheel
