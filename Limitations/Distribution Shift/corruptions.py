import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from scipy.ndimage import zoom as scizoom
from scipy.ndimage import gaussian_filter
import skimage as sk
from skimage.filters import gaussian
import cv2
import warnings


warnings.simplefilter("ignore", UserWarning)


class Corruptions:
    """
    Factory for common image corruptions used in robustness benchmarks.
    """

    # Noise

    @staticmethod
    def gaussian_noise(x, severity=1):
        c =[0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        x = x / 255.0
        return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    @staticmethod
    def shot_noise(x, severity=1):
        c = [60, 25, 12, 5, 3][severity -1]
        x= x/ 255.0
        return np.clip(np.random.poisson(x *c) / float(c), 0, 1) * 255

    @staticmethod
    def impulse_noise(x, severity=1):
        c = [0.03, 0.06,0.09, 0.17, 0.27][severity - 1]
        x= sk.util.random_noise(x /255.0, mode='s&p', amount=c)
        return np.clip(x, 0, 1) * 255

    # Blur

    @staticmethod
    def defocus_blur(x, severity=1):
        c = [1,2, 3,4, 6][severity - 1]
        x= x/ 255.0
        return np.clip(gaussian(x, sigma=c, channel_axis=-1), 0, 1) * 255

    @staticmethod
    def glass_blur(x, severity=1):
        # Each tuple: (blur sigma, max pixel displacement, iterations)
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        sigma, k, iterations = c
        x = x / 255.0
        h, w =x.shape[:2]

        for _ in range(iterations):
            dx = gaussian_filter(np.random.randn(h, w), sigma, mode='constant') * k
            dy= gaussian_filter(np.random.randn(h, w), sigma, mode='constant') * k
            x_idx = np.clip(np.arange(w)[None, :] + dx, 0, w - 1).astype(int)
            y_idx =np.clip(np.arange(h)[:, None] + dy, 0, h - 1).astype(int)
            x= x[y_idx, x_idx]

        return np.clip(gaussian(x, sigma=sigma, channel_axis=-1), 0, 1) * 255

    @staticmethod
    def motion_blur(x, severity=1):
        c =[(10, 3), (15, 5), (15, 8),(15, 12), (20,15)][severity - 1]
        kernel_size, angle = c
        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        kernel = np.diag(np.ones(kernel_size))
        kernel= cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        kernel =kernel / kernel.sum()
        return cv2.filter2D(x, -1, kernel)

    @staticmethod
    def zoom_blur(x, severity=1):
        c =[np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][severity - 1]

        x= (x / 255.0).astype(np.float32)
        out = np.zeros_like(x)
        h, w =x.shape[:2]

        for zoom_factor in c:
            new_h, new_w = int(h *zoom_factor), int(w* zoom_factor)
            zoomed = cv2.resize(x, (new_w, new_h))
            sh = (new_h -h) // 2
            sw = (new_w- w) // 2
            out+= zoomed[sh:sh+h, sw:sw+w]

        return np.clip(out /len(c), 0, 1) * 255

    # Weather

    @staticmethod
    def plasma_fractal(mapsize=32, wibbledecay=3):
        """
        Diamond-square fractal heightmap, used to generate fog texture.
        """
        assert (mapsize & (mapsize - 1) ==0)
        maparray= np.empty((mapsize, mapsize), dtype=np.float32)
        maparray[0, 0]= 0
        stepsize = mapsize
        wibble = 100

        def wibbledmean(array):
            return array / 4 +wibble * np.random.uniform(-wibble, wibble, array.shape)

        def fillsquares():
            
            cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            squareaccum= cornerref + np.roll(cornerref, shift=-1, axis=0)
            squareaccum += np.roll(squareaccum, shift=-1, axis=1)
            maparray[stepsize// 2:mapsize:stepsize,
                     stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

        def filldiamonds():
            mapsize = maparray.shape[0]
            drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
            ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            
            ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
            lulsum =ulgrid + np.roll(ulgrid, -1, axis=1)
            
            ltsum = ldrsum + lulsum
            maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] =wibbledmean(ltsum)
            
            tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
            tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
            ttsum = tdrsum +tulsum
            maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize]= wibbledmean(ttsum)

        while stepsize >= 2:
            fillsquares()
            filldiamonds()
            stepsize //=2
            wibble/= wibbledecay

        maparray -= maparray.min()
        return maparray /maparray.max()

    @staticmethod
    def fog(x, severity=1):
        # Severity parameters tuned for CIFAR-10-C: (fog density, wibbledecay)
        c = [(.2, 3),(.5, 3), (0.75, 2.5),(1, 2), (1.5, 1.75)][severity - 1]

        x = x /255.0
        max_val= x.max()

        map_size =32
        if x.shape[0] >map_size:
            map_size =int(2**np.ceil(np.log2(x.shape[0])))

        fractal= Corruptions.plasma_fractal(mapsize=map_size, wibbledecay=c[1])
        fractal =fractal[:x.shape[0], :x.shape[1]][..., None]

        x += c[0] * fractal
        return np.clip(x * max_val /(max_val + c[0]), 0, 1) * 255

    @staticmethod
    def brightness(x, severity=1):
        c = [0.1,0.2, 0.3,0.4, 0.5][severity - 1]
        x = x /255.0
        hsv= sk.color.rgb2hsv(x)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + c, 0, 1)
        return sk.color.hsv2rgb(hsv)* 255

    @staticmethod
    def contrast(x, severity=1):
        c = [0.4, 0.3, 0.2, 0.1,0.05][severity - 1]
        x = x/ 255.0
        means =np.mean(x, axis=(0, 1), keepdims=True)
        return np.clip((x - means) * c +means, 0, 1) * 255

    # Digital
    @staticmethod
    def elastic_transform(x, severity=1):
        c = [(244 * 2, 244 * 0.7, 244 * 0.1),
             (244 * 2, 244 * 0.08, 244 * 0.2),
             (244 * 0.05, 244 * 0.01, 244 * 0.02),
             (244 * 0.07, 244 * 0.01, 244 * 0.02),
             (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]
        alpha = [0.5, 1.0, 1.5, 2.0, 2.5][severity - 1] * 3
        sigma = [2.0, 2.5, 3.0, 3.5, 4.0][severity - 1]

        shape = x.shape
        dx =gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), sigma)* alpha
        dy =gaussian_filter((np.random.rand(*shape[:2])* 2- 1), sigma) * alpha

        x_grid, y_grid = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x =np.float32(x_grid + dx)
        map_y = np.float32(y_grid + dy)

        return cv2.remap(x, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    @staticmethod
    def pixelate(x, severity=1):
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
        h, w= x.shape[:2]
        # Shrink then blow back up — nearest neighbor gives the blocky look
        small =cv2.resize(x, (int(w * c), int(h * c)), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def jpeg_compression(x, severity=1):
        c = [25,18, 15, 10, 7][severity - 1]
        _, enc =cv2.imencode('.jpg', x, [int(cv2.IMWRITE_JPEG_QUALITY), c])
        return cv2.imdecode(enc, 1)


CORRUPTION_DICT = {
    'gaussian_noise': Corruptions.gaussian_noise,
    'shot_noise': Corruptions.shot_noise,
    'impulse_noise': Corruptions.impulse_noise,
    'defocus_blur': Corruptions.defocus_blur,
    'glass_blur': Corruptions.glass_blur,
    'motion_blur': Corruptions.motion_blur,
    'zoom_blur': Corruptions.zoom_blur,
    'fog': Corruptions.fog,
    'brightness': Corruptions.brightness,
    'contrast': Corruptions.contrast,
    'elastic_transform': Corruptions.elastic_transform,
    'pixelate': Corruptions.pixelate,
    'jpeg_compression': Corruptions.jpeg_compression
}


class CorruptedDataset(data.Dataset):
    """
    Wraps any PyTorch dataset and applies a corruption on-the-fly.
    """
    def __init__(self, base_dataset, corruption_name, severity=1):
        self.base_dataset = base_dataset
        self.corruption_name = corruption_name
        self.severity = severity

        if corruption_name not in CORRUPTION_DICT:
            raise ValueError(f"Unknown corruption: {corruption_name}. Available: {list(CORRUPTION_DICT.keys())}")

        self.corruption_fn = CORRUPTION_DICT[corruption_name]

    def __getitem__(self, index):
        img, target = self.base_dataset[index]

        # Normalize input to numpy (H, W, C) float32 in [0, 255]
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy() * 255.0
        elif isinstance(img, Image.Image):
            img_np =np.array(img).astype(np.float32)
        else:
            img_np =np.array(img).astype(np.float32)

        img_corrupted= self.corruption_fn(img_np, self.severity)
        img_corrupted =np.clip(img_corrupted, 0, 255).astype(np.uint8)
        img_pil =Image.fromarray(img_corrupted)

        if hasattr(self.base_dataset, 'transform') and self.base_dataset.transform is not None:
            img_pil = self.base_dataset.transform(img_pil)

        return img_pil, target

    def __len__(self):
        return len(self.base_dataset)


if __name__ == '__main__':
    from torchvision.datasets import CIFAR100
    from torchvision.transforms import  ToTensor

    base_dataset = CIFAR100(root='data', train=True, transform=None, download=True)
    corrupted_dataset = CorruptedDataset(base_dataset, corruption_name='gaussian_noise', severity=3)

    print("Testing data loading...")
    for i in range(5):
        img, label = corrupted_dataset[i]
        print(f"Index{i}: Label: {label}, Type: {type(img)}, Size: {img.size}")

    print("Success!")
