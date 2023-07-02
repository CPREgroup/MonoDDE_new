from mindspore.dataset.vision import Normalize


class Normalization:
    def __init__(self, cfg):
        self.std = cfg.INPUT.PIXEL_STD
        self.mean = cfg.INPUT.PIXEL_MEAN
        self.to_bgr =cfg.INPUT.TO_BGR
        self.device=cfg.MODEL.DEVICE
        self.normal=Normalize(mean=self.mean, std=self.std)


    def __call__(self, image):
        if self.device=='Ascend' or 'GPU':
            image = self.normal(image)
        else:
            print('Do not use Normalize in cpu environmental.')
        if self.to_bgr:
            image = image[[2, 1, 0]]

        return image