from torchvision import transforms as T
import torch.nn.functional as F
from PIL import ImageOps
import PIL
import random


def get_clip_transforms(
    image_size=None
):
    assert image_size is not None
    return clip_preprocess(image_size)


def pad_img(desired_size):
    def fn(im):
        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        im = im.resize(new_size, PIL.Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = PIL.Image.new("RGB", (desired_size, desired_size))
        new_im.paste(
            im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
        )

        return new_im

    return fn


def crop_or_pad(n_px, pad=False):
    if pad:
        return pad_img(n_px)
    else:
        return T.CenterCrop(n_px)


def clip_preprocess(n_px, use_pad=False):
    return T.Compose(
        [
            T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
            crop_or_pad(n_px, pad=use_pad),
            lambda image: image.convert("RGB"),
            T.ToTensor(),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
