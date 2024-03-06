import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import clip
import math
from PIL import Image
from scipy.ndimage import filters
from torch import nn
import cv2


def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    This function overlays the cam mask on the image as an heatmap.By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02 * max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1 * (1 - attn_map ** 0.7).reshape(attn_map.shape + (1,)) * img + \
               (attn_map ** 0.7).reshape(attn_map.shape + (1,)) * attn_map_c
    return attn_map


def viz_attn(img, attn_map, blur=True):
    image = Image.fromarray(np.uint8(255 * getAttMap(img, attn_map, blur)))
    # Image.fromarray(np.uint8(255 * img)).show()
    image.show()


def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize(resize)
    return np.asarray(image).astype(np.float32) / 255.

class ReshapeTransform:
    def __init__(self):
        pass

    def __call__(self, x, h, w):# x是个token序列
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        #拿到所有组成原图的token，将它们reshape回原图的大小
        x = x.permute(1, 2, 0)[..., 1:]
        x = x.reshape(x.size(0), x.size(1), h, w)
        return x

class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module, reshape_transform=None):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        self.reshape_transform = reshape_transform

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
        model: nn.Module,
        input: torch.Tensor,
        texts: torch.Tensor,
        aux_texts,
        target,
        reshape_transform,
        criterion,
        layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    # if input.grad is not None:
    #     input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:
        _, h, w = input.mask.shape
        h, w = h//model.visual.patch_size, w//model.visual.patch_size
        # Do a forward and backward pass.
        output = model(input.tensors, texts, aux_texts, input.mask)
        loss_dict, indices = criterion(output, target, aux_texts)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        losses.backward()

        grad = hook.gradient.float()
        act = hook.activation.float()

        if reshape_transform is not None:
            grad = reshape_transform(grad, h, w)
            act = reshape_transform(act, h, w)

        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.tensors.shape[2:],
        mode='bicubic',
        align_corners=False)

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return gradcam

def main():
    # @title Run

    # @markdown #### Image & Caption settings

    image_caption = 'the cat'  # @param {type:"string"}
    # @markdown ---
    # @markdown #### CLIP model settings
    # clip_model = "RN50"  # @param ["RN50", "RN101", "RN50x4", "RN50x16", 'ViT-B/16']
    # saliency_layer = "layer4"  # @param ["layer4", "layer3", "layer2", "layer1"]
    clip_model = "ViT-B/16"
    saliency_layer = "ln_1"
    # @markdown ---
    # @markdown #### Visualization settings
    blur = True  # @param {type:"boolean"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device, jit=False)

    # Download the image from the web.
    image_path = r'imgs_in/1.jpg'

    image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    image_np = load_image(image_path, model.visual.input_resolution)
    text_input = clip.tokenize([image_caption]).to(device)

    attn_map = gradCAM(
        model.visual,
        image_input,
        model.encode_text(text_input).float(),
        ReshapeTransform() if 'ViT' in clip_model else None,
        getattr(model.visual.transformer.resblocks[-3], saliency_layer) if 'ViT' in clip_model else getattr(model.visual, saliency_layer)
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()

    viz_attn(image_np, attn_map, blur)

if __name__ == '__main__':
    main()