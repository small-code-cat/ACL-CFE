# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/9/4 18:16 
# @Author : wzy 
# @File : model_test.py
# ---------------------------------------
import numpy as np
from PIL import Image
import torch
import argparse
from arguments import get_args_parser
from matplotlib import pyplot as plt
from datasets import build_dataset
from torch.utils.data import DataLoader
# from Grad_CAM import GradCAM, show_cam_on_image
import torchvision
from models import build_model
import utils.misc as utils
from grad_cam.clip_gramcam import gradCAM, ReshapeTransform, viz_attn, load_image
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


def prepare_text_inputs(targets, device):
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    texts = [i['text'] for i in targets[0]['hois']]
    text_inputs = ['person']
    action_text_inputs = []

    text_tokens = []
    for action_text, object_text in texts:
        action_text = action_text.replace("_", " ")
        object_text = object_text.replace("_", " ")
        action_token = _tokenizer.encode(action_text)
        object_token = _tokenizer.encode(object_text)

        action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
        object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
        text_tokens.append([action_token, object_token])
        if action_text not in action_text_inputs:
            action_text_inputs.append(action_text)
        if object_text not in text_inputs:
            text_inputs.append(object_text)
    object_token = [torch.as_tensor([sot_token] + _tokenizer.encode(i) + [eot_token], dtype=torch.long).to(device) for i
                    in text_inputs]
    object_index = torch.tensor(
        [text_inputs.index(i) for i in [hoi['text'][-1].replace("_", " ") for t in targets for hoi in t['hois']]],
        device=device)
    action_token = [torch.as_tensor([sot_token] + _tokenizer.encode(i) + [eot_token], dtype=torch.long).to(device)
                    for i in action_text_inputs]
    action_index = torch.tensor(
        [action_text_inputs.index(i) for i in [hoi['text'][0].replace("_", " ") for t in targets for hoi in t['hois']]],
        device=device)
    aux_texts = {
        'object_token': object_token,
        'object_index': object_index,
        'object_texts': text_inputs,
        'action_token': action_token,
        'action_index': action_index,
        'action_texts': action_text_inputs
    }
    return text_tokens, aux_texts

def main(args):
    # model = torchvision.models.resnet50(pretrained=True)
    device = torch.device('cpu')
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # cam = GradCAM(model=model, target_layers=layers, use_cuda=False
    for images, targets in data_loader_val:
        if targets[0]['image_id'].item() == 4811:
            # 不能用forward，
            images = images.to(device)
            print(targets[0])
            texts, aux_texts = prepare_text_inputs(targets, device)
            attn_map = gradCAM(
                model,
                images,
                texts,
                aux_texts,
                targets,
                ReshapeTransform(),
                criterion,
                getattr(model.visual.transformer.resblocks[-1], 'ln_1')
                # getattr(model.visual.transformer.resblocks[-1], 'ln_1')
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()
            _, h, w = images.mask.shape
            viz_attn(load_image(targets[0]['filename'], (w,h)), attn_map, True)
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
