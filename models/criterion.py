# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Suchen for HOI detection
import torch
import torch.nn.functional as F
from torch import nn
from utils import box_ops
from utils.misc import accuracy, get_world_size, is_dist_avail_and_initialized

class CustomWeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomWeightedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, weights=None, reduction='mean'):
        # 计算交叉熵损失
        loss_list = [F.cross_entropy(i, t, weight=w, reduction='sum') for i, t, w in zip(input, target, weights)]
        loss = sum(loss_list) / len(loss_list) if reduction=='mean' else sum(loss_list)
        return loss

class MySigmoidWithOffset(nn.Module):
    def __init__(self):
        super(MySigmoidWithOffset, self).__init__()

    def forward(self, x, offset=0):
        return 1 / (1 + torch.exp(-(x - offset)))

my_loss = CustomWeightedCrossEntropyLoss()

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

    def get_target_classes_t(self, target_classes_i, num_tgts):
        target_classes_t = torch.zeros((len(target_classes_i), num_tgts), device=target_classes_i.device,
                    dtype=target_classes_i.dtype)
        target_classes_t[(torch.arange(len(target_classes_i)), target_classes_i)] = 1
        return target_classes_t

    def loss_labels(self, outputs, targets, indices, num_boxes, aux_texts, log=True):
        def _f(x):
            return 10 / (1 + torch.exp(4*x))
        def _get_visual_weights(hoi_text_similarity, label_map):
            hoi_text_similarity = hoi_text_similarity.clone().detach()
            weights = torch.where(label_map, _f(hoi_text_similarity), torch.tensor(1, dtype=hoi_text_similarity.dtype, device=hoi_text_similarity.device))
            return weights
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'logits_per_hoi' in outputs
        src_logits = outputs['logits_per_hoi']
        if aux_texts is not None:
            src_person_logits = outputs['logits_per_hoi_person']
            src_action_logits = outputs['logits_per_hoi_action']
            src_object_logits = outputs['logits_per_hoi_object']
        target_classes_i, target_classes_t, target_action_classes, target_object_classes = self._get_tgt_labels(targets, indices, src_logits.device, aux_texts['action_texts'] if aux_texts is not None else None, aux_texts['object_texts'] if aux_texts is not None else None)

        idx = self._get_src_permutation_idx(indices)
        # image-to-text alignment loss
        hoi_text_similarity = outputs['hoi_text_similarity'][idx]
        label_map = torch.zeros(hoi_text_similarity.shape, dtype=torch.bool, device=hoi_text_similarity.device)
        label_map[(torch.arange(len(hoi_text_similarity)), target_classes_i)] = True
        weights = _get_visual_weights(hoi_text_similarity, label_map)
        loss_i = my_loss(src_logits[idx], target_classes_i, weights=weights)
        if aux_texts is not None:
            target_person_classes = torch.zeros(target_object_classes.shape, device=target_object_classes.device,
                        dtype=target_object_classes.dtype)
            loss_person = F.cross_entropy(src_person_logits[idx], target_person_classes)
            loss_action = F.cross_entropy(src_action_logits[idx], target_action_classes)
            loss_object = F.cross_entropy(src_object_logits[idx], target_object_classes)
        # text-to-image alignment loss
        if self.training:
            num_tgts = target_classes_t.shape[1]
            loss_t = self.masked_out_cross_entropy(src_logits[idx][:, :num_tgts].t(), target_classes_t.t(), weights[:, :num_tgts].t())
            # if aux_texts is not None:
            #     num_tgts = torch.max(target_object_classes)+1
            #     target_person_classes_t = self.get_target_classes_t(target_person_classes, num_tgts)
            #     target_object_classes_t = self.get_target_classes_t(target_object_classes, num_tgts)
            #     loss_person_t = self.masked_out_cross_entropy(src_person_logits[idx][:, :num_tgts].t(), target_person_classes_t.t())
            #     loss_object_t = self.masked_out_cross_entropy(src_object_logits[idx][:, :num_tgts].t(),
            #                                                   target_object_classes_t.t())
            #     num_tgts = torch.max(target_action_classes) + 1
            #     target_action_classes_t = self.get_target_classes_t(target_action_classes, num_tgts)
            #     loss_action_t = self.masked_out_cross_entropy(src_action_logits[idx][:, :num_tgts].t(),
            #                                                   target_action_classes_t.t())
            #     losses = {"loss_ce": (loss_i + loss_t) / 2, 'loss_aux': ((loss_person+loss_person_t)/2 + (loss_action+loss_action_t)/2 + (loss_object+loss_object_t)/2) / 3}
            # else:
            losses = {"loss_ce": (loss_i + loss_t) / 2, 'loss_aux': (loss_person + loss_action + loss_object) / 3}
        else:
            losses = {'loss_ce': loss_i}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_i)[0]
        return losses

    def masked_out_cross_entropy(self, src_logits, target_classes, weights):
        loss = 0
        num_pos = target_classes.sum(dim=-1)
        # If there is only one active positive label, then this will be ordinary cross entropy
        indices = torch.nonzero(num_pos < 2, as_tuple=True)[0]
        targets_one_pos = torch.argmax(target_classes[indices], dim=-1)
        # loss += F.cross_entropy(src_logits[indices], targets_one_pos, reduction="sum")
        loss += my_loss(src_logits[indices], targets_one_pos, weights=weights[indices], reduction='sum')

        # If there are multiple positive labels, then we compute them one by one. Each time,
        # the other positive labels are masked out.
        indices = torch.nonzero(num_pos > 1, as_tuple=True)[0]
        for i in indices:
            t = target_classes[i]
            cnt = sum(t)
            loss_t = 0
            for j in torch.nonzero(t):
                mask = (t == 0)
                mask[j] = True
                tgt = t[mask].argmax(dim=-1, keepdim=True)
                # loss_t += F.cross_entropy(src_logits[i:i+1, mask], tgt, reduction="sum")
                loss_t += my_loss(src_logits[i:i+1, mask], tgt, weights=weights[i:i+1, mask], reduction="sum")
            loss += (loss_t / cnt)
        loss = loss / len(src_logits)
        return loss

    def loss_confidences(self, outputs, targets, indices, num_boxes, aux_texts, log=True):
        """ Bounding box confidence score for the interaction prediction. """
        assert 'box_scores' in outputs
        box_scores = outputs['box_scores'].sigmoid()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([torch.ones(len(J), dtype=torch.int64, device=box_scores.device) for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(box_scores.shape[:2], 0, dtype=torch.int64, device=box_scores.device)
        target_classes[idx] = target_classes_o
        target_classes = target_classes.to(box_scores.dtype)

        weight = torch.ones_like(target_classes) * self.eos_coef
        weight[idx] = 1.
        loss_conf = F.binary_cross_entropy(box_scores.flatten(), target_classes.flatten(), weight=weight.flatten())
        losses = {'loss_conf': loss_conf}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, aux_texts, log=False):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = []
        for t, (_, indices_per_t) in zip(targets, indices):
            for i in indices_per_t:
                person_id = t["hois"][i]["subject_id"]
                object_id = t["hois"][i]["object_id"]
                target_boxes.append(torch.cat([t["boxes"][person_id], t["boxes"][object_id]]))
        target_boxes = torch.stack(target_boxes, dim=0)

        loss_pbbox = F.l1_loss(src_boxes[:, :4], target_boxes[:, :4], reduction='none')
        loss_obbox = F.l1_loss(src_boxes[:, 4:], target_boxes[:, 4:], reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_pbbox.sum() / num_boxes + loss_obbox.sum() / num_boxes

        loss_pgiou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[:, :4]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[:, :4])))
        loss_ogiou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[:, 4:]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[:, 4:])))

        losses['loss_giou'] = loss_pgiou.sum() / num_boxes + loss_ogiou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_tgt_labels(self, targets, indices, device, action_texts, object_texts):
        if self.training:
            unique_hois, cnt = {}, 0 # Get unique hoi ids in the mini-batch
            for t in targets:
                for hoi in t["hois"]:
                    hoi_id = hoi["hoi_id"]
                    if hoi_id not in unique_hois:
                        unique_hois[hoi_id] = cnt
                        cnt += 1
            target_classes_i = []
            target_action_classes = []
            target_object_classes = []
            for t, (_, indices_per_t) in zip(targets, indices):
                for i in indices_per_t:
                    hoi_id = t["hois"][i]["hoi_id"]
                    target_classes_i.append(unique_hois[hoi_id])
                    target_action_classes.append(action_texts.index(t["hois"][i]['text'][0].replace("_", " ")))
                    target_object_classes.append(object_texts.index(t["hois"][i]['text'][-1].replace("_", " ")))

            num_fgs = len(torch.cat([src for (src, _) in indices]))
            target_classes_t = torch.zeros((num_fgs, cnt), dtype=torch.int64)
            for i, cls_id in zip(range(len(target_classes_i)), target_classes_i):
                target_classes_t[i, cls_id] = 1
            target_classes_t = target_classes_t.to(device)
            target_object_classes = torch.as_tensor(target_object_classes).to(device)
            target_action_classes = torch.as_tensor(target_action_classes).to(device)
        else:
            target_classes_i = []
            for t, (_, indices_per_t) in zip(targets, indices):
                for i in indices_per_t:
                    target_classes_i.append(t["hois"][int(i)]["hoi_id"])
            target_classes_t = None # Skip the calculation of text-to-image alignment at inference
            target_action_classes = None
            target_object_classes = None
        target_classes_i = torch.as_tensor(target_classes_i).to(device)
        return target_classes_i, target_classes_t, target_action_classes, target_object_classes

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            "confidences": self.loss_confidences,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, aux_texts=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets, aux_texts)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["hois"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, aux_texts=aux_texts))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if self.training:
                    aux_outputs.update({
                        'logits_per_hoi': outputs['logits_per_hoi'],
                        'logits_per_hoi_person':outputs['logits_per_hoi_person'],
                        'logits_per_hoi_action': outputs['logits_per_hoi_action'],
                        'logits_per_hoi_object':outputs['logits_per_hoi_object']
                    })
                else:
                    aux_outputs.update({
                        'logits_per_hoi': outputs['logits_per_hoi']
                    })
                indices = self.matcher(aux_outputs, targets, aux_texts)
                for loss in ['boxes', 'confidences']:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, aux_texts=aux_texts)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, indices