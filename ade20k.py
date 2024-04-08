
import os
from pathlib import Path
import numpy as np
import tqdm

from computer_vision_datasets import download, get_released_datasets, SegDataset

homepath = Path(os.path.expanduser('~'))
datapath = homepath / 'data'

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

if __name__ == '__main__':
    download('ADE20K', datapath)
    ds = SegDataset(datapath / 'ade20k', 'test')

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    sam_checkpoint = 'sam_vit_b_01ec64.pth'
    model_type = 'vit_b'
    device = 'cuda'
    sam = sam_model_registry[model_type](sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, box_nms_thresh=0.5, stability_score_thresh=0.9, pred_iou_thresh=0.7)

    res_per_class_per_mask = {}
    for sample_ind, sample in enumerate(ds):
        img, gts, class_names = sample
        masks = mask_generator.generate(img)
        for gtind, gt in enumerate(gts):
            iou_with_bboxes = []
            for mask in masks:
                bbox_tl_col, bbox_tl_row, bbox_w, bbox_h = map(int, mask['bbox'])
                try:
                    gt_at_bbox = gt[bbox_tl_row:bbox_tl_row+bbox_h, bbox_tl_col:bbox_tl_col+bbox_w]
                except TypeError:
                    breakpoint()
                iou_with_bbox = (gt_at_bbox.sum() # intersection
                / (gt.sum() - gt_at_bbox.sum() + bbox_h * bbox_w)  # union (pixels of gt outside + pixels of bounding box)
                )
                iou_with_bboxes.append(iou_with_bbox)
            # now arg sort them and take the 3 best
            iou_with_bboxes = np.array(iou_with_bboxes)
            best_10 = iou_with_bboxes.argsort()[-10:]
            best_idx, best_iou = None, None
            for idx in best_10:
                mask = masks[idx]['segmentation']
                iou = compute_iou(mask, gt)
                if best_iou is None or iou > best_iou:
                    best_idx, best_iou = idx, iou
            class_name = class_names[gtind]
            if class_name in res_per_class_per_mask:
                res_per_class_per_mask[class_name].append(best_iou)
            else:
                res_per_class_per_mask[class_name] = [best_iou]

        print(f'{sample_ind}/{len(ds)}, mIoU: {np.mean([np.mean(res) for res in res_per_class_per_mask.values()]):.3f}', end='\r')













