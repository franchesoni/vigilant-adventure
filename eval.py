import os
from pathlib import Path
import numpy as np
import tqdm
from PIL import Image

from computer_vision_datasets import download, get_released_datasets, SegDataset

homepath = Path(os.path.expanduser("~"))
datapath = homepath / "storage"
dataset_name = ["ADE20K", "LVIS"][1]


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


def evaluate(ds, mask_generator, dstdir, limit=1e9):
    iou_thresholds = np.linspace(0.05, 0.95, 20)
    res_per_class_per_mask = {}
    dstdir = Path(dstdir)
    for sample_ind, sample in enumerate(ds):
        (dstdir / str(sample_ind)).mkdir(exist_ok=True, parents=True)
        if sample_ind == limit:
            break
        img, gts, class_names = sample
        Image.fromarray(img).save(dstdir / str(sample_ind) / "input_img.png")
        masks = mask_generator.generate(img)
        for gtind, gt in enumerate(gts):
            iou_with_bboxes = []
            for mask in masks:
                bbox_tl_col, bbox_tl_row, bbox_w, bbox_h = map(int, mask["bbox"])
                gt_at_bbox = gt[
                    bbox_tl_row : bbox_tl_row + bbox_h,
                    bbox_tl_col : bbox_tl_col + bbox_w,
                ]
                iou_with_bbox = gt_at_bbox.sum() / (  # intersection
                    gt.sum() - gt_at_bbox.sum() + bbox_h * bbox_w
                )  # union (pixels of gt outside + pixels of bounding box)
                iou_with_bboxes.append(iou_with_bbox)
            # now arg sort them and take the 3 best
            iou_with_bboxes = np.array(iou_with_bboxes)
            best_10 = iou_with_bboxes.argsort()[-10:]
            best_idx, best_iou = None, None
            for idx in best_10:
                mask = masks[idx]["segmentation"]
                iou = compute_iou(mask.cpu(), gt)
                if best_iou is None or iou > best_iou:
                    best_idx, best_iou = idx, iou
            best_mask = (masks[best_idx]['segmentation'].cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(gt*255).save(dstdir / str(sample_ind) / f"{gtind}_gt.png")
            Image.fromarray(best_mask).save(dstdir / str(sample_ind) / f"{gtind}_pred.png")
            Image.fromarray((img*0.5 + best_mask[...,None]*np.ones(3).reshape(1,1,3)*0.5).astype(np.uint8)).save(dstdir / str(sample_ind) / f"{gtind}_prednimg.png")
            tp = (gt*best_mask)[...,None]*np.array([1, 1, 1]).reshape(1,1,3)
            fp = (best_mask*(1-gt))[...,None]*np.array([1, 0, 0]).reshape(1,1,3)
            fn = (gt*(255-best_mask))[...,None]*np.array([0, 0, 1]).reshape(1,1,3)
            Image.fromarray(((tp + fp + fn)).astype(np.uint8)).save(dstdir / str(sample_ind) / f"{gtind}_error.png")
            class_name = class_names[gtind]
            if class_name in res_per_class_per_mask:
                res_per_class_per_mask[class_name].append(best_iou)
            else:
                res_per_class_per_mask[class_name] = [best_iou]

        mIoU = np.mean([np.mean(res) for res in res_per_class_per_mask.values()])
        res_per_mask = np.array(
            [
                res
                for class_name in res_per_class_per_mask
                for res in res_per_class_per_mask[class_name]
            ]
        ).reshape(1, -1)
        ARs_at_threshs = (res_per_mask > iou_thresholds.reshape(-1, 1)).mean(axis=1)
        mAR = np.mean(ARs_at_threshs)

        print(
            f"{sample_ind}/{min(len(ds), limit)}, mIoU: {mIoU:.3f}, AR: {mAR:.3f} @{len(masks)}",
            end="\r",
        )
    print(
        f"{sample_ind}/{min(len(ds), limit)}, mIoU: {mIoU:.3f}, AR: {mAR:.3f} @{len(masks)}"
    )


if __name__ == "__main__":
    LIMIT = 10
    METHOD = ["SAM", "MSSAM"][0]
    download(dataset_name, datapath)
    ds = SegDataset(datapath / dataset_name.lower(), "train")
    # visualize dataset
    Path('tmp').mkdir(exist_ok=True)
    for sample_ind, sample in enumerate(ds):
        if sample_ind == LIMIT:
            break
        Path(f'tmp/{sample_ind}').mkdir(exist_ok=True)
        img = sample[0]
        masks = sample[1]
        for mask_ind, mask in enumerate(masks):
            Image.fromarray(mask*255).save(f'tmp/{sample_ind}/{mask_ind}.png')
        Image.fromarray(img).save(f'tmp/{sample_ind}/img.png')

    from mask_generators import SAMPaper, MSSam
    from segment_anything import sam_model_registry

    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda:2"
    sam = sam_model_registry[model_type](sam_checkpoint)
    sam.to(device=device)
    if METHOD == "SAM":
        mask_generator = SAMPaper(sam)
        #  10/10, mIoU: 0.307, AR: 0.191 @1000
    elif METHOD == "MSSAM":
        mask_generator = MSSam(sam, box_nms_thresh=0.97, use_cpu=True, logits=True)
        # 10/10, mIoU: 0.630, AR: 0.510 @993
    evaluate(ds, mask_generator, dstdir=f'tmp/{METHOD}', limit=LIMIT)
