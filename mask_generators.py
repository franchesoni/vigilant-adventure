import numpy as np
import skimage
import os
import cv2
from pathlib import Path
from sklearn.decomposition import PCA
from torchvision.ops.boxes import batched_nms
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.amg import (
    build_all_layer_point_grids,
    batch_iterator,
    batched_mask_to_box,
    uncrop_boxes_xyxy,
    calculate_stability_score,
    uncrop_masks,
    box_xyxy_to_xywh,
)
import torch


class SAMPaper:
    def __init__(
        self,
        sam,
        points_per_batch=64,
        points_per_side=64,
        scales=[0, 1, 2],
        box_nms_thresh=0.9,
        use_cpu=False,
        logits=False,
        max_proposals=1000,
    ):
        self.predictor = SamPredictor(sam)
        self.points_per_batch = points_per_batch
        self.point_grids = build_all_layer_point_grids(points_per_side, 0, 1)
        self.scales = scales
        self.box_nms_thresh = box_nms_thresh
        self.use_cpu = use_cpu
        self.logits = logits
        self.max_proposals = max_proposals

    def generate(self, img):
        im_size = img.shape[:2]
        crop_box = (0, 0, im_size[1], im_size[0])
        self.predictor.set_image(img)

        points_scale = np.array(im_size)[None, ::-1]
        points_for_image = self.point_grids[0] * points_scale

        maskss, iou_predss, stability_scores = [], [], []
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            transformed_points = self.predictor.transform.apply_coords(
                points, self.predictor.original_size
            )
            in_points = torch.as_tensor(
                transformed_points, device=self.predictor.device
            )
            in_labels = torch.ones(
                len(points), device=self.predictor.device, dtype=torch.int
            )
            masks, iou_preds, _ = self.predictor.predict_torch(
                in_points[:, None, :],
                in_labels[:, None],
                multimask_output=True,
                return_logits=True,
            )
            stability_score = calculate_stability_score(
                masks, self.predictor.model.mask_threshold, 1.0
            )
            masks, iou_preds, stability_score = (
                masks[:, self.scales],
                iou_preds[:, self.scales],
                stability_score[:, self.scales],
            )
            if self.use_cpu:
                masks, iou_preds, stability_score = (
                    masks.cpu(),
                    iou_preds.cpu(),
                    stability_score.cpu(),
                )
            stability_scores.append(stability_score.reshape(-1))
            masks = (
                masks if self.logits else self.predictor.model.mask_threshold < masks
            )
            maskss.append(masks.reshape(-1, *masks.shape[2:]))
            iou_predss.append(iou_preds.reshape(-1))
            print(".")
        print("generated! processing...")
        del masks, iou_preds
        maskss, iou_predss, stability_scores = (
            torch.cat(maskss),
            torch.cat(iou_predss),
            torch.cat(stability_scores),
        )
        masks, iou_preds = maskss, iou_predss
        torch.cuda.empty_cache()

        boxes = batched_mask_to_box(
            self.predictor.model.mask_threshold < masks if self.logits else masks
        )
        keep_by_nms = batched_nms(
            boxes.float(),
            iou_preds,
            torch.zeros_like(boxes[:, 0]),
            iou_threshold=self.box_nms_thresh,
        )

        masks = masks[keep_by_nms]
        iou_preds = iou_preds[keep_by_nms]
        stability_scores = stability_scores[keep_by_nms]
        boxes = boxes[keep_by_nms]

        masks = uncrop_masks(masks, crop_box, im_size[0], im_size[1])
        boxes = uncrop_boxes_xyxy(boxes, crop_box)

        if self.max_proposals < len(masks):
            scores = iou_preds + stability_scores
            # sort logits according to scores
            sorted_indices = scores.argsort(descending=True)
            masks = masks[sorted_indices]
            # keep only the top mask_proposals
            masks = masks[: self.max_proposals]

        print("processed!")
        return [
            {"segmentation": mask, "bbox": box_xyxy_to_xywh(box).tolist()}
            for mask, box in zip(masks, boxes)
        ]


def rgb_to_int(image):
    # Assumes image is numpy array of shape (H, W, 3) with dtype uint8
    int_image = (
        image[:, :, 0].astype(np.int32) * 256 * 256
        + image[:, :, 1].astype(np.int32) * 256
        + image[:, :, 2].astype(np.int32)
    )
    return int_image


class MSSam(SAMPaper):
    def __init__(self, *args, pca_dim=256, pamss_path="pamss/build/pamss", **kwargs):
        assert (
            kwargs["logits"] and kwargs["box_nms_thresh"] > 0.9
        ), "MSSam requires logits and box_nms_thresh > 0.9"
        super().__init__(*args, **kwargs)
        self.pca_dim = pca_dim
        self.pamss_path = pamss_path

    def generate(self, img, order=0):
        # get logits from the proposals
        logits = torch.stack([m["segmentation"] for m in super().generate(img)], dim=0)
        print("running pca...")
        C, H, W = logits.shape
        logits = logits.reshape(C, H * W).T
        pca = PCA(n_components=self.pca_dim)
        pcafeats = pca.fit_transform(logits.cpu().numpy())
        pcafeats = pcafeats.reshape(H, W, self.pca_dim)
        pcafeats = pcafeats.transpose(2, 0, 1).astype(
            np.float32
        )  # convert to format expected by MS
        print("running Mumford-Shah...")
        np.save("tmp.npy", pcafeats)  # temporary save, file-based communication
        # now we call pamss. Signature is ./pamsspath <input.npy> <output.png> <labels.png>, we call it for different levels, in a geometric way
        n_masks = [
            self.max_proposals // (2**i)
            for i in range(1, int(np.floor(np.log2(self.max_proposals))))
        ]
        for n in n_masks:
            call_command = (
                f"./{self.pamss_path} -o {order} -n {n} tmp.npy tmp.png labels_{n}.png"
            )
            print(f"running: `{call_command}`")
            os.system(call_command)
            print(f"done with `{call_command}!")
        masks = []
        for n in n_masks[::-1]:
            rgb_labels = cv2.imread(f"labels_{n}.png")
            labels = skimage.measure.label(rgb_to_int(rgb_labels))
            assert (
                labels.max() + 1 <= n
            ), f"expected #labels<{n}, got {labels.max()+1} labels"
            for i in range(n):
                mask = labels == i
                masks.append(mask)
        masks = torch.from_numpy(np.array(masks))
        boxes = batched_mask_to_box(masks)
        return [
            {"segmentation": mask, "bbox": box_xyxy_to_xywh(box).tolist()}
            for mask, box in zip(masks, boxes)
        ]


def tryit(
    imgpath,
    scales=[0, 1, 2],
    points_per_batch=256,
    use_cpu=False,
    box_nms_threshold=0.9,
    logits=False,
):
    # test the generator on cat.png
    imgpath = Path(imgpath)
    # img = cv2.imread('cat.png')
    img = cv2.imread(str(imgpath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda:1"
    sam = sam_model_registry[model_type](sam_checkpoint)
    sam.to(device=device)
    # mask_generator = SAMPaper(
    #     sam, points_per_batch=points_per_batch, scales=scales, box_nms_thresh=nms, use_cpu=use_cpu, logits=logits
    # )
    mask_generator = MSSam(
        sam,
        points_per_batch=points_per_batch,
        scales=scales,
        box_nms_thresh=box_nms_threshold,
        use_cpu=use_cpu,
        logits=logits,
    )
    masks = mask_generator.generate(img)


def run_pca(imgpath, masks, shape, scales, n_components):
    H, W, C = shape
    # we will PCA them to 3d and show the image
    feats = (
        torch.stack([m["segmentation"] for m in masks], dim=0).reshape(C, H * W).T
    )  # HW, C
    pca = PCA(n_components=n_components)
    pcafeats = pca.fit_transform(feats.cpu().numpy())
    pcafeats = pcafeats.reshape(H, W, n_components)
    np.save(
        str(imgpath.parent / (imgpath.stem + f"_logitspca_s{scales}.npy")),
        pcafeats.astype(np.float32).transpose(2, 0, 1),
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(tryit)

    # minmaxnorm = lambda x: (x - x.min()) / (x.max() - x.min())
    # def pct_norm(x, pct=1):
    #     toppct, botpct = np.percentile(x, 100-pct), np.percentile(x, pct)
    #     return (np.clip(x, botpct, toppct) - botpct)  / (toppct - botpct)
    # import matplotlib.pyplot as plt
    # plt.imsave(str(imgpath.parent / (imgpath.stem + f'logitspca_s{scales}.png')), pct_norm(pcafeats))
    # plt.axis('off')
