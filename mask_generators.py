import numpy as np
from pathlib import Path
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
    ):
        self.predictor = SamPredictor(sam)
        self.points_per_batch = points_per_batch
        self.point_grids = build_all_layer_point_grids(points_per_side, 0, 1)
        self.scales = scales
        self.box_nms_thresh = box_nms_thresh

    def generate(self, img, max_proposals=1000):
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
            stability_scores.append(stability_score)
            maskss.append(
                self.predictor.model.mask_threshold
                < masks.reshape(-1, *masks.shape[2:])
            )
            iou_predss.append(iou_preds.reshape(-1))
        del masks, iou_preds
        maskss, iou_predss, stability_scores = (
            torch.cat(maskss),
            torch.cat(iou_predss),
            torch.cat(stability_scores),
        )
        masks, iou_preds = maskss, iou_predss
        torch.cuda.empty_cache()

        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            iou_preds,
            torch.zeros_like(boxes[:, 0]),
            iou_threshold=self.box_nms_thresh,
        )

        masks = masks[keep_by_nms]
        iou_preds = iou_preds[keep_by_nms]
        boxes = boxes[keep_by_nms]

        stability_score = calculate_stability_score(
            masks, self.predictor.model.mask_threshold, 1.0
        )
        masks = uncrop_masks(masks, crop_box, im_size[0], im_size[1])
        boxes = uncrop_boxes_xyxy(boxes, crop_box)

        if max_proposals < len(masks):
            scores = iou_preds + stability_score
            # sort logits according to scores
            sorted_indices = scores.argsort(descending=True)
            masks = masks[sorted_indices]
            # keep only the top mask_proposals
            masks = masks[:max_proposals]

        return [
            {"segmentation": mask, "bbox": box_xyxy_to_xywh(box).tolist()}
            for mask, box in zip(masks, boxes)
        ]


if __name__ == "__main__":
    # test the generator on cat.png
    import cv2

    imgpath = Path("snavona.jpg")
    # img = cv2.imread('cat.png')
    img = cv2.imread(str(imgpath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scales = [0, 1, 2]

    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda:1"
    sam = sam_model_registry[model_type](sam_checkpoint)
    sam.to(device=device)
    mask_generator = SAMPaper(sam, points_per_batch=32, scales=scales)
    masks = mask_generator.generate(img)
    H, W = masks[0]["segmentation"].shape
    C = len(masks)
    feats = (
        torch.stack([m["segmentation"] for m in masks], dim=0).reshape(C, H * W).T
    )  # HW, C

    # we will PCA them to 3d and show the image
    from sklearn.decomposition import PCA

    n_components = 120
    pca = PCA(n_components=n_components)
    pcafeats = pca.fit_transform(feats.cpu().numpy())
    pcafeats = pcafeats.reshape(H, W, n_components)
    np.save(
        str(imgpath.parent / (imgpath.stem + f"_logitspca_s{scales}.npy")),
        pcafeats.astype(np.float32).transpose(2, 0, 1),
    )

    # minmaxnorm = lambda x: (x - x.min()) / (x.max() - x.min())
    # def pct_norm(x, pct=1):
    #     toppct, botpct = np.percentile(x, 100-pct), np.percentile(x, pct)
    #     return (np.clip(x, botpct, toppct) - botpct)  / (toppct - botpct)
    # import matplotlib.pyplot as plt
    # plt.imsave(str(imgpath.parent / (imgpath.stem + f'logitspca_s{scales}.png')), pct_norm(pcafeats))
    # plt.axis('off')
