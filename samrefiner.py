from PIL import Image
import numpy as np
import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
n_points = 200
n_steps = 100
n_epochs = 10
n_feats = 16
seed = 0
outfile = 'feats.npy'

if __name__ == "__main__":
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("loading sam...")
    sam = sam_model_registry[model_type](sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    sam_predictor = SamPredictor(sam)

    print("processing img...")
    image = np.array(Image.open("cat.png").convert("RGB"))
    H, W = image.shape[:2]
    sam_predictor.set_image(image)

    print("initializing features...")
    # initialize as normal distribution
    feats = torch.randn(n_feats, H, W, requires_grad=True, device=device)

    # feats is F, H, W
    # points is B, 2, the format is row, col

    for epoch in range(n_epochs):
        optim = torch.optim.Adam([feats], lr=0.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=1, total_steps=n_steps, pct_start=0.05
        )
        print("getting points...")
        points = (np.random.rand(n_points, 2) * np.array([H, W])[None]).astype(int)
        with torch.no_grad():
            coords_torch = sam_predictor.transform.apply_coords(
                points[:, ::-1], sam_predictor.original_size
            )  # n_points, 2
            coords_torch = torch.as_tensor(
                coords_torch, dtype=torch.float, device=device
            )  # n_points, 2, the format is col, row
            labels_torch = torch.ones(n_points).int().to(device)
            coords_torch, labels_torch = (
                coords_torch.view(n_points, 1, 2),
                labels_torch.view(n_points, 1),
            )  # (n_points, 1, 2), (n_points, 1)  add batch dim
            print("predicting masks...")
            masks, iou_predictions, low_res_masks = sam_predictor.predict_torch(
                coords_torch, labels_torch, multimask_output=True, return_logits=True
            )
            masks = torch.sigmoid(masks)  # convert into probabilities

        print("optimizing...")
        # get the features at the points
        f_indices = torch.arange(n_feats).view(n_feats, 1, 1)
        rows = points[:, 0]
        cols = points[:, 1]

        for step in range(n_steps):
            point_feats = feats[f_indices, rows, cols].view(n_feats, n_points).T  # B, F
            # compute the distance maps
            distance_maps = torch.norm(
                feats.view(1, n_feats, H, W)
                - point_feats.view(n_points, n_feats, 1, 1),
                dim=1,
                p=2,
                keepdim=True,
            )  # B, 1, H, W
            distance_maps = distance_maps**2 / 2  # assume sigma=1
            probs = torch.exp(-distance_maps)
            mse_loss = torch.nn.functional.mse_loss(probs, masks[:, :1], reduce="mean")
            grad_reg_loss = torch.norm(feats[:, 1:] - feats[:, :-1], dim=0).mean() + torch.norm(feats[:, :, 1:] - feats[:,:, :-1], dim=0).mean()
            loss = mse_loss + 0.1 * grad_reg_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            print(f"epoch {epoch} step {step}, loss: {loss.item()}", end="\r")

    print('saving...')
    np.save(outfile, feats.detach().cpu().numpy())
    print("done!")
