from pathlib import Path
import shutil
from typing import Any, Literal
from diffusers import AutoPipelineForInpainting
import numpy as np
import torch
import json
import pandas as pd
from PIL import Image, ImageDraw
from diffusers import StableDiffusionGLIGENPipeline
import argparse


def process_dataset(dataset_path: Path):
    """
    Expects the dataset folder to be in the format:

        dataset/
            - loco-all-v1.json
            - subset-1/
            ...

    """
    j = json.load((dataset_path / "loco-all-v1.json").open("rb"))
    df_i = pd.DataFrame(j["images"])
    df_a = pd.DataFrame(j["annotations"])
    annotations_df = df_a.join(df_i.set_index("id"), on="image_id")
    annotations_df["path"] = annotations_df["path"].apply(lambda x: Path(*Path(x).parts[2:]))
    # category 5 = "forklift"
    fork_lift_idx = annotations_df[annotations_df["category_id"] == 5].index.values
    np.random.shuffle(fork_lift_idx)
    save_dir = dataset_path / "forklifts"
    save_dir.mkdir(exist_ok=True)
    forklift_df = annotations_df.loc[fork_lift_idx[:50]]
    forklift_df.to_csv(save_dir / "annotations.csv")
    forklift_df["path"].apply(lambda x: shutil.copy(dataset_path.joinpath(x), save_dir))


def visualize_image(forklifts_path, annot: dict[str, Any]):
    img = Image.open(forklifts_path.joinpath(annot["file_name"]))
    x, y, w, h = eval(annot["bbox"])
    draw = ImageDraw.Draw(img)
    draw.rectangle((x, y, x + w, y + h), outline=(255, 255, 255))
    # The segmentation mask is the bbox for some reason :/ not helpful
    segmentations = eval(annot["segmentation"])
    for seg in segmentations:
        draw.polygon(seg, fill=(255, 255, 255), outline=(255, 255, 255))
    return img


def process_bbox(annot, occlusion=True, normalize=True):
    x0, y0, w, h = eval(annot["bbox"])
    x1 = x0 + w
    y1 = y0 + h
    occlusion_ratio_w = 0
    occlusion_ratio_h = 0
    if occlusion:
        # we want to occlude a larger portion vertically. i.e. a forklift behind some boxes
        occlusion_ratio_w = np.random.uniform(0.1, 0.4)
        occlusion_ratio_h = np.random.uniform(0.4, 0.7)
    x0 += w * occlusion_ratio_w
    y0 += h * occlusion_ratio_h
    if normalize:
        x0 /= annot["width"]
        x1 /= annot["width"]
        y0 /= annot["height"]
        y1 /= annot["height"]
    return x0, y0, x1, y1


def make_mask(img, annot):
    img = Image.new("RGB", img.size, (0, 0, 0))

    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = process_bbox(annot, normalize=False)
    draw.rectangle(
        (x0, y0, x1, y1),
        fill=(255, 255, 255),
        outline=(255, 255, 255),
    )
    return img


class SDXLInpaint:
    def __init__(self, device="cuda") -> None:
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
        ).to(device)

        self.generator = torch.Generator(device=device).manual_seed(0)

    def __call__(self, img: Image.Image, annot: dict) -> Image.Image:
        mask = make_mask(img, annot)
        image = self.pipe(
            prompt="a stack of boxes",
            image=img,
            mask_image=mask,
            guidance_scale=5,
            num_inference_steps=30,
            strength=1,
            generator=self.generator,
        ).images[0]
        return image.resize(img.size)


class GLIGEN:
    def __init__(self, device="cuda") -> None:
        self.pipe = StableDiffusionGLIGENPipeline.from_pretrained(
            "masterful/gligen-1-4-inpainting-text-box", variant="fp16", torch_dtype=torch.float16
        ).to(device)
        self.generator = torch.Generator(device=device).manual_seed(0)

    def refine(self, img: Image.Image, annot: dict):
        bbox = [process_bbox(annot, occlusion=False)]
        _img = img.resize((512, 512))
        image = (
            self.pipe(
                prompt="a forklift",
                gligen_phrases=["a forklift"],
                gligen_inpaint_image=_img,
                gligen_boxes=bbox,
                gligen_scheduled_sampling_beta=1,
                output_type="pil",
                num_inference_steps=50,
            )
            .images[0]
            .resize(img.size)
        )
        return image

    def __call__(self, img: Image.Image, annot: dict, refine=False) -> Image.Image:
        if refine:
            img = self.refine(img, annot)
        bbox = [process_bbox(annot)]
        _img = img.resize((512, 512))

        image = (
            self.pipe(
                prompt="a stack of boxes",
                gligen_phrases=["a stack of boxes"],
                gligen_inpaint_image=_img,
                gligen_boxes=bbox,
                gligen_scheduled_sampling_beta=1,
                output_type="pil",
                num_inference_steps=50,
            )
            .images[0]
            .resize(img.size)
        )
        return image


def augment(forklifts_path: Path, save_dir: Path, method: Literal["sdxl", "gligen"] = "gligen"):
    save_dir.mkdir(exist_ok=True)
    if method == "sdxl":
        model = SDXLInpaint()
    # Insert objects described by text at the region defined by bounding boxes
    elif method == "gligen":
        model = GLIGEN()
    annotations = pd.read_csv(forklifts_path / "annotations.csv")

    for annot in annotations.to_dict("records"):
        img = Image.open(forklifts_path.joinpath(annot["file_name"]))
        image = model(img, annot)
        image.save(save_dir / annot["file_name"])


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=Path, required=True, help="The directory where the dataset is stored")
    args.add_argument(
        "--method",
        choices=["sdxl", "gligen"],
        default="gligen",
        required=False,
        help="The method to use for inpainting ",
    )
    pargs = args.parse_args()
    # NOTE this is optional. This is already done and the results are stored in ./dataset folder
    # process_dataset(pargs.dataset)

    augment(pargs.dataset / "forklifts", pargs.dataset / "augmented-forklifts", method = pargs.method)
