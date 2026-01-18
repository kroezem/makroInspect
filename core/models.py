import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor
from transformers import Sam3Processor, Sam3Model


class ModelManager:
    def __init__(self, device="cuda"):
        self.device = device
        self.eval_repo = "facebookresearch/dinov2"
        self.eval_model = "dinov2_vitb14"
        self.seg_repo = "facebook/sam3"
        self.patch_size = 14

        # Residents
        self.segmenter = None
        self.processor = None
        self.evaluator = None

        self.io_pool = ThreadPoolExecutor(max_workers=4)

        # DINO Norms
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device).half()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device).half()

    def load_evaluator(self):
        if self.evaluator is not None: return
        self.evaluator = torch.hub.load(self.eval_repo, self.eval_model)
        self.evaluator.to(self.device).half().eval()
        # Warmup
        dummy = torch.zeros(1, 3, 14, 14, device=self.device, dtype=torch.float16)
        with torch.inference_mode():
            self.evaluator.forward_features(dummy)

    def load_segmenter(self):
        if self.segmenter is not None: return
        print(f"[ModelManager] Loading SAM3 (SDPA/FP16)...")
        self.processor = Sam3Processor.from_pretrained(self.seg_repo)

        self.segmenter = Sam3Model.from_pretrained(
            self.seg_repo,
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        ).to(self.device)
        self.segmenter.eval()

    @torch.no_grad()
    def compute_patches(self, paths: List[Path]):
        self.load_evaluator()
        # Threaded IO
        rgb_imgs = list(self.io_pool.map(self._read_image_fast, paths))
        tensors = [torch.from_numpy(i).permute(2, 0, 1) for i in rgb_imgs if i is not None]
        if not tensors: return None

        x = torch.stack(tensors).to(self.device).half()

        with torch.inference_mode():
            out = self.evaluator.forward_features(x)
            tokens = out["x_norm_patchtokens"] if isinstance(out, dict) else out[0]
            N, L, C = tokens.shape
            h, w = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
            return tokens.reshape(N, h, w, C).contiguous()

    @torch.no_grad()
    def segment_batch(self, pil_images: List, prompt: Union[str, List[str]] = "object"):
        self.load_segmenter()
        if not pil_images: return []

        # 1. Handle Prompt Alignment (Fix for Multi-Project)
        if isinstance(prompt, str):
            prompt_list = [prompt] * len(pil_images)
        else:
            prompt_list = prompt

        # 2. Pre-process
        inputs = self.processor(
            images=pil_images,
            text=prompt_list,
            return_tensors="pt"
        ).to(self.device)

        # 3. FP16 Conversion
        inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

        # 4. Inference & Return
        with torch.inference_mode():
            outputs = self.segmenter(**inputs)
            target_sizes = [img.size[::-1] for img in pil_images]
            return self.processor.post_process_instance_segmentation(
                outputs, target_sizes=target_sizes, threshold=0.5
            )

    def _read_image_fast(self, path):
        try:
            img = cv2.imread(str(path))
            if img is None: return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            return None
