"""
Image generator using Stable Diffusion for animal care visualization.
Supports basic generation and ControlNet conditioning.
"""

import torch
from diffusers import DiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from PIL import Image
import os
from typing import Dict, Any, List, Optional

class AnimalImageGenerator:
    def __init__(self, model_id: str = "segmind/tiny-sd", use_controlnet: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_controlnet = use_controlnet
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        if use_controlnet:
            # Load ControlNet model (Canny edge detection)
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )

        # Keep memory use low. CPU offload helps small GPUs; CPU-only still runs normally.
        self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()
        if self.device == "cuda" and hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

    def preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocess image for ControlNet (Canny edge detection)."""
        import cv2
        import numpy as np

        image = load_image(image_path)
        image = np.array(image)

        # Convert to grayscale and apply Canny
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

        return image

    def generate_image(self,
                      prompt: str,
                      negative_prompt: str = "",
                      num_inference_steps: int = 12,
                      guidance_scale: float = 7.5,
                      width: int = 384,
                      height: int = 384,
                      control_image: Optional[Image.Image] = None) -> Image.Image:
        """Generate a single image from prompt."""
        if self.use_controlnet and control_image is not None:
            image = self.pipe(
                prompt,
                image=control_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]
        else:
            image = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]

        return image

    def generate_variations(self,
                           prompt: str,
                           negative_prompt: str = "",
                           num_images: int = 1,
                           num_inference_steps: int = 12,
                           guidance_scale: float = 7.5,
                           width: int = 384,
                           height: int = 384,
                           control_image: Optional[Image.Image] = None) -> List[Image.Image]:
        """Generate multiple image variations."""
        images = []
        for _ in range(num_images):
            image = self.generate_image(
                prompt,
                negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                control_image=control_image
            )
            images.append(image)
        return images

    def save_images(self, images: List[Image.Image], output_dir: str, prefix: str = "generated"):
        """Save generated images to disk."""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        for i, img in enumerate(images):
            path = os.path.join(output_dir, f"{prefix}_{i+1}.png")
            img.save(path)
            saved_paths.append(path)
        return saved_paths

    def generate_from_structured_data(self,
                                     prompt_data: Dict[str, str],
                                     output_dir: str = "outputs",
                                     num_variations: int = 1,
                                     num_inference_steps: int = 12,
                                     guidance_scale: float = 7.5,
                                     width: int = 384,
                                     height: int = 384) -> Dict[str, Any]:
        """Generate images from structured prompt data."""
        positive = prompt_data['positive']
        negative = prompt_data['negative']
        data = prompt_data['data']

        # Generate variations
        images = self.generate_variations(
            positive,
            negative,
            num_variations,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        )

        # Save images
        prefix = f"{data['animal_type']}_{data['breed']}_{data['condition']}".replace(" ", "_")
        saved_paths = self.save_images(images, output_dir, prefix)

        return {
            'images': images,
            'paths': saved_paths,
            'prompt': positive,
            'negative_prompt': negative,
            'data': data
        }

if __name__ == "__main__":
    # Example usage
    generator = AnimalImageGenerator(use_controlnet=False)

    prompt_data = {
        'positive': "A healthy Golden Retriever dog in a home, looking vibrant and energetic, realistic photograph, high detail, professional photography, AI-generated illustration for educational purposes",
        'negative': "blurry, low quality, distorted, ugly, deformed, disfigured, cartoon, anime, illustration, painting, drawing, human, person, people, crowd, text, watermark, violent, scary, frightening, harmful, dangerous",
        'data': {
            'animal_type': 'dog',
            'breed': 'Golden Retriever',
            'condition': 'healthy',
            'environment': 'home'
        }
    }

    result = generator.generate_from_structured_data(prompt_data, num_variations=1)
    print("Generated images saved to:", result['paths'])
