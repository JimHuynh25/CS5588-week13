"""
Image generator using Stable Diffusion for animal care visualization.
Supports basic generation and ControlNet conditioning.
"""

import torch
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
import cv2
import numpy as np
from PIL import Image
import os
from typing import Dict, Any, List, Optional

class AnimalImageGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", use_controlnet: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_controlnet = use_controlnet

        if use_controlnet:
            # Load ControlNet model (Canny edge detection)
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16
            )
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id,
                controlnet=controlnet,
                torch_dtype=torch.float16
            )
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            )

        self.pipe.to(self.device)

        # Enable attention slicing for memory efficiency
        self.pipe.enable_attention_slicing()

    def preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocess image for ControlNet (Canny edge detection)."""
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
                      num_inference_steps: int = 20,
                      guidance_scale: float = 7.5,
                      control_image: Optional[Image.Image] = None) -> Image.Image:
        """Generate a single image from prompt."""
        if self.use_controlnet and control_image is not None:
            image = self.pipe(
                prompt,
                image=control_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        else:
            image = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

        return image

    def generate_variations(self,
                           prompt: str,
                           negative_prompt: str = "",
                           num_images: int = 3,
                           control_image: Optional[Image.Image] = None) -> List[Image.Image]:
        """Generate multiple image variations."""
        images = []
        for _ in range(num_images):
            image = self.generate_image(prompt, negative_prompt, control_image=control_image)
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
                                     num_variations: int = 3) -> Dict[str, Any]:
        """Generate images from structured prompt data."""
        positive = prompt_data['positive']
        negative = prompt_data['negative']
        data = prompt_data['data']

        # Generate variations
        images = self.generate_variations(positive, negative, num_variations)

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

    result = generator.generate_from_structured_data(prompt_data, num_variations=2)
    print("Generated images saved to:", result['paths'])