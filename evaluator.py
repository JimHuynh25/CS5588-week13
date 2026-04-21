"""
Evaluator for generated images in animal care visualization system.
Implements metrics for prompt alignment, consistency, diversity, and visual quality.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Dict, Any, Tuple

class ImageEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP model for prompt alignment
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)

    def calculate_clip_score(self, image: Image.Image, prompt: str) -> float:
        """Calculate CLIP score for prompt-image alignment."""
        inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.item()

        return score

    def calculate_image_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculate structural similarity between two images."""
        # Convert to numpy arrays
        img1_array = np.array(img1.convert('RGB'))
        img2_array = np.array(img2.convert('RGB'))

        # Convert to grayscale for SSIM
        img1_gray = cv2.cvtColor(img1_array, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2_array, cv2.COLOR_RGB2GRAY)

        # Calculate SSIM
        ssim_score = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCORR_NORMED)[0][0]

        return ssim_score

    def calculate_diversity(self, images: List[Image.Image]) -> float:
        """Calculate diversity among a set of images (lower similarity = higher diversity)."""
        if len(images) < 2:
            return 0.0

        similarities = []
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                sim = self.calculate_image_similarity(images[i], images[j])
                similarities.append(sim)

        # Average similarity, diversity = 1 - average_similarity
        avg_similarity = np.mean(similarities)
        diversity = 1 - avg_similarity

        return diversity

    def calculate_consistency(self, images: List[Image.Image]) -> float:
        """Calculate consistency among variations (higher similarity = higher consistency)."""
        if len(images) < 2:
            return 1.0

        similarities = []
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                sim = self.calculate_image_similarity(images[i], images[j])
                similarities.append(sim)

        return np.mean(similarities)

    def evaluate_visual_quality(self, image: Image.Image) -> Dict[str, float]:
        """Basic visual quality metrics."""
        img_array = np.array(image.convert('RGB'))

        # Brightness
        brightness = np.mean(img_array)

        # Contrast (standard deviation)
        contrast = np.std(img_array)

        # Sharpness (using Laplacian variance)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness
        }

    def evaluate_generation(self,
                           images: List[Image.Image],
                           prompt: str,
                           data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive evaluation of generated images."""
        results = {
            'num_images': len(images),
            'prompt': prompt,
            'data': data,
            'metrics': {}
        }

        # Prompt alignment (average CLIP score)
        clip_scores = [self.calculate_clip_score(img, prompt) for img in images]
        results['metrics']['prompt_alignment'] = {
            'mean_clip_score': np.mean(clip_scores),
            'std_clip_score': np.std(clip_scores),
            'individual_scores': clip_scores
        }

        # Consistency
        consistency = self.calculate_consistency(images)
        results['metrics']['consistency'] = consistency

        # Diversity
        diversity = self.calculate_diversity(images)
        results['metrics']['diversity'] = diversity

        # Visual quality (average across images)
        quality_metrics = [self.evaluate_visual_quality(img) for img in images]
        avg_quality = {
            key: np.mean([qm[key] for qm in quality_metrics])
            for key in quality_metrics[0].keys()
        }
        results['metrics']['visual_quality'] = avg_quality

        return results

    def compare_prompts(self,
                       naive_results: Dict[str, Any],
                       structured_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare naive vs structured prompt results."""
        comparison = {
            'naive': naive_results['metrics'],
            'structured': structured_results['metrics'],
            'improvements': {}
        }

        # Calculate improvements
        for metric in ['prompt_alignment', 'consistency', 'diversity']:
            if metric in naive_results['metrics'] and metric in structured_results['metrics']:
                if metric == 'prompt_alignment':
                    naive_score = naive_results['metrics'][metric]['mean_clip_score']
                    structured_score = structured_results['metrics'][metric]['mean_clip_score']
                else:
                    naive_score = naive_results['metrics'][metric]
                    structured_score = structured_results['metrics'][metric]

                improvement = structured_score - naive_score
                comparison['improvements'][metric] = improvement

        return comparison

    def save_evaluation_report(self, evaluation: Dict[str, Any], file_path: str):
        """Save evaluation results to JSON file."""
        import json
        with open(file_path, 'w') as f:
            json.dump(evaluation, f, indent=2)

if __name__ == "__main__":
    # Example usage
    evaluator = ImageEvaluator()

    # Mock images (in real usage, load actual generated images)
    mock_image = Image.new('RGB', (512, 512), color='gray')

    images = [mock_image] * 3  # Simulate 3 variations
    prompt = "A healthy dog in a home"
    data = {'animal_type': 'dog', 'condition': 'healthy'}

    results = evaluator.evaluate_generation(images, prompt, data)
    print("Evaluation results:")
    print(f"Prompt alignment: {results['metrics']['prompt_alignment']['mean_clip_score']:.3f}")
    print(f"Consistency: {results['metrics']['consistency']:.3f}")
    print(f"Diversity: {results['metrics']['diversity']:.3f}")
    print(f"Visual quality: {results['metrics']['visual_quality']}")