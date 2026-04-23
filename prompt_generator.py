"""
Prompt generator for animal care visualization system.
Converts structured input data into detailed prompts for Stable Diffusion.
"""

from typing import Dict, Any, List
import random

class PromptGenerator:
    def __init__(self):
        # Prompt templates for different conditions
        self.condition_templates = {
            'healthy': [
                "A healthy {breed} {animal_type} in a {environment}, looking vibrant and energetic",
                "Beautiful photograph of a healthy {breed} {animal_type} in {environment}, full of life",
                "High-quality image of a well-cared {breed} {animal_type} in {environment}"
            ],
            'injured': [
                "A {breed} {animal_type} with an injury in a veterinary clinic, receiving care",
                "Educational illustration of an injured {breed} {animal_type} in {environment}, showing medical attention",
                "Realistic depiction of a {breed} {animal_type} with injury in {environment}, veterinary care setting"
            ],
            'sick': [
                "A sick {breed} {animal_type} in {environment}, showing signs of illness, veterinary illustration",
                "Medical visualization of a {breed} {animal_type} with health condition in {environment}",
                "Illustrative image of an unwell {breed} {animal_type} in {environment}, educational purpose"
            ],
            'elderly': [
                "An elderly {breed} {animal_type} in {environment}, showing signs of age gracefully",
                "Senior {breed} {animal_type} in {environment}, dignified and cared for",
                "Mature {breed} {animal_type} in {environment}, receiving appropriate care"
            ],
            'pregnant': [
                "A pregnant {breed} {animal_type} in {environment}, expecting offspring",
                "Expectant {breed} {animal_type} in {environment}, showing pregnancy care",
                "Maternal {breed} {animal_type} in {environment}, preparing for birth"
            ],
            'malnourished': [
                "A malnourished {breed} {animal_type} in {environment}, showing signs of malnutrition",
                "Educational image of an undernourished {breed} {animal_type} in {environment}",
                "Illustrative depiction of malnutrition in {breed} {animal_type} in {environment}"
            ]
        }

        # Negative prompts to avoid unwanted elements
        self.negative_prompts = [
            "blurry, low quality, distorted, ugly, deformed, disfigured, bad anatomy, malformed face, asymmetrical eyes",
            "cartoon, anime, illustration, painting, drawing, cgi, 3d render, digital art, oversaturated",
            "close-up portrait, cropped, out of frame, duplicate animal, extra limbs, extra ears, fused features",
            "human, person, people, crowd, text, watermark",
            "violent, scary, frightening, harmful, dangerous"
        ]

        self.supported_styles = {"realistic", "educational", "artistic"}

    def generate_prompt(self, data: Dict[str, Any], style: str = "realistic") -> str:
        """Generate a detailed prompt from structured data."""
        style = style.strip().lower()
        if style not in self.supported_styles:
            style = "realistic"

        animal_type = data['animal_type']
        breed = data['breed']
        condition = data['condition']
        environment = data['environment']

        override = data.get('prompt_override')
        if override:
            prompt = override.strip()
        else:
            # Select random template for the condition
            templates = self.condition_templates.get(condition, self.condition_templates['healthy'])
            template = random.choice(templates)

            # Fill in the template
            prompt = template.format(
                breed=breed,
                animal_type=animal_type,
                environment=environment
            )

        # Add style and quality descriptors
        if style == "realistic":
            prompt += ", realistic veterinary photograph, photorealistic, natural lighting, high detail, professional photography"
        elif style == "educational":
            prompt += ", educational veterinary reference image, clinically clear, informative composition, medically accurate, realistic anatomy"
        elif style == "artistic":
            prompt += ", artistic representation, beautiful composition"

        prompt += ", AI-generated content for educational purposes"

        return prompt

    def generate_negative_prompt(self) -> str:
        """Generate a combined negative prompt."""
        return ", ".join(self.negative_prompts)

    def generate_variations(self, data: Dict[str, Any], num_variations: int = 3) -> List[str]:
        """Generate multiple prompt variations for the same input."""
        prompts = []
        for _ in range(num_variations):
            prompt = self.generate_prompt(data)
            prompts.append(prompt)
        return prompts

    def create_structured_prompt(self, data: Dict[str, Any], style: str = "realistic") -> Dict[str, str]:
        """Create a complete prompt structure with positive and negative prompts."""
        positive = self.generate_prompt(data, style=style)
        negative = self.generate_negative_prompt()

        return {
            'positive': positive,
            'negative': negative,
            'style': style,
            'data': data
        }

if __name__ == "__main__":
    import random
    random.seed(42)  # For reproducible results

    generator = PromptGenerator()
    sample_data = {
        'animal_type': 'dog',
        'breed': 'Golden Retriever',
        'condition': 'healthy',
        'environment': 'home'
    }

    prompt = generator.generate_prompt(sample_data)
    negative = generator.generate_negative_prompt()

    print("Positive prompt:", prompt)
    print("Negative prompt:", negative)

    variations = generator.generate_variations(sample_data, 3)
    print("Variations:")
    for i, var in enumerate(variations, 1):
        print(f"{i}. {var}")
