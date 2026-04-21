"""
Main script for the Animal Care Visualization System using Stable Diffusion.
Demonstrates the complete pipeline from data to image generation and evaluation.
"""

import os
import json
from data_loader import AnimalDataLoader
from prompt_generator import PromptGenerator
from image_generator import AnimalImageGenerator
from evaluator import ImageEvaluator
from typing import Dict, Any, List

def run_experiment(data_samples: List[Dict[str, Any]],
                  use_controlnet: bool = False,
                  output_dir: str = "outputs",
                  demo_mode: bool = True) -> Dict[str, Any]:
    """Run the complete experiment pipeline."""

    # Initialize components
    prompt_gen = PromptGenerator()
    if not demo_mode:
        generator = AnimalImageGenerator(use_controlnet=use_controlnet)
        evaluator = ImageEvaluator()
    else:
        generator = None
        evaluator = None

    results = []

    for i, data in enumerate(data_samples):
        print(f"Processing sample {i+1}/{len(data_samples)}: {data}")

        # Generate structured prompt
        prompt_data = prompt_gen.create_structured_prompt(data)

        if demo_mode:
            # In demo mode, just show prompts
            print(f"Positive prompt: {prompt_data['positive'][:100]}...")
            print(f"Negative prompt: {prompt_data['negative'][:100]}...")
            result = {
                'sample_id': i,
                'data': data,
                'prompt_data': prompt_data,
                'demo_mode': True
            }
        else:
            # Generate images
            generation_result = generator.generate_from_structured_data(
                prompt_data,
                output_dir=output_dir,
                num_variations=3
            )

            # Evaluate results
            evaluation = evaluator.evaluate_generation(
                generation_result['images'],
                generation_result['prompt'],
                data
            )

            result = {
                'sample_id': i,
                'data': data,
                'prompt_data': prompt_data,
                'generation': generation_result,
                'evaluation': evaluation
            }

            print(f"Generated {len(generation_result['images'])} images")
            print(f"CLIP Score: {evaluation['metrics']['prompt_alignment']['mean_clip_score']:.3f}")
            print(f"Consistency: {evaluation['metrics']['consistency']:.3f}")
            print(f"Diversity: {evaluation['metrics']['diversity']:.3f}")

        results.append(result)
        print("-" * 50)

    return {'experiment_results': results}

def compare_naive_vs_structured(data_sample: Dict[str, Any],
                               output_dir: str = "outputs") -> Dict[str, Any]:
    """Compare naive vs structured prompt approaches."""

    prompt_gen = PromptGenerator()
    generator = AnimalImageGenerator()
    evaluator = ImageEvaluator()

    # Naive prompt
    naive_prompt = f"A {data_sample['animal_type']} in a {data_sample['environment']}"
    naive_negative = "blurry, ugly, deformed"

    # Generate naive images
    naive_images = generator.generate_variations(naive_prompt, naive_negative, 3)
    naive_eval = evaluator.evaluate_generation(naive_images, naive_prompt, data_sample)

    # Structured prompt
    structured_prompt_data = prompt_gen.create_structured_prompt(data_sample)
    structured_images = generator.generate_variations(
        structured_prompt_data['positive'],
        structured_prompt_data['negative'],
        3
    )
    structured_eval = evaluator.evaluate_generation(
        structured_images,
        structured_prompt_data['positive'],
        data_sample
    )

    # Compare
    comparison = evaluator.compare_prompts(naive_eval, structured_eval)

    # Save images
    generator.save_images(naive_images, os.path.join(output_dir, "naive"), "naive")
    generator.save_images(structured_images, os.path.join(output_dir, "structured"), "structured")

    return {
        'data': data_sample,
        'naive': {
            'prompt': naive_prompt,
            'evaluation': naive_eval
        },
        'structured': {
            'prompt': structured_prompt_data,
            'evaluation': structured_eval
        },
        'comparison': comparison
    }

def main():
    """Main function to run the system."""

    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load or generate data
    loader = AnimalDataLoader()
    data_samples = loader.generate_sample_data(3)  # Generate 3 samples for demo

    print("Animal Care Visualization System")
    print("=" * 50)
    print(f"Processing {len(data_samples)} data samples")
    print()

    # Run main experiment in demo mode (no image generation)
    print("Running in DEMO MODE (no image generation)")
    experiment_results = run_experiment(data_samples, output_dir=output_dir, demo_mode=True)

    # Save results
    with open(os.path.join(output_dir, "experiment_results.json"), 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print("\nDemo Summary:")
    print("-" * 30)
    total_samples = len(experiment_results['experiment_results'])
    print(f"Total samples processed: {total_samples}")

    print("\nSample Prompts Generated:")
    for result in experiment_results['experiment_results'][:2]:  # Show first 2
        print(f"Data: {result['data']}")
        print(f"Prompt: {result['prompt_data']['positive'][:150]}...")
        print()

    print(f"\nResults saved to {output_dir}/experiment_results.json")
    print("\nTo run with actual image generation:")
    print("1. Ensure you have sufficient RAM/VRAM")
    print("2. Change demo_mode=False in main.py")
    print("3. Run: python main.py")
    print("\nNote: Image generation requires downloading models (~5GB) and GPU recommended")

if __name__ == "__main__":
    import numpy as np
    main()