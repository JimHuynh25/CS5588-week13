"""
Main script for the Animal Care Visualization System using Stable Diffusion.
Demonstrates the complete pipeline from data to image generation and evaluation.
"""

import os
import json
from data_loader import AnimalDataLoader
from prompt_generator import PromptGenerator
from typing import Dict, Any, List

def get_env_bool(name: str, default: bool) -> bool:
    """Read a boolean setting from an environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

def get_env_int(name: str, default: int) -> int:
    """Read an integer setting from an environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Ignoring invalid {name}={value!r}; using {default}.")
        return default

def run_experiment(data_samples: List[Dict[str, Any]],
                  use_controlnet: bool = False,
                  output_dir: str = "outputs",
                  demo_mode: bool = True,
                  model_id: str = "segmind/tiny-sd",
                  num_variations: int = 1,
                  image_size: int = 384,
                  num_inference_steps: int = 12,
                  guidance_scale: float = 7.5,
                  evaluate_images: bool = False) -> Dict[str, Any]:
    """Run the complete experiment pipeline."""

    # Initialize components
    prompt_gen = PromptGenerator()
    if not demo_mode:
        from image_generator import AnimalImageGenerator

        generator = AnimalImageGenerator(model_id=model_id, use_controlnet=use_controlnet)
        if evaluate_images:
            from evaluator import ImageEvaluator
            evaluator = ImageEvaluator()
        else:
            evaluator = None
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
                num_variations=num_variations,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=image_size,
                height=image_size
            )

            if evaluator is not None:
                evaluation = evaluator.evaluate_generation(
                    generation_result['images'],
                    generation_result['prompt'],
                    data
                )
            else:
                evaluation = None

            generation_summary = {
                key: value
                for key, value in generation_result.items()
                if key != 'images'
            }

            result = {
                'sample_id': i,
                'data': data,
                'prompt_data': prompt_data,
                'generation': generation_summary,
                'evaluation': evaluation
            }

            print(f"Generated {len(generation_result['paths'])} image(s)")
            print(f"Saved to: {', '.join(generation_result['paths'])}")
            if evaluation is not None:
                print(f"CLIP Score: {evaluation['metrics']['prompt_alignment']['mean_clip_score']:.3f}")
                print(f"Consistency: {evaluation['metrics']['consistency']:.3f}")
                print(f"Diversity: {evaluation['metrics']['diversity']:.3f}")

        results.append(result)
        print("-" * 50)

    return {'experiment_results': results}

def compare_naive_vs_structured(data_sample: Dict[str, Any],
                               output_dir: str = "outputs") -> Dict[str, Any]:
    """Compare naive vs structured prompt approaches."""

    from image_generator import AnimalImageGenerator
    from evaluator import ImageEvaluator

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

    generate_images = get_env_bool("GENERATE_IMAGES", True)
    sample_count = get_env_int("SAMPLE_COUNT", 1 if generate_images else 3)
    num_variations = get_env_int("NUM_VARIATIONS", 1)
    image_size = get_env_int("IMAGE_SIZE", 384)
    num_inference_steps = get_env_int("INFERENCE_STEPS", 12)
    evaluate_images = get_env_bool("EVALUATE_IMAGES", False)
    model_id = os.getenv("MODEL_ID", "segmind/tiny-sd")

    # Load or generate data
    loader = AnimalDataLoader()
    data_samples = loader.generate_sample_data(sample_count)

    print("Animal Care Visualization System")
    print("=" * 50)
    print(f"Processing {len(data_samples)} data samples")
    print()

    if generate_images:
        print("Running in IMAGE GENERATION MODE (low-memory defaults)")
        print(f"Model: {model_id}")
        print(f"Images per sample: {num_variations}")
        print(f"Image size: {image_size}x{image_size}")
        print(f"Inference steps: {num_inference_steps}")
        print()
    else:
        print("Running in DEMO MODE (no image generation)")

    experiment_results = run_experiment(
        data_samples,
        output_dir=output_dir,
        demo_mode=not generate_images,
        model_id=model_id,
        num_variations=num_variations,
        image_size=image_size,
        num_inference_steps=num_inference_steps,
        evaluate_images=evaluate_images
    )

    # Save results
    with open(os.path.join(output_dir, "experiment_results.json"), 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print("\nSummary:")
    print("-" * 30)
    total_samples = len(experiment_results['experiment_results'])
    print(f"Total samples processed: {total_samples}")

    print("\nSample Prompts Generated:")
    for result in experiment_results['experiment_results'][:2]:  # Show first 2
        print(f"Data: {result['data']}")
        print(f"Prompt: {result['prompt_data']['positive'][:150]}...")
        print()

    print(f"\nResults saved to {output_dir}/experiment_results.json")
    if generate_images:
        print(f"Generated image paths are listed in {output_dir}/experiment_results.json")
    else:
        print("\nTo run with low-memory image generation:")
        print("Run: python main.py")
    print("\nLow-memory settings can be changed with MODEL_ID, IMAGE_SIZE, INFERENCE_STEPS, SAMPLE_COUNT, and NUM_VARIATIONS.")

if __name__ == "__main__":
    main()
