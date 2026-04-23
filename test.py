"""
Test script to validate the system components without heavy dependencies.
"""

def test_data_loader():
    """Test data loader functionality."""
    from data_loader import AnimalDataLoader

    loader = AnimalDataLoader()
    samples = loader.generate_sample_data(2)

    assert len(samples) == 2
    assert all(key in samples[0] for key in ['animal_type', 'breed', 'condition', 'environment'])
    print("Data loader test passed")

def test_prompt_generator():
    """Test prompt generator functionality."""
    from prompt_generator import PromptGenerator

    generator = PromptGenerator()
    data = {
        'animal_type': 'dog',
        'breed': 'Golden Retriever',
        'condition': 'healthy',
        'environment': 'home'
    }

    prompt = generator.generate_prompt(data)
    educational_prompt = generator.generate_prompt(data, style="educational")
    negative = generator.generate_negative_prompt()

    assert 'Golden Retriever' in prompt
    assert 'dog' in prompt
    assert 'healthy' in prompt
    assert 'home' in prompt
    assert 'veterinary reference image' in educational_prompt
    assert len(negative) > 0
    print("Prompt generator test passed")

def test_imports():
    """Test that all modules can be imported."""
    try:
        import data_loader
        import prompt_generator
        import image_generator
        import evaluator
        import main
        print("All modules imported successfully")
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    return True

if __name__ == "__main__":
    print("Running system validation tests...")
    print("-" * 40)

    test_imports()
    test_data_loader()
    test_prompt_generator()

    print("-" * 40)
    print("Basic validation complete!")
    print("\nTo run the full system:")
    print("1. Install Python 3.8+")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run: python main.py")
    print("\nNote: Image generation requires GPU for reasonable performance")
