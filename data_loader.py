"""
Data loader for animal care visualization system.
Loads or generates structured input data for image generation.
"""

import json
import random
from typing import List, Dict, Any

class AnimalDataLoader:
    def __init__(self):
        # Sample data for demonstration
        self.animal_types = ['dog', 'cat', 'bird', 'horse', 'rabbit']
        self.breeds = {
            'dog': ['Golden Retriever', 'Labrador', 'German Shepherd', 'Poodle', 'Bulldog'],
            'cat': ['Persian', 'Siamese', 'Maine Coon', 'British Shorthair', 'Ragdoll'],
            'bird': ['Parrot', 'Canary', 'Cockatiel', 'Finch', 'Owl'],
            'horse': ['Thoroughbred', 'Arabian', 'Quarter Horse', 'Appaloosa', 'Mustang'],
            'rabbit': ['Dutch', 'Mini Lop', 'Netherland Dwarf', 'Rex', 'Lionhead']
        }
        self.conditions = ['healthy', 'injured', 'sick', 'elderly', 'pregnant', 'malnourished']
        self.environments = ['veterinary clinic', 'farm', 'home', 'wild', 'zoo', 'shelter']

    def generate_sample_data(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Generate sample structured input data."""
        data = []
        for _ in range(num_samples):
            animal_type = random.choice(self.animal_types)
            breed = random.choice(self.breeds[animal_type])
            condition = random.choice(self.conditions)
            environment = random.choice(self.environments)

            data.append({
                'animal_type': animal_type,
                'breed': breed,
                'condition': condition,
                'environment': environment
            })
        return data

    def load_from_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def save_to_json(self, data: List[Dict[str, Any]], file_path: str):
        """Save data to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    loader = AnimalDataLoader()
    sample_data = loader.generate_sample_data(5)
    print("Sample data:")
    for item in sample_data:
        print(item)
    loader.save_to_json(sample_data, 'sample_data.json')