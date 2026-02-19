
import json
import random
import numpy as np
from typing import List, Dict, Tuple

class SyntheticDataGenerator:
    """
    Generate synthetic educational queries for training

    Data Generation Approach:
    1. Template-based generation with variations
    2. Intent-specific patterns
    3. Topic-difficulty combinations
    4. Natural language variations

    Assumptions:
    - Students ask questions in predictable patterns
    - Intent can be inferred from question structure
    - Difficulty correlates with technical terminology

    Limitations:
    - May not capture all real-world variations
    - Limited to predefined topics
    - English language only
    - Template-based may lack naturalness
    """

    def __init__(self):
        self.intent_templates = {
            'Explanation': [
                "What is {topic}?",
                "Can you explain {topic}?",
                "I don't understand {topic}",
                "How does {topic} work?",
                "What does {topic} mean?",
                "Define {topic}",
                "Explain the concept of {topic}",
                "Help me understand {topic}"
            ],
            'Example': [
                "Show me an example of {topic}",
                "Can you give me a {topic} example?",
                "I need examples for {topic}",
                "Demonstrate {topic} with examples",
                "Show {topic} in action",
                "Provide sample code for {topic}",
                "What are some {topic} examples?"
            ],
            'Doubt clarification': [
                "I'm confused about {topic}",
                "I'm stuck on {topic}",
                "Having trouble with {topic}",
                "Can't figure out {topic}",
                "Need help with {topic}",
                "Why isn't {topic} working?",
                "What am I missing in {topic}?"
            ],
            'Revision': [
                "Summarize {topic}",
                "Quick review of {topic}",
                "Recap {topic} for me",
                "What are the key points of {topic}?",
                "Refresh my memory on {topic}",
                "Review {topic} concepts",
                "Main takeaways from {topic}?"
            ]
        }

        self.topics = {
            'Beginner': [
                'variables', 'loops', 'functions', 'arrays', 'basic statistics',
                'linear regression', 'data types', 'conditional statements'
            ],
            'Intermediate': [
                'neural networks', 'backpropagation', 'gradient descent',
                'overfitting', 'regularization', 'cross-validation',
                'decision trees', 'random forests', 'feature engineering'
            ],
            'Advanced': [
                'transformers', 'attention mechanism', 'GANs', 'reinforcement learning',
                'PPO algorithm', 'meta-learning', 'few-shot learning',
                'neural architecture search', 'model compression'
            ]
        }

        self.difficulty_indicators = {
            'Beginner': ['basic', 'simple', 'introduction to', 'starting with'],
            'Intermediate': ['intermediate', 'standard', 'typical'],
            'Advanced': ['advanced', 'complex', 'sophisticated', 'deep dive into']
        }

    def generate_query(self, intent: str, difficulty: str) -> Dict:
        """Generate a single query"""
        topic = random.choice(self.topics[difficulty])
        template = random.choice(self.intent_templates[intent])
        query = template.format(topic=topic)

        # Add difficulty indicator randomly
        if random.random() > 0.7:
            indicator = random.choice(self.difficulty_indicators[difficulty])
            query = f"{indicator} {query}"

        return {
            'query': query,
            'intent': intent,
            'topic': topic,
            'difficulty': difficulty
        }

    def generate_dataset(self, samples_per_category: int = 250) -> List[Dict]:
        """
        Generate complete dataset

        Total samples: 4 intents × 3 difficulties × samples_per_category
        = 12 × 250 = 3000 samples (approx 200KB)
        """
        dataset = []

        for intent in self.intent_templates.keys():
            for difficulty in self.topics.keys():
                for _ in range(samples_per_category):
                    dataset.append(self.generate_query(intent, difficulty))

        # Shuffle dataset
        random.shuffle(dataset)
        return dataset

    def save_dataset(self, filepath: str, samples_per_category: int = 250):
        """Generate and save dataset"""
        dataset = self.generate_dataset(samples_per_category)

        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"✅ Generated {len(dataset)} samples")
        print(f"📁 Saved to: {filepath}")

        # Statistics
        intents = {}
        difficulties = {}
        for item in dataset:
            intents[item['intent']] = intents.get(item['intent'], 0) + 1
            difficulties[item['difficulty']] = difficulties.get(item['difficulty'], 0) + 1

        print("\n📊 Dataset Statistics:")
        print(f"Intent distribution: {intents}")
        print(f"Difficulty distribution: {difficulties}")

        return dataset

# Generate dataset
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    dataset = generator.save_dataset(
        'ai_teaching_assistant/data/synthetic_queries.json',
        samples_per_category=250
    )
