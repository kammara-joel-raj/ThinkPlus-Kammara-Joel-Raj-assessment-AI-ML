import os

class Config:
    """Configuration for AI Teaching Assistant"""

    # Model Configuration
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # 384 dim, lightweight
    INTENT_MODEL_NAME = "distilbert-base-uncased"

    # Intent Categories
    INTENT_CLASSES = [
        "Explanation",
        "Example", 
        "Doubt clarification",
        "Revision"
    ]

    # Difficulty Levels
    DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced"]

    # Topic Categories (expandable)
    TOPIC_CATEGORIES = [
        "Neural Networks",
        "Machine Learning",
        "Deep Learning",
        "Natural Language Processing",
        "Computer Vision",
        "Reinforcement Learning",
        "Statistics",
        "Linear Algebra",
        "Calculus",
        "Optimization",
        "Backpropagation",
        "Transformers",
        "CNNs",
        "RNNs",
        "Gradient Descent"
    ]

    # Data paths
    DATA_DIR = "ai_teaching_assistant/data"
    MODEL_DIR = "ai_teaching_assistant/models"

    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    MAX_SEQ_LENGTH = 128

    # RL parameters for adaptive learning
    RL_GAMMA = 0.99
    RL_LR = 3e-4
    RL_CLIP_RANGE = 0.2

    # Knowledge graph structure
    KNOWLEDGE_GRAPH = {
        'Neural Networks': {
            'prerequisites': ['Linear Algebra', 'Calculus', 'Python'],
            'topics': ['Perceptron', 'Activation Functions', 'Backpropagation', 'Gradient Descent'],
            'difficulty': 'Intermediate',
            'estimated_time': 120
        },
        'Backpropagation': {
            'prerequisites': ['Chain Rule', 'Derivatives', 'Neural Networks'],
            'topics': ['Gradient Computation', 'Weight Updates', 'Loss Functions'],
            'difficulty': 'Intermediate',
            'estimated_time': 45
        },
        'Machine Learning': {
            'prerequisites': ['Statistics', 'Linear Algebra', 'Programming'],
            'topics': ['Supervised Learning', 'Unsupervised Learning', 'Model Evaluation'],
            'difficulty': 'Beginner',
            'estimated_time': 90
        },
        'Deep Learning': {
            'prerequisites': ['Neural Networks', 'Optimization', 'Linear Algebra'],
            'topics': ['CNNs', 'RNNs', 'Transformers', 'Attention'],
            'difficulty': 'Advanced',
            'estimated_time': 180
        }
    }