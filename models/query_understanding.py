
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple
import json

class QueryUnderstandingModule:
    """
    Student Query Understanding using NLP + ML

    Architecture:
    1. Sentence Embeddings: SBERT (all-MiniLM-L6-v2) - 384 dimensions
    2. Intent Classification: Multi-class classifier
    3. Topic Extraction: Semantic similarity matching
    4. Difficulty Assessment: Complexity analysis

    Separation of Concerns:
    - ML: Embeddings, classification, clustering
    - LLM: Used for prompt-based refinement (optional)
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load SBERT model for embeddings
        print("📥 Loading Sentence-BERT model...")
        self.embedder = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
        self.embedder.to(self.device)

        # Precompute topic embeddings for fast similarity search
        self.topic_embeddings = self._precompute_topic_embeddings()

        # Intent classification patterns (rule-based + ML hybrid)
        self.intent_patterns = {
            'Explanation': [
                'what is', 'explain', 'how does', 'understand', 
                'mean', 'define', 'concept', 'definition'
            ],
            'Example': [
                'example', 'show me', 'demonstrate', 'illustrate',
                'sample', 'code', 'instance'
            ],
            'Doubt clarification': [
                'confused', 'unclear', 'stuck', 'help',
                'problem', 'issue', 'trouble', 'wrong'
            ],
            'Revision': [
                'review', 'summarize', 'recap', 'refresh',
                'revise', 'go over', 'key points'
            ]
        }

        # Difficulty indicators
        self.difficulty_keywords = {
            'Beginner': [
                'basic', 'simple', 'start', 'begin', 'introduction',
                'first', 'elementary', 'fundamental'
            ],
            'Advanced': [
                'advanced', 'complex', 'sophisticated', 'deep',
                'architecture', 'optimization', 'state-of-the-art'
            ]
        }

    def _precompute_topic_embeddings(self) -> Dict[str, torch.Tensor]:
        """Precompute embeddings for all topics"""
        topics = self.config.TOPIC_CATEGORIES
        embeddings = self.embedder.encode(
            topics,
            convert_to_tensor=True,
            device=self.device
        )
        return {topic: emb for topic, emb in zip(topics, embeddings)}

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate SBERT embedding for query
        Returns 384-dimensional vector
        """
        embedding = self.embedder.encode(
            query,
            convert_to_tensor=True,
            device=self.device
        )
        return embedding.cpu().numpy()

    def classify_intent(self, query: str, embedding: torch.Tensor = None) -> Tuple[str, float]:
        """
        Multi-stage intent classification:
        1. Pattern matching (rule-based)
        2. Embedding similarity (ML-based)
        3. Confidence scoring
        """
        query_lower = query.lower()

        # Stage 1: Pattern matching with scoring
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            intent_scores[intent] = score

        # Stage 2: If no clear winner, use embedding similarity
        max_score = max(intent_scores.values())
        if max_score == 0 or list(intent_scores.values()).count(max_score) > 1:
            # Use embedding-based classification
            if embedding is None:
                embedding = self.get_query_embedding(query)

            # Create intent prototypes (could be learned from data)
            intent_prototypes = {
                'Explanation': 'explain and define the concept clearly',
                'Example': 'show practical examples and demonstrations',
                'Doubt clarification': 'help solve problems and confusions',
                'Revision': 'summarize and review key points'
            }

            proto_embeddings = self.embedder.encode(
                list(intent_prototypes.values()),
                convert_to_tensor=True,
                device=self.device
            )

            query_tensor = torch.tensor(embedding).to(self.device)
            similarities = util.cos_sim(query_tensor, proto_embeddings)[0]

            best_idx = torch.argmax(similarities).item()
            intent = list(intent_prototypes.keys())[best_idx]
            confidence = similarities[best_idx].item()
        else:
            # Use pattern matching result
            intent = max(intent_scores, key=intent_scores.get)
            confidence = min(0.6 + (max_score * 0.1), 0.95)

        return intent, confidence

    def extract_topic(self, query: str, embedding: torch.Tensor = None) -> Tuple[str, float]:
        """
        Topic extraction using semantic similarity
        Compares query embedding with precomputed topic embeddings
        """
        if embedding is None:
            query_tensor = self.embedder.encode(
                query,
                convert_to_tensor=True,
                device=self.device
            )
        else:
            query_tensor = torch.tensor(embedding).to(self.device)

        # Compute cosine similarity with all topics
        max_similarity = -1
        best_topic = self.config.TOPIC_CATEGORIES[0]

        for topic, topic_emb in self.topic_embeddings.items():
            similarity = util.cos_sim(query_tensor, topic_emb).item()
            if similarity > max_similarity:
                max_similarity = similarity
                best_topic = topic

        # Also check for direct mentions
        query_lower = query.lower()
        for topic in self.config.TOPIC_CATEGORIES:
            if topic.lower() in query_lower:
                return topic, 0.95

        confidence = max(0.4, min(max_similarity, 0.95))
        return best_topic, confidence

    def assess_difficulty(self, query: str, topic: str) -> Tuple[str, float]:
        """
        Difficulty assessment using:
        1. Keyword indicators
        2. Query complexity (length, technical terms)
        3. Topic inherent difficulty
        """
        query_lower = query.lower()

        # Check explicit difficulty indicators
        for difficulty, keywords in self.difficulty_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return difficulty, 0.85

        # Use topic's inherent difficulty
        if topic in self.config.KNOWLEDGE_GRAPH:
            inherent_difficulty = self.config.KNOWLEDGE_GRAPH[topic]['difficulty']
            return inherent_difficulty, 0.70

        # Analyze query complexity
        word_count = len(query.split())
        technical_terms = [
            'algorithm', 'optimization', 'architecture', 'parameter',
            'gradient', 'derivative', 'matrix', 'tensor'
        ]
        tech_count = sum(1 for term in technical_terms if term in query_lower)

        if word_count > 15 or tech_count >= 2:
            return 'Advanced', 0.65
        elif word_count < 8 and tech_count == 0:
            return 'Beginner', 0.65
        else:
            return 'Intermediate', 0.70

    def analyze_query(self, query: str) -> Dict:
        """
        Complete query analysis pipeline

        Returns:
        {
            'intent': str,
            'topic': str,
            'difficulty_level': str,
            'confidence_scores': {
                'intent': float,
                'topic': float,
                'difficulty': float
            },
            'embedding': np.ndarray (384-dim)
        }
        """
        # Generate embedding (expensive operation - do once)
        embedding = self.get_query_embedding(query)

        # Parallel classification
        intent, intent_conf = self.classify_intent(query, embedding)
        topic, topic_conf = self.extract_topic(query, embedding)
        difficulty, diff_conf = self.assess_difficulty(query, topic)

        return {
            'intent': intent,
            'topic': topic,
            'difficulty_level': difficulty,
            'confidence_scores': {
                'intent': intent_conf,
                'topic': topic_conf,
                'difficulty': diff_conf,
                'overall': (intent_conf + topic_conf + diff_conf) / 3
            },
            'embedding': embedding,
            'query': query
        }

# Example usage
if __name__ == "__main__":
    from configs.config import Config

    config = Config()
    qum = QueryUnderstandingModule(config)

    # Test queries
    test_queries = [
        "I don't understand backpropagation",
        "Show me an example of gradient descent",
        "I'm confused about transformers",
        "Summarize neural networks for me"
    ]

    print("\n🧪 Testing Query Understanding Module\n")
    for query in test_queries:
        result = qum.analyze_query(query)
        print(f"Query: {query}")
        print(f"Intent: {result['intent']} (conf: {result['confidence_scores']['intent']:.2f})")
        print(f"Topic: {result['topic']} (conf: {result['confidence_scores']['topic']:.2f})")
        print(f"Difficulty: {result['difficulty_level']} (conf: {result['confidence_scores']['difficulty']:.2f})")
        print(f"Overall Confidence: {result['confidence_scores']['overall']:.2f}")
        print(f"Embedding shape: {result['embedding'].shape}")
        print("-" * 60)
