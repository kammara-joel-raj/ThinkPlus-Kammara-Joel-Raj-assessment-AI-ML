
import numpy as np
import torch
from typing import Dict, List, Tuple
import json

class AdaptiveLearningPathRecommender:
    """
    Adaptive Learning Path Recommendation using Reinforcement Learning

    Architecture:
    - State: Student query analysis (intent, topic, difficulty, embedding)
    - Action: Select next learning resource from knowledge graph
    - Reward: Estimated learning effectiveness
    - Policy: PPO-inspired adaptive selection

    Components:
    1. Knowledge Graph Navigation
    2. Student State Modeling
    3. Resource Sequencing (RL-based)
    4. Personalization Engine
    """

    def __init__(self, config):
        self.config = config
        self.knowledge_graph = config.KNOWLEDGE_GRAPH

        # Learning resource types with effectiveness scores
        self.resource_types = {
            'video': {'effectiveness': 0.85, 'time': 15, 'engagement': 0.9},
            'reading': {'effectiveness': 0.75, 'time': 10, 'engagement': 0.7},
            'practice': {'effectiveness': 0.90, 'time': 20, 'engagement': 0.95},
            'quiz': {'effectiveness': 0.80, 'time': 10, 'engagement': 0.85},
            'project': {'effectiveness': 0.95, 'time': 60, 'engagement': 0.98},
            'qa': {'effectiveness': 0.85, 'time': 5, 'engagement': 0.88}
        }

        # RL parameters (simplified PPO approach)
        self.gamma = config.RL_GAMMA
        self.exploration_rate = 0.15

    def _get_prerequisites(self, topic: str) -> List[str]:
        """Get prerequisites for a topic from knowledge graph"""
        if topic in self.knowledge_graph:
            return self.knowledge_graph[topic].get('prerequisites', [])
        return []

    def _get_subtopics(self, topic: str) -> List[str]:
        """Get subtopics from knowledge graph"""
        if topic in self.knowledge_graph:
            return self.knowledge_graph[topic].get('topics', [])
        return []

    def _calculate_reward(self, resource_type: str, student_state: Dict) -> float:
        """
        Calculate reward for selecting a resource
        Considers:
        - Resource effectiveness
        - Student intent alignment
        - Difficulty appropriateness
        """
        base_reward = self.resource_types[resource_type]['effectiveness']

        # Intent-resource alignment bonus
        intent = student_state['intent']
        alignment_bonus = 0.0

        if intent == 'Example' and resource_type in ['practice', 'project']:
            alignment_bonus = 0.15
        elif intent == 'Explanation' and resource_type in ['video', 'reading']:
            alignment_bonus = 0.15
        elif intent == 'Doubt clarification' and resource_type in ['qa', 'practice']:
            alignment_bonus = 0.15
        elif intent == 'Revision' and resource_type in ['reading', 'quiz']:
            alignment_bonus = 0.15

        # Difficulty appropriateness
        difficulty = student_state['difficulty_level']
        if difficulty == 'Beginner' and resource_type in ['video', 'reading']:
            alignment_bonus += 0.05
        elif difficulty == 'Advanced' and resource_type in ['project', 'practice']:
            alignment_bonus += 0.05

        return base_reward + alignment_bonus

    def _select_resource_type(self, student_state: Dict, 
                             position: int, total_steps: int) -> str:
        """
        RL-based resource selection using epsilon-greedy policy

        Position in path matters:
        - Start: Prerequisites, fundamentals
        - Middle: Core content, practice
        - End: Assessment, projects
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.exploration_rate:
            return np.random.choice(list(self.resource_types.keys()))

        # Exploitation: Calculate expected rewards
        resource_rewards = {}
        for res_type in self.resource_types.keys():
            reward = self._calculate_reward(res_type, student_state)

            # Position-based bonus
            position_ratio = position / total_steps
            if position_ratio < 0.3:  # Early stage
                if res_type in ['video', 'reading']:
                    reward += 0.1
            elif position_ratio > 0.7:  # Late stage
                if res_type in ['practice', 'quiz', 'project']:
                    reward += 0.1

            resource_rewards[res_type] = reward

        # Select best resource
        return max(resource_rewards, key=resource_rewards.get)

    def generate_learning_path(self, query_analysis: Dict) -> Dict:
        """
        Generate adaptive learning path based on query analysis

        Pipeline:
        1. Extract student state from query analysis
        2. Check prerequisites
        3. Generate resource sequence using RL policy
        4. Calculate path metrics

        Returns:
        {
            'path': List[Dict],
            'total_time': int,
            'estimated_success_rate': float,
            'num_resources': int,
            'personalization_score': float
        }
        """
        topic = query_analysis['topic']
        intent = query_analysis['intent']
        difficulty = query_analysis['difficulty_level']

        path = []
        total_time = 0

        # Step 1: Prerequisites (if beginner or explicit request)
        if difficulty == 'Beginner' or intent == 'Explanation':
            prerequisites = self._get_prerequisites(topic)
            if prerequisites:
                path.append({
                    'step': len(path) + 1,
                    'title': 'Prerequisites Review',
                    'description': f"Review: {', '.join(prerequisites[:2])}",
                    'type': 'reading',
                    'estimated_time': 10,
                    'difficulty': 'Beginner'
                })
                total_time += 10

        # Step 2: Core content based on intent
        if intent == 'Example':
            # Focus on practical examples
            path.append({
                'step': len(path) + 1,
                'title': f'{topic} - Worked Examples',
                'description': f'Practical examples with step-by-step solutions',
                'type': 'practice',
                'estimated_time': 20,
                'difficulty': difficulty
            })
            total_time += 20

        # Main content
        subtopics = self._get_subtopics(topic)
        resource_type = self._select_resource_type(
            query_analysis, 
            len(path), 
            5  # Expected total steps
        )

        path.append({
            'step': len(path) + 1,
            'title': f'Core Concepts: {topic}',
            'description': f"Deep dive into: {', '.join(subtopics[:3]) if subtopics else topic}",
            'type': resource_type,
            'estimated_time': self.resource_types[resource_type]['time'],
            'difficulty': difficulty
        })
        total_time += self.resource_types[resource_type]['time']

        # Step 3: Intent-specific resources
        if intent == 'Doubt clarification':
            path.append({
                'step': len(path) + 1,
                'title': 'Common Misconceptions',
                'description': f'Address frequent confusion points in {topic}',
                'type': 'qa',
                'estimated_time': 5,
                'difficulty': difficulty
            })
            total_time += 5

        # Step 4: Practice
        practice_type = self._select_resource_type(
            query_analysis,
            len(path),
            5
        )

        path.append({
            'step': len(path) + 1,
            'title': 'Practice Problems',
            'description': f'{difficulty}-level exercises for {topic}',
            'type': 'practice',
            'estimated_time': 20,
            'difficulty': difficulty
        })
        total_time += 20

        # Step 5: Advanced/Revision based on difficulty
        if difficulty == 'Advanced':
            path.append({
                'step': len(path) + 1,
                'title': 'Advanced Applications',
                'description': 'Real-world use cases and implementations',
                'type': 'project',
                'estimated_time': 60,
                'difficulty': 'Advanced'
            })
            total_time += 60

        if intent == 'Revision':
            path.append({
                'step': len(path) + 1,
                'title': 'Summary & Key Points',
                'description': 'Quick reference guide and cheat sheet',
                'type': 'reading',
                'estimated_time': 10,
                'difficulty': difficulty
            })
            total_time += 10

        # Calculate metrics
        avg_effectiveness = np.mean([
            self.resource_types[step['type']]['effectiveness']
            for step in path
        ])

        personalization_score = self._calculate_personalization(
            path, query_analysis
        )

        return {
            'path': path,
            'total_time': total_time,
            'estimated_success_rate': avg_effectiveness,
            'num_resources': len(path),
            'personalization_score': personalization_score,
            'topic': topic,
            'intent': intent,
            'difficulty': difficulty
        }

    def _calculate_personalization(self, path: List[Dict], 
                                   query_analysis: Dict) -> float:
        """
        Calculate how personalized the path is
        Higher score = more tailored to student needs
        """
        intent = query_analysis['intent']
        difficulty = query_analysis['difficulty_level']

        # Check intent alignment
        intent_specific_count = 0
        for step in path:
            if intent == 'Example' and step['type'] in ['practice', 'project']:
                intent_specific_count += 1
            elif intent == 'Explanation' and step['type'] in ['video', 'reading']:
                intent_specific_count += 1
            elif intent == 'Revision' and step['type'] in ['reading', 'quiz']:
                intent_specific_count += 1

        intent_score = intent_specific_count / len(path)

        # Check difficulty progression
        difficulty_appropriate = sum(
            1 for step in path if step['difficulty'] == difficulty
        ) / len(path)

        # Diversity score (avoid repetition)
        unique_types = len(set(step['type'] for step in path))
        diversity_score = unique_types / len(self.resource_types)

        personalization = (
            0.5 * intent_score + 
            0.3 * difficulty_appropriate + 
            0.2 * diversity_score
        )

        return min(personalization, 1.0)

# Example usage
if __name__ == "__main__":
    from configs.config import Config

    config = Config()
    recommender = AdaptiveLearningPathRecommender(config)

    # Simulate query analysis
    test_analysis = {
        'intent': 'Explanation',
        'topic': 'Neural Networks',
        'difficulty_level': 'Intermediate',
        'confidence_scores': {'overall': 0.85}
    }

    print("\n🧪 Testing Adaptive Learning Path Recommender\n")
    result = recommender.generate_learning_path(test_analysis)

    print(f"Topic: {result['topic']}")
    print(f"Intent: {result['intent']}")
    print(f"Difficulty: {result['difficulty']}\n")
    print(f"📊 Path Metrics:")
    print(f"  Total Time: {result['total_time']} minutes")
    print(f"  Resources: {result['num_resources']}")
    print(f"  Success Rate: {result['estimated_success_rate']:.1%}")
    print(f"  Personalization: {result['personalization_score']:.1%}\n")

    print("📚 Learning Path:")
    for step in result['path']:
        print(f"  {step['step']}. {step['title']}")
        print(f"     Type: {step['type']} | Time: {step['estimated_time']}min")
        print(f"     {step['description']}")
        print()
