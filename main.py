
#!/usr/bin/env python3
"""
AI-Powered Teaching Assistant
Main Application Entry Point

This system demonstrates:
1. Student Query Understanding (NLP + ML)
2. Adaptive Learning Path Recommendation (RL-based)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import torch
from configs.config import Config
from models.query_understanding import QueryUnderstandingModule
from models.learning_path_recommender import AdaptiveLearningPathRecommender

class AITeachingAssistant:
    """
    Main Teaching Assistant System

    Architecture:
    - Module 1: Query Understanding (SBERT + Transformers)
    - Module 2: Learning Path Recommendation (RL + Knowledge Graph)

    Clear separation: ML for classification, LLM for refinement
    """

    def __init__(self):
        print("🚀 Initializing AI Teaching Assistant...\n")

        self.config = Config()

        # Initialize modules
        print("📥 Loading Query Understanding Module...")
        self.query_module = QueryUnderstandingModule(self.config)

        print("📥 Loading Learning Path Recommender...")
        self.recommender = AdaptiveLearningPathRecommender(self.config)

        print("\n✅ System Ready!\n")

    def process_query(self, query: str) -> dict:
        """
        Complete pipeline for processing student query

        Steps:
        1. Analyze query (intent, topic, difficulty)
        2. Generate adaptive learning path
        3. Return comprehensive response
        """
        print(f"📝 Processing query: '{query}'\n")

        # Step 1: Query Understanding
        print("🔍 Stage 1: Query Analysis...")
        analysis = self.query_module.analyze_query(query)

        print(f"  ✓ Intent: {analysis['intent']} "
              f"(confidence: {analysis['confidence_scores']['intent']:.2%})")
        print(f"  ✓ Topic: {analysis['topic']} "
              f"(confidence: {analysis['confidence_scores']['topic']:.2%})")
        print(f"  ✓ Difficulty: {analysis['difficulty_level']} "
              f"(confidence: {analysis['confidence_scores']['difficulty']:.2%})")
        print(f"  ✓ Overall Confidence: {analysis['confidence_scores']['overall']:.2%}\n")

        # Step 2: Learning Path Generation
        print("🧠 Stage 2: Generating Adaptive Learning Path...")
        learning_path = self.recommender.generate_learning_path(analysis)

        print(f"  ✓ Generated {learning_path['num_resources']} personalized resources")
        print(f"  ✓ Estimated time: {learning_path['total_time']} minutes")
        print(f"  ✓ Success rate: {learning_path['estimated_success_rate']:.1%}")
        print(f"  ✓ Personalization score: {learning_path['personalization_score']:.1%}\n")

        # Combine results
        result = {
            'query': query,
            'analysis': {
                'intent': analysis['intent'],
                'topic': analysis['topic'],
                'difficulty_level': analysis['difficulty_level'],
                'confidence_scores': analysis['confidence_scores']
            },
            'learning_path': learning_path['path'],
            'metrics': {
                'total_time': learning_path['total_time'],
                'num_resources': learning_path['num_resources'],
                'estimated_success_rate': learning_path['estimated_success_rate'],
                'personalization_score': learning_path['personalization_score']
            }
        }

        return result

    def display_results(self, result: dict):
        """Pretty print results"""
        print("=" * 70)
        print("📊 ANALYSIS RESULTS")
        print("=" * 70)

        print(f"\n🎯 Query Classification:")
        print(f"  Intent:     {result['analysis']['intent']}")
        print(f"  Topic:      {result['analysis']['topic']}")
        print(f"  Difficulty: {result['analysis']['difficulty_level']}")
        print(f"  Confidence: {result['analysis']['confidence_scores']['overall']:.1%}")

        print(f"\n📚 Adaptive Learning Path ({result['metrics']['num_resources']} steps):")
        print(f"  Total Duration: {result['metrics']['total_time']} minutes")
        print(f"  Success Rate:   {result['metrics']['estimated_success_rate']:.1%}")
        print(f"  Personalized:   {result['metrics']['personalization_score']:.1%}")

        print(f"\n📝 Learning Steps:")
        for i, step in enumerate(result['learning_path'], 1):
            print(f"\n  Step {i}: {step['title']}")
            print(f"  └─ Type: {step['type']} | Duration: {step['estimated_time']}min")
            print(f"  └─ {step['description']}")

        print("\n" + "=" * 70)

def interactive_mode():
    """Interactive CLI mode"""
    assistant = AITeachingAssistant()

    print("\n" + "="*70)
    print("🎓 AI TEACHING ASSISTANT - INTERACTIVE MODE")
    print("="*70)
    print("\nType 'quit' or 'exit' to stop\n")

    while True:
        try:
            query = input("Student Query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not query:
                continue

            print()
            result = assistant.process_query(query)
            print()
            assistant.display_results(result)
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")

def demo_mode():
    """Demo mode with example queries"""
    assistant = AITeachingAssistant()

    demo_queries = [
        "I don't understand backpropagation",
        "Show me examples of neural networks",
        "I'm stuck on gradient descent",
        "Summarize transformers for me",
        "What are advanced CNN architectures?"
    ]

    print("\n" + "="*70)
    print("🎓 AI TEACHING ASSISTANT - DEMO MODE")
    print("="*70)
    print(f"\nRunning {len(demo_queries)} example queries...\n")

    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*70}")
        print(f"DEMO {i}/{len(demo_queries)}")
        print(f"{'='*70}\n")

        result = assistant.process_query(query)
        assistant.display_results(result)

        if i < len(demo_queries):
            input("\nPress Enter for next demo...")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='AI-Powered Teaching Assistant'
    )
    parser.add_argument(
        '--mode',
        choices=['demo', 'interactive'],
        default='demo',
        help='Run mode: demo or interactive'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to process'
    )

    args = parser.parse_args()

    if args.query:
        # Single query mode
        assistant = AITeachingAssistant()
        result = assistant.process_query(args.query)
        assistant.display_results(result)
    elif args.mode == 'interactive':
        interactive_mode()
    else:
        demo_mode()
