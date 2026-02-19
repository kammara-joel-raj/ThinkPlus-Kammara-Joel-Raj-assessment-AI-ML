
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestQueryUnderstanding(unittest.TestCase):
    """Test cases for Query Understanding Module"""

    def setUp(self):
        from configs.config import Config
        self.config = Config()

    def test_intent_classes(self):
        """Test that intent classes are properly defined"""
        expected = ["Explanation", "Example", "Doubt clarification", "Revision"]
        self.assertEqual(self.config.INTENT_CLASSES, expected)

    def test_difficulty_levels(self):
        """Test that difficulty levels are properly defined"""
        expected = ["Beginner", "Intermediate", "Advanced"]
        self.assertEqual(self.config.DIFFICULTY_LEVELS, expected)

    def test_knowledge_graph_structure(self):
        """Test knowledge graph has required fields"""
        kg = self.config.KNOWLEDGE_GRAPH
        self.assertIn('Neural Networks', kg)

        nn_entry = kg['Neural Networks']
        self.assertIn('prerequisites', nn_entry)
        self.assertIn('topics', nn_entry)
        self.assertIn('difficulty', nn_entry)
        self.assertIn('estimated_time', nn_entry)

class TestSyntheticData(unittest.TestCase):
    """Test cases for synthetic data generation"""

    def test_data_file_exists(self):
        """Test that synthetic data file was created"""
        import os
        data_path = 'ai_teaching_assistant/data/synthetic_queries.json'
        self.assertTrue(os.path.exists(data_path))

    def test_data_structure(self):
        """Test that data has correct structure"""
        import json

        with open('ai_teaching_assistant/data/synthetic_queries.json', 'r') as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        # Check first item structure
        item = data[0]
        self.assertIn('query', item)
        self.assertIn('intent', item)
        self.assertIn('topic', item)
        self.assertIn('difficulty', item)

if __name__ == '__main__':
    unittest.main()
