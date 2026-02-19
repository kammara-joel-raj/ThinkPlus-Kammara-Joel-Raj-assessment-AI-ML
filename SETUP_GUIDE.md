# Setup and Installation Guide

## Quick Start (5 minutes)

### Prerequisites
- Python 3.9 or higher
- pip package manager
- 2GB free disk space
- 4GB RAM minimum (8GB recommended)
- Internet connection for downloading models

### Step 1: Clone Repository
```bash
git clone <your-repository-url>
cd ai_teaching_assistant
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# This will install:
# - PyTorch (CPU version)
# - Transformers
# - Sentence-Transformers
# - Scikit-learn
# - And other dependencies
```

**Note**: First run will download SBERT model (~80MB). This happens automatically.

### Step 4: Verify Installation
```bash
# Run tests
python -m pytest ai_teaching_assistant/tests/

# Or run individual module tests
python ai_teaching_assistant/models/query_understanding.py
```

### Step 5: Run Demo
```bash
# Run demo mode
python ai_teaching_assistant/main.py --mode demo

# Or run interactive mode
python ai_teaching_assistant/main.py --mode interactive
```

---

## Detailed Installation

### Option 1: Using pip (Recommended)

```bash
# Clone repository
git clone <repo-url>
cd thinkpluss

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n teaching_assistant python=3.9
conda activate teaching_assistant

# Install dependencies
pip install -r requirements.txt
```


## Usage Examples

### Demo Mode (Recommended for first-time users)
```bash
python main.py --mode demo
```
Runs 5 example queries to showcase system capabilities.

### Interactive Mode
```bash
python main.py --mode interactive
```
Interactive CLI where you can enter your own queries.

### Single Query
```bash
python main.py --query "I don't understand backpropagation"
```
Process a single query and display results.

### Python API
```python
from main import AITeachingAssistant

# Initialize system
assistant = AITeachingAssistant()

# Process query
result = assistant.process_query("Show me examples of neural networks")

# Access results
print(result['analysis']['intent'])  # "Example"
print(result['analysis']['topic'])   # "Neural Networks"
print(result['learning_path'])       # List of learning steps
```

---

## Troubleshooting

### Issue: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'sentence_transformers'
```
**Solution**: 
```bash
pip install sentence-transformers
```

### Issue: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Use CPU version or reduce batch size
```python
# Force CPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Issue: Slow first run
**Cause**: SBERT model downloading (~80MB)  
**Solution**: Wait for initial download. Subsequent runs will be fast.

### Issue: Import errors
```
ImportError: cannot import name 'Config' from 'configs.config'
```
**Solution**: Ensure __init__.py files exist:
```bash
touch ai_teaching_assistant/__init__.py
touch ai_teaching_assistant/configs/__init__.py
touch ai_teaching_assistant/models/__init__.py
```

---

## Testing

### Run All Tests
```bash
python -m pytest ai_teaching_assistant/tests/ -v
```

### Run Specific Test
```bash
python -m pytest ai_teaching_assistant/tests/test_basic.py -v
```

### Test Individual Modules
```bash
# Test query understanding
python ai_teaching_assistant/models/query_understanding.py

# Test learning path recommender
python ai_teaching_assistant/models/learning_path_recommender.py

# Test data generator
python ai_teaching_assistant/utils/data_generator.py
```

---

## Configuration

Edit `ai_teaching_assistant/configs/config.py` to customize:

```python
# Change SBERT model
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight
# or
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"  # More accurate, slower

# Add topics
TOPIC_CATEGORIES = [
    "Neural Networks",
    "Machine Learning",
    # Add your topics here
]

# Adjust RL parameters
RL_GAMMA = 0.99
RL_LR = 3e-4
```

---

## Performance Optimization

### 1. Use GPU (if available)
```python
# Automatic GPU detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 2. Batch Processing
```python
# Process multiple queries at once
queries = ["query1", "query2", "query3"]
embeddings = embedder.encode(queries, batch_size=32)
```

### 3. Model Caching
Models are automatically cached after first load. No additional setup needed.

---

## Dataset Regeneration

To regenerate synthetic dataset with different parameters:

```python
from ai_teaching_assistant.utils.data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator()
dataset = generator.save_dataset(
    'ai_teaching_assistant/data/synthetic_queries.json',
    samples_per_category=500  # Increase for more data
)
```

---

## Deployment

### Local Deployment
```bash
# Run as a service
python ai_teaching_assistant/main.py --mode interactive
```

### API Deployment (Future)
```python
from flask import Flask, request, jsonify
from ai_teaching_assistant.main import AITeachingAssistant

app = Flask(__name__)
assistant = AITeachingAssistant()

@app.route('/analyze', methods=['POST'])
def analyze():
    query = request.json['query']
    result = assistant.process_query(query)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## System Requirements

### Minimum Requirements
- CPU: 2 cores
- RAM: 4 GB
- Storage: 2 GB
- OS: Linux, macOS, Windows

### Recommended Requirements
- CPU: 4+ cores
- RAM: 8 GB
- Storage: 5 GB
- GPU: Optional (NVIDIA with CUDA for faster inference)
- OS: Linux or macOS

### Performance Expectations
- **CPU**: ~200ms per query
- **GPU**: ~20ms per query
- **Throughput**: 300+ queries/minute (CPU)

---

## Additional Resources

- **Documentation**: See TECHNICAL_DOCUMENTATION.md
- **Examples**: See examples/ directory (if available)
- **Issues**: Report on GitHub Issues
- **Discussions**: GitHub Discussions

---

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review error messages carefully
3. Check GitHub Issues for similar problems
4. Open a new issue with:
   - Error message
   - Python version
   - Operating system
   - Steps to reproduce

---

## Next Steps

After successful setup:

1. ✅ Run demo mode to see examples
2. ✅ Try interactive mode with your own queries
3. ✅ Explore the code in `models/` directory
4. ✅ Read TECHNICAL_DOCUMENTATION.md for details
5. ✅ Customize configuration for your use case

