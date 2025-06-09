# ⚠️ IMPORTANT: USE POWERSHELL ONLY

> **All scripts must be run in Windows PowerShell at `D:\Loop`.**
> Do NOT use Git Bash, MINGW64, or other terminals. Failure to do so will cause errors.

# 🧬 NeoEvolve: LLM + Evolutionary Algorithm System

**Autonomous Algorithm Discovery through Large Language Models and Evolutionary Computation**

NeoEvolve is an advanced AI system inspired by DeepMind's AlphaEvolve that combines the power of Large Language Models (LLMs) with evolutionary algorithms to autonomously discover, optimize, and evolve algorithms for complex computational problems.

## 🎯 Features

- **Multi-LLM Support**: Works with OpenAI GPT-4, Google Gemini, and other LLM providers
- **Evolutionary Framework**: Sophisticated population management, selection, mutation, and crossover
- **Safe Code Execution**: Sandboxed environment for secure algorithm evaluation
- **Comprehensive Evaluation**: Multi-dimensional scoring (correctness, efficiency, readability)
- **Task Management**: Built-in benchmark suite and custom task support
- **Real-time Dashboard**: Streamlit-based UI for monitoring evolution progress
- **Extensible Architecture**: Modular design for easy customization and extension

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key or Google Gemini API key

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/neoevolve/neo-evolve.git
cd neo-evolve
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up API keys:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
# OR
export GEMINI_API_KEY="your-gemini-api-key"
```

### Basic Usage

1. **Run a simple evolution:**
```bash
python run.py --task sorting --generations 20
```

2. **Start the web dashboard:**
```bash
python run.py --ui
```

3. **Run benchmark suite:**
```bash
python run.py --benchmark
```

## 📋 Available Tasks

NeoEvolve comes with several built-in algorithm discovery tasks:

- **Sorting Algorithms**: Discover efficient sorting methods
- **Matrix Multiplication**: Optimize matrix operations
- **Graph Traversal**: Evolve graph algorithms (BFS, DFS, etc.)
- **Optimization Problems**: Solve knapsack and similar challenges

## 🏗️ Architecture

```
neo_evolve/
├── agents/           # LLM integration and prompt engineering
├── evaluator/        # Code execution and performance evaluation
├── evolution/        # Population management and evolutionary operators
├── tasks/           # Task definitions and management
├── ui/              # Web dashboard interface
└── data/            # Results and metrics storage
```

### Core Components

1. **LLM Agent**: Handles code generation using various LLM providers
2. **Evolution Engine**: Orchestrates the evolutionary process
3. **Code Evaluator**: Safely executes and scores generated algorithms
4. **Task Manager**: Manages problem definitions and test cases
5. **Metrics Collector**: Tracks evolution progress and performance

## 🔧 Configuration

Customize NeoEvolve through `config.yaml`:

```yaml
# LLM Configuration
llm:
  default_provider: "openai"  # or "gemini"
  providers:
    openai:
      model: "gpt-4"
      temperature: 0.7
    gemini:
      model: "gemini-2.0-flash-exp"
      temperature: 0.7

# Evolution Parameters
evolution:
  population_size: 20
  generations: 50
  selection_rate: 0.3
  mutation_rate: 0.7
  crossover_rate: 0.5

# Evaluation Settings
evaluation:
  timeout_seconds: 30
  max_memory_mb: 512
  safety_checks: true
```

## 📊 Example Results

After running evolution on a sorting task:

```python
# Best discovered algorithm (Generation 15, Fitness: 0.94)
def sort_array(arr):
    if len(arr) <= 1:
        return arr[:]
    
    # Adaptive algorithm selection
    if len(arr) < 10:
        return insertion_sort(arr)
    elif is_nearly_sorted(arr):
        return tim_sort_variant(arr)
    else:
        return quick_sort_optimized(arr)
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test categories:

```bash
pytest tests/test_basic_functionality.py -v
```

## 📈 Monitoring Evolution

The dashboard provides real-time insights:

- **Fitness Evolution**: Track best, average, and worst scores over generations
- **Population Diversity**: Monitor genetic diversity to prevent premature convergence
- **Algorithm Gallery**: Browse top-performing discovered algorithms
- **Performance Metrics**: Detailed analysis of correctness, efficiency, and readability

## 🔬 Advanced Usage

### Custom Tasks

Create custom algorithm discovery tasks:

```python
from neo_evolve.tasks import TaskManager

task_manager = TaskManager()
custom_task = task_manager.create_custom_task(
    name="fibonacci_optimization",
    description="Optimize Fibonacci sequence calculation",
    test_cases=[
        {"input": [10], "expected": 55},
        {"input": [20], "expected": 6765},
        # ... more test cases
    ]
)
```

### Custom Mutation Strategies

Implement domain-specific mutations:

```python
from neo_evolve.evolution.mutation import MutationStrategy

class CustomMutation(MutationStrategy):
    def mutate(self, individual, **kwargs):
        # Your custom mutation logic
        return mutated_individual
```

### Integration with External Systems

```python
import asyncio
from neo_evolve.evolution import EvolutionEngine

async def integrate_with_external_system():
    config = load_your_config()
    engine = EvolutionEngine(config)
    
    best_algorithm = await engine.evolve(
        task_description="Your problem description",
        test_cases=your_test_cases
    )
    
    return best_algorithm
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by DeepMind's AlphaEvolve research
- Built on the shoulders of giants in evolutionary computation
- Thanks to the open-source AI community

## 📚 Citation

If you use NeoEvolve in your research, please cite:

```bibtex
@software{neoevolve2024,
  title={NeoEvolve: LLM + Evolutionary Algorithm System for Autonomous Algorithm Discovery},
  author={NeoEvolve Team},
  year={2024},
  url={https://github.com/neoevolve/neo-evolve}
}
```

## 🔗 Links

- [Documentation](https://neoevolve.readthedocs.io/)
- [Examples](examples/)
- [API Reference](docs/api/)
- [Community Discord](https://discord.gg/neoevolve)

---

**Built with ❤️ for the future of autonomous algorithm discovery**

## Troubleshooting Terminal Issues

- If you see errors about the terminal, working directory, or environment variables, make sure you are using Windows PowerShell and are in the `D:\Loop` directory.
- Do not use Git Bash or MINGW64 for any of the scripts in this project.
- If you see a warning or error at script startup, follow the instructions printed.

## 🚀 Docker Deployment (with Gemini)

You can run NeoEvolve in a containerized environment with Gemini LLM support:

1. **Build the Docker image:**
   ```bash
   docker build -t neoevolve .
   ```
2. **Run the container with your Gemini API key:**
   ```bash
   docker run -e GEMINI_API_KEY=your-gemini-api-key -p 8501:8501 neoevolve
   ```
3. **Access the web dashboard:**
   Open [http://localhost:8501](http://localhost:8501) in your browser.

You can override the default command to run other scripts, e.g.:
```bash
docker run -e GEMINI_API_KEY=your-gemini-api-key neoevolve python run.py --task sorting --generations 5
```
