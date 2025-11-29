# AI Synthetic Dataset Generator

A powerful tool for generating **realistic, fully synthetic, privacy-safe datasets** for testing surveillance, access control, and monitoring systems in organizations. This project supports multiple AI models and can generate various types of security-related datasets.

## Features

- 🎯 **Multiple AI Model Support**: Works with Hugging Face models, Ollama (local), OpenAI GPT-5.1, Claude Opus 4.5, and Gemini 3.0
- 🔒 **Privacy-First**: Generates 100% synthetic data with no real names, companies, or sensitive attributes
- 📊 **Multiple Dataset Types**: Supports employee directories, visitor logs, access control event logs, CCTV summaries, and security incident logs
- ⚡ **GPU Acceleration**: Automatically detects and uses MPS (Apple Silicon), CUDA (NVIDIA), or CPU
- 🎨 **Customizable**: Define organization names, scenarios, schemas, date ranges, and row counts

## Supported Models

### Cloud APIs (Best Quality)
- **Claude Opus 4.5** - Recommended for structured data generation
- **GPT-5.1** - Strong instruction following
- **Gemini 3.0 Pro Preview** - Good for structured output

### Local Models
- **Ollama (gpt-oss:20b)** - Local inference, requires 16GB+ RAM
- **Hugging Face Models**:
  - Phi-3-mini-4k-instruct (currently configured)
  - Mistral-7B-Instruct (recommended alternative)
  - Qwen2.5-7B-Instruct (alternative)

## Requirements

- Python 3.12+
- Jupyter Notebook
- API Keys (optional, for cloud models):
  - OpenAI API key
  - Anthropic API key
  - Google API key (for Gemini)
  - Hugging Face token

### Python Dependencies

```bash
pip install torch accelerate transformers openai python-dotenv huggingface-hub requests ipython
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-synthetic-dataset-generator
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=sk-proj-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=...
   HF_TOKEN=hf_...
   ```

5. **For Ollama (optional, for local inference)**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama serve  # Start Ollama server
   ollama pull gpt-oss:20b  # Pull the model (requires 16GB+ RAM)
   ```

## Usage

1. **Open the Jupyter Notebook**
   ```bash
   jupyter notebook app.ipynb
   ```

2. **Run cells in order**:
   - Cell 1: Import dependencies
   - Cell 2: Load and validate API keys
   - Cell 3: Set model constants
   - Cell 4: Initialize API clients
   - Cell 5: Configure device (MPS/CUDA/CPU)
   - Cell 6: Define prompts (system and user prompts)

3. **Choose a model to generate data**:
   - **Cell 8**: Hugging Face Phi-3 model (local)
   - **Cell 12**: Ollama gpt-oss:20b (local, requires Ollama)
   - **Cell 13**: OpenAI GPT-5.1 (cloud)
   - **Cell 14**: Claude Opus 4.5 (cloud, recommended)
   - **Cell 15**: Gemini 3.0 (cloud)

4. **Generated datasets** are saved as CSV files:
   - `phi3_generated_dataset.csv`
   - `gpt_oss_generated_dataset.csv`
   - `gpt_5_1_generated_dataset.csv`
   - `claude_generated_dataset.csv`
   - `gemini_3_generated_dataset.csv`

## Customizing Datasets

Edit the `user_prompt` in Cell 6 to customize:
- Organization name
- Time window/date ranges
- Dataset type (access control logs, employee directories, etc.)
- Number of rows
- Column schema
- Specific patterns or constraints

## Privacy & Ethical Compliance

This tool is designed with privacy and ethics in mind:

✅ **What it generates:**
- Fictional employee/visitor IDs (EMP001, VIS001, etc.)
- Job roles, departments, access levels
- Timestamps, room/zone identifiers
- Access methods as labels only (CARD, PIN, FINGERPRINT, FACE_ID)

❌ **What it never generates:**
- Real names, companies, or public figures
- Race, ethnicity, religion, political views
- Medical information or sexual orientation
- Biometric templates, vectors, or measurements
- Any data that could identify real individuals

## Example Use Cases

- **Access Control Testing**: Generate realistic access control event logs for system testing
- **Security System Evaluation**: Create synthetic datasets for testing surveillance systems
- **Analytics Development**: Build datasets for developing analytics without real data
- **Training Data**: Generate training datasets for ML models in security contexts

## Troubleshooting

### Kernel crashes when loading large models
- Use smaller models (Phi-3-mini) or cloud APIs
- Ensure sufficient RAM (16GB+ for large local models)
- Try quantization: `load_in_8bit=True` in model loading

### Chat template errors
- Some models (like GPT-2) don't support chat format
- Use instruction-tuned models (Phi-3, Mistral, Qwen) or convert messages to plain text

### Connection errors
- Verify API keys are set correctly in `.env`
- Check network connectivity for cloud APIs
- For Ollama, ensure `ollama serve` is running

## Model Recommendations

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| Best quality | Claude Opus 4.5 | Excellent structured output |
| Local inference | Ollama gpt-oss:20b | Good balance of quality and privacy |
| Fast local | Phi-3-mini | Small, efficient, good instructions |
| Cost-effective | Gemini 3.0 | Good quality at lower cost |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Hugging Face for model hosting and transformers library
- OpenAI, Anthropic, and Google for API access
- Ollama for local model inference
