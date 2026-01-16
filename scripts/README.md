# Scripts

This directory contains utility scripts for model conversion, deployment, and maintenance.

## 📜 Available Scripts

### 🔄 Model Conversion
- **[convert_pyannote.py](convert_pyannote.py)** - Convert pyannote models to MLX format
  ```bash
  python scripts/convert_pyannote.py
  ```

### ☁️ Deployment
- **[upload_to_hf.py](upload_to_hf.py)** - Upload models to Hugging Face Hub
  ```bash
  python scripts/upload_to_hf.py
  ```

### 🛠️ Setup
- **[setup.sh](setup.sh)** - Environment setup script
  ```bash
  bash scripts/setup.sh
  ```

## Usage Examples

### Convert a Model
```bash
cd /path/to/speaker-diarization-community-1-mlx
python scripts/convert_pyannote.py
```

### Upload to Hugging Face
```bash
# Set your token in .env file first
echo "HF_TOKEN=hf_your_token_here" > .env
python scripts/upload_to_hf.py
```

### Setup Environment
```bash
bash scripts/setup.sh
```

## Adding New Scripts

When adding new utility scripts:
1. Place them in this `scripts/` directory
2. Make them executable if they're shell scripts: `chmod +x script.sh`
3. Update this README.md with documentation
4. Include usage examples and requirements