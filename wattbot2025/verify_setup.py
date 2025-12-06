from pathlib import Path
import sys


def check_setup():
    print(" Checking WattBot2025 Setup...")

    issues = []

    # Check directories
    dirs = ["data/raw", "data/chunks", "data/processed", "data/cache"]
    for d in dirs:
        if not Path(d).exists():
            issues.append(f"Missing directory: {d}")

    # Check files
    files = {
        "data/raw/metadata.csv": "Metadata file",
        "data/raw/train_QA.csv": "Training data",
        "data/raw/test_Q.csv": "Test questions",
        ".env": "API keys file",
        "configs/config.yaml": "Configuration"
    }

    for file, desc in files.items():
        if not Path(file).exists():
            issues.append(f"Missing {desc}: {file}")

    # Check for chunks
    chunks = list(Path("data/chunks").glob("*.json"))
    if not chunks:
        issues.append("No chunk files found. Run chunker.py")

    # Check Python packages
    packages = ["sentence_transformers", "anthropic", "openai", "yaml", "pandas"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            issues.append(f"Missing package: {pkg}")

    if issues:
        print("\n Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(" All checks passed! Ready to run.")
        return True


if __name__ == "__main__":
    if check_setup():
        print("\n Next step: Run 'python run.py' to start the pipeline")