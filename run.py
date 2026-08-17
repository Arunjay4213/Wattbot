import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from src.vector_pipeline import VectorDBPipeline

def main():
    print("wattbot!!!!")

    # Initialize pipeline
    pipeline = VectorDBPipeline()

    # Build or load index
    print("\n Step 1: Building Vector Index...")
    pipeline.build_or_load_index()

    # Test on training data
    print("\n Step 2: Testing on Training Data...")
    pipeline.test_on_training_data()

    # Ask user whether to proceed
    response = input("\n🤔 Proceed with test set? (y/n): ")

    if response.lower() == 'y':
        print("\n Step 3: Processing Test Questions...")
        pipeline.process_test_questions()
        print("\n Complete! Check data/processed/submission.csv")
    else:
        print("\n Pipeline ready. Run process_test_questions() when ready.")


if __name__ == "__main__":
    main()