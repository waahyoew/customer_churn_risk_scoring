"""
Main entry point for the Churn Prediction Pipeline.
Executes the pipeline and returns appropriate system exit codes for CI/CD environments.
"""

import sys
from pathlib import Path

# Add project root to the Python path to allow absolute imports from 'src'
sys.path.append(str(Path(__file__).resolve().parent))

from src.pipeline import run_pipeline
from src.logger import setup_logger

def main() -> None:
    """
    Main execution function.
    Catches uncaught pipeline exceptions and returns a non-zero exit code on failure.
    """
    logger = setup_logger('MainEntryPoint')
    
    try:
        run_pipeline()
        sys.exit(0)  # Success exit code
    except Exception as fatal_error:
        logger.critical(f"Pipeline execution aborted due to a fatal error: {fatal_error}")
        sys.exit(1)  # Failure exit code

if __name__ == '__main__':
    main()
