#!/usr/bin/env python3
"""
DEPRECATED: This script is non-functional after removal of evaluate.py

The batch evaluation functionality has been deprecated along with the automated
evaluation system. This file is retained for historical reference only.

For the current evaluation approach, see samples.md.

To evaluate multiple models, run inference.py on each model for your test set
and manually compare the outputs.
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    logger.error("=" * 80)
    logger.error("DEPRECATED: batch_evaluate.py is no longer functional")
    logger.error("=" * 80)
    logger.error("")
    logger.error("The automated evaluation system has been replaced with a")
    logger.error("sample-based evaluation approach documented in samples.md.")
    logger.error("")
    logger.error("To evaluate multiple models:")
    logger.error("1. Run inference.py on each model for your test questions")
    logger.error("2. Manually compare outputs")
    logger.error("3. Count accuracy (correct/incorrect answers)")
    logger.error("4. Document observable differences")
    logger.error("")
    logger.error("See samples.md for the evaluation methodology and examples.")
    logger.error("=" * 80)
    sys.exit(1)


if __name__ == "__main__":
    main()
