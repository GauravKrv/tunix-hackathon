#!/usr/bin/env python3
"""
DEPRECATED: This script is non-functional after removal of evaluate.py

This visualization script was designed to generate charts from the automated
evaluation results, including reasoning quality scores and coherence metrics.

Since those metrics have been removed (see EVALUATION_CHANGELOG.md for rationale),
this visualization script is no longer functional.

For the current evaluation approach, see samples.md, which focuses on concrete
sample outputs rather than computed quality metrics.

This file is retained for historical reference only.
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
    logger.error("DEPRECATED: visualize_results.py is no longer functional")
    logger.error("=" * 80)
    logger.error("")
    logger.error("This script was designed to visualize reasoning quality scores")
    logger.error("and coherence metrics from the automated evaluation system.")
    logger.error("")
    logger.error("Those metrics have been removed because they were:")
    logger.error("- Not directly interpretable")
    logger.error("- Not defensible (based on heuristics)")
    logger.error("- Potentially misleading")
    logger.error("")
    logger.error("The new evaluation approach uses concrete sample outputs")
    logger.error("documented in samples.md with only accuracy as a quantitative metric.")
    logger.error("")
    logger.error("If you need visualizations:")
    logger.error("1. Count correct answers for accuracy across test sets")
    logger.error("2. Create simple bar charts of accuracy by category/difficulty")
    logger.error("3. Avoid derived quality scores")
    logger.error("")
    logger.error("See samples.md for the current evaluation methodology.")
    logger.error("=" * 80)
    sys.exit(1)


if __name__ == "__main__":
    main()
