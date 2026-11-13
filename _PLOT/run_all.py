"""
-------------------------------------------------------------------------------------------------
This code accompanies the paper titled "How LLM capability shapes human learning to collaborate in safety-critical decisions"
Author: Jiajie Shen (maintainer); other contributors listed in paper.
-------------------------------------------------------------------------------------------------
"""

from plot_human_ai_2 import main as plot_human_ai
from plot_accuracy_by_round import main as plot_accuracy


def main() -> None:
    plot_human_ai()
    plot_accuracy()


if __name__ == "__main__":
    main()
