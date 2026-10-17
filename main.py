import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

from lib.graph import BrowserAgentGraph


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Run the LangGraph browser agent.")
    parser.add_argument("instruction", help="What the browser agent should accomplish.")
    parser.add_argument("--cloud", action="store_true", help="Use Browser Use cloud instead of local Chrome.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Maximum local vision decisions (default: 60).",
    )
    args = parser.parse_args()

    response = BrowserAgentGraph().invoke(
        args.instruction,
        is_cloud=args.cloud,
        max_steps=args.max_steps,
    )
    print(response)


if __name__ == "__main__":
    main()
