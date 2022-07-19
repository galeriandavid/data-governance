import argparse
import logging
import sys
from pylint.lint import Run


logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(prog="LINT")

parser.add_argument(
    "-p",
    "--path",
    help="path to directory you want to run pylint",
    default=".",
    type=str,
)
parser.add_argument(
    "-t",
    "--threshold",
    help="Score thresold to fail pylint",
    default=9,
    type=float,
)

args = parser.parse_args()

results = Run(["--recursive=y", args.path], do_exit=False)

final_score = results.linter.stats.global_note

if final_score < args.threshold:
    message = f"Pylint failed | Score: {final_score} | Threshold: {args.threshold}"
    logging.error(message)
    raise Exception(message)

message = f"Pylint failed | Score: {final_score} | Threshold: {args.threshold}"
logging.info(message)
sys.exit(0)
