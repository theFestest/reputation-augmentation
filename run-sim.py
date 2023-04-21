#!/opt/homebrew/bin/python3.10

import sys
import itertools
import subprocess
import logging

# Make sure logging gets sent to the screen
logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # Change to warning for pending implementations

logger = logging.getLogger(__name__)

# Base sim command
BASE_COMMAND = ["python3.10", "-m" "src.simulation"]

# $BASE_COMMAND --answering_population 100 --rep-c1 3 --rep-c2 1 --use-rep True


def main():
    answering_population = [100]  # [100, 1000]  # , 10000]  # skip 10k for now to run faster.
    rep_c1 = [10]  # [3, 10]  # , 30]
    rep_c2 = [0.1]  # , 0.01]  # , 0.001]  # 1 is way too fast. with 5k q's 0.001 is so slow its almost identical to no rep
    reputation_affinity = [100]  # [1, 10, 100]
    use_rep = [True, False]
    components = [answering_population, rep_c1, rep_c2, reputation_affinity, use_rep]

    # Note that we can skip reputation False variations of rep parameters.
    # - This is bc permutations with varied reputation parameters but rep false doesn't use the rep parameters.
    # - The whole test set only takes a few minutes to run, so we leave it enabled for simplicity for now.
    #   It is easier to quickly compare if there are reputation true/false results next to each other.
    all_combinations = [p for p in itertools.product(*components)]

    save_dir = "sim_states/fixed_boost"

    subprocess.run(["mkdir", "-p", save_dir])

    for args in all_combinations:

        if args[4] is True:
            logger.info("Running arg combination: %s", args)
            subprocess.run([*BASE_COMMAND,
                            "--answering_population", f"{args[0]}",
                            "--rep-c1", f"{args[1]}",
                            "--rep-c2", f"{args[2]}",
                            "--use-rep",
                            "--confidence-threshold", "15",
                            "--fixed-threshold",
                            "--experience-boost", "0.4",
                            "--reputation-affinity", f"{args[3]}",
                            "--questions_per_epoch", "5000",
                            # "--silence_logging",
                            "--save_directory", save_dir])
        else:
            logger.info("Running arg combination: %s", args)
            subprocess.run([*BASE_COMMAND,
                            "--answering_population", f"{args[0]}",
                            "--rep-c1", f"{args[1]}",
                            "--rep-c2", f"{args[2]}",
                            "--confidence-threshold", "15",
                            "--fixed-threshold",
                            "--experience-boost", "0.4",
                            "--reputation-affinity", f"{args[3]}",
                            "--questions_per_epoch", "5000",
                            # "--silence_logging",
                            "--save_directory", save_dir])

    logger.info("Finished running %s arg combinations", len(all_combinations))

    # TODO: Consider add post-run automatic processing to get some summary stats?


if __name__ == "__main__":
    main()
