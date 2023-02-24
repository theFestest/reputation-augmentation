#!/opt/homebrew/bin/python3.10

import itertools
import subprocess

# Base sim command
BASE_COMMAND = ["python3.10", "-m" "src.simulation"]

# $BASE_COMMAND --answering_population 100 --rep-c1 3 --rep-c2 1 --use-rep True


def main():
    answering_population = [100, 1000, 10000]
    rep_c1 = [3, 10, 100]
    rep_c2 = [1, 0.1, 0.01]
    use_rep = [True, False]
    components = [answering_population, rep_c1, rep_c2, use_rep]

    all_combinations = [p for p in itertools.product(*components)]

    for args in all_combinations:
        print("Running arg combination: %s", args)
        subprocess.run([*BASE_COMMAND,
                        "--answering_population", f"{args[0]}",
                        "--rep-c1", f"{args[1]}",
                        "--rep-c2", f"{args[2]}",
                        "--use-rep", f"{args[3]}",
                        # "--silence_logging",
                        "--save_directory", "sim_states/tests"])


if __name__ == "__main__":
    main()
