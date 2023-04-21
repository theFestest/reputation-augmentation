
import argparse
import json
import sys
import logging
import datetime
import random
import copy
import math
# import numpy as np

from .players import AnsweringEntity, QuestionPool

# Make sure logging gets sent to the screen
logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)  # Change to warning for pending implementations

logger = logging.getLogger(__name__)

# Possible improvements:
#   - Include timing info for subtrails to inform which combos do lots more work?
#   - plot live results of counts required to hit threshold? How to best analyze? Extra scripts?
# Lower priority:
#   - implement state loading
#   - integrate literal time/place based context (this analogizes to categories, so perhaps not. all are categorical)
#   - implement program to load logged stats for analysis

# Data files derived from: https://en.wikipedia.org/wiki/Category:Main_topic_classifications


# Run with: python3.10 -m src.simulation, or similar
def main():
    # Configure the simulation with arg parser for a default simulation run or given params.
    argparser = argparse.ArgumentParser()

    random_seed = datetime.datetime.now().isoformat()
    random.seed(a=random_seed)

    # Load context from full dataset in its own file
    # Note that this file has 2 levels of heirachical knowledge categories.
    #   - Greater than 2 levels of nesting would be increasingly unlikely to select and a much heavier data set to use.
    #   - Wikipedia has further nested categories, but for simplicity we opt for 2 levels here.
    #   - This simulation also treats nested categories as if they're all the same specificity.
    #   - Currently reputation operates cross categories if their secondary domains are the same.
    #   That is, secondary domain repuation is able to influence across primary level domains.
    # We also note that a practical implementation may wish to use further nesting or more disjoint categories.
    data_file = "data/context_data.json"
    # data_file = "data/simple.json"
    with open(data_file) as f:
        data_set = json.load(f)
    context_set = data_set

    # Reputation model
    argparser.add_argument("--answering_population", help="Count of answering parties", type=int, required=True)
    argparser.add_argument("--reputation-affinity", help="", type=float, default=10)
    argparser.add_argument("--rep-c1", help="Reputation growth limit c1", type=float, required=True)
    argparser.add_argument("--rep-c2", help="Reputation growth rate c2", type=float, required=True)
    argparser.add_argument("--use-rep", help="Use reputation weighting", default=False, action="store_true")
    argparser.add_argument("--fixed-threshold", help="Threshold as counts not rep", default=False, action="store_true")
    # Underlying assumptions
    argparser.add_argument("--experience_domains", help="Count of experience domains", type=int, default=3)
    argparser.add_argument("--bassline_contention", help="Strongly to favor T or F: (0.5, 1]", type=float, default=0.7)
    argparser.add_argument("--confidence-threshold", help="How many parties do we use to vote", type=float, default=31)
    argparser.add_argument("--experience-boost", help="Domain knowledge helpfulness: (0, 0.5]", type=float, default=0.1)
    argparser.add_argument("--secondary_context_count", help="Count of secondary context domains", type=int, default=2)
    # SImulation configuration
    argparser.add_argument("--questions_per_epoch", help="Count of questions per epoch", type=int, default=2000)
    argparser.add_argument("--epochs", help="Count of epochs to run", type=int, default=1)
    argparser.add_argument("--epochs_per_save", help="Count of epochs between saves", type=int, default=1)
    argparser.add_argument("--save_directory", help="Location to save outputs", type=str, default="sim_states")
    argparser.add_argument("--silence_logging", help="Run the sim faster by turning off logging", action="store_true")

    # Parse inputs
    args = argparser.parse_args()

    # PARAMETER
    answering_population_count = args.answering_population  # 100
    # PARAMETER: How strongly do we favor identities with reputation in a questions context
    reputation_affinity = args.reputation_affinity
    # PARAMETER
    rep_c1 = args.rep_c1
    # PARAMETER
    rep_c2 = args.rep_c2
    # PARAMETER
    use_reputation: bool = args.use_rep
    # PARAMETER
    fixed_threshold = args.fixed_threshold
    # PARAMETER
    experience_domains = args.experience_domains  # 3
    # PARAMETER
    bassline_contention_center = args.bassline_contention
    # PARAMETER
    confidence_threshold = args.confidence_threshold
    # PARAMETER
    experience_boost = args.experience_boost
    # PARAMETER:
    secondary_context_count = args.secondary_context_count
    # PARAMETER
    questions_per_epoch = args.questions_per_epoch  # 2000  # 4000

    # Config
    save_directory: str = args.save_directory
    # Config
    epochs = args.epochs
    # Config
    epochs_per_save = args.epochs_per_save

    if args.silence_logging:
        # NOTE: only applies for this file, not globally.
        logger.setLevel(logging.WARN)

    # Initialize statistics
    # - Overall stats
    total_questions = 0
    total_aborted = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    # - Per question stats
    incorrectly_resolved = 0
    indeterminate_resolution = 0
    # majority_reputation = []
    # minority_reputation = []

    # Initialize question pool from context_set
    question_pool = QuestionPool(context_set=context_set)

    answering_parties = []
    # Initialize the desired number of answering parties
    for _ in range(answering_population_count):
        answering_parties.append(
            AnsweringEntity(
                context_set,
                rep_c1,
                rep_c2,
                experience_domains,
                exp_boost=experience_boost
            )
        )

    # Start running epochs
    for epoch_number in range(epochs):
        logger.info("Running epoch #%s", epoch_number)
        # Process the desired number of questions
        for question_number in range(questions_per_epoch):
            logger.info("Running question #%s in epoch #%s", question_number, epoch_number)
            # Generate a question from context
            this_question: QuestionPool.Question = question_pool.generate_question(secondary_context_count,
                                                                                   confidence_threshold,
                                                                                   bassline_contention_center)
            total_questions += 1
            #   - Assign the question, q, a bassline contention. [parameter]
            #   - Assign the question a secret “true” outcome for analysis purposes.
            #   - Assign a threshold confidence [parameter]
            logger.info("Generated question with domains %s", this_question)

            # Select who will be voting
            #   - based on required confidence threshold and current reputation
            #   - select psuedo randomly weighted by established reputation
            available_reputation = 0
            remaining_voters = copy.copy(answering_parties)
            participating_voters: list[AnsweringEntity] = []
            # Note: we will interpret this confidence threshold as a minimum common one.
            #   - A given user may opt to increase important questions at will.
            #   - EXPERIMENT: Try to evaluate archievable tolerances once high rep is established?
            if use_reputation:
                # TODO: speed up this very heavy computation? (relative to the rest) (particularly for 1k / 10k voters)
                current_rep = [voter.calculate_confidence(this_question.all_context) for voter in answering_parties]
                # PARAMETER: the sensitivity of this to reputation can vary
                voter_affinity = [math.floor(i*reputation_affinity) for i in current_rep]
            else:
                # Default to constant counts if no reputation (should be same as reputation_affinity = 1)
                voter_affinity = [1 for _ in range(len(answering_parties))]
            while (available_reputation < this_question.req_confidence_theshold):
                if len(remaining_voters) > 0:
                    # Selection can favor high reputation instead of being uniform via "reputation_affinity"
                    #   - this will be important for selecting the knowledgeable individuals
                    #   - note that this favoritism may want to be stronger with larger entity counts!
                    #   - maybe count=[floor(entity.rep/0.2) for entity in remaining_voters], so high rep occurs more
                    #   - use random.sample(iterable, COUNTS, k) to adjust distribution
                    selected_participant: list = random.sample(remaining_voters, counts=voter_affinity, k=1)
                    selected_participant: AnsweringEntity = selected_participant[0]
                    # selected_participant: AnsweringEntity = random.choice(remaining_voters)
                    # Find the list position of the selected voter
                    voter_pos = remaining_voters.index(selected_participant)
                    # Remove selected voter from remaining selection
                    remaining_voters.remove(selected_participant)
                    # Remove selected voter's affinity from affinity list
                    voter_affinity.pop(voter_pos)
                    if use_reputation and not fixed_threshold:
                        available_reputation += selected_participant.calculate_confidence(this_question.all_context)
                    else:  # if fixed_threshold or not use_reputation:
                        # Contributions are counts with a fixed threshodl or reputation disabled
                        available_reputation += 1
                    participating_voters.append(selected_participant)
                else:
                    # We record aborting this question for statistics below.
                    logger.info("Out of voters! Cannot reach threshold to evaluate this question.")
                    break  # Don't loop forever, just give up on this question.

            # If we used everyone and can't meet the threshold, record this and abort.
            #   Note: it may not be immediately obvious who "everyone" is in a decentralized voting game.
            #   This cutoff is an approximation for a time bound or a recency heuristic.
            if len(remaining_voters) == 0 and available_reputation < this_question.req_confidence_theshold:
                # Question aborted
                # indeterminate_resolution += 1
                #   (Do we consider an aborted question indeterminate? I say no, bc it wouldn't be accepted)
                this_question.indeterminate_resolution = True
                this_question.aborted = True
                this_question.parties_used = None
                total_aborted += 1
                continue  # Aborting this question, so go around.
            else:
                this_question.parties_used = len(participating_voters)
                # participants_utilized.append(len(participating_voters))
            # Record status of how many parties it took
            logger.info("Met confidence threshold with %s voters", this_question.parties_used)

            # Allow selected entities to vote; collect votes
            #   - Each entity has a probability c to vote the “true” outcome [parameter]
            #       Key: we suppose some entities naturally have a higher probability
            #       of correctness (due to their knowledge of the context) and we hope
            #       reputation can be a tool to allow long term system behavior to favor
            #       this knowledge.
            #   - Each entity has a particular stake contributed when voting [parameter; these may just be uniform]
            # Implemention of the resolution algorithm
            collected_votes: list[tuple[bool, float, float]] = []  # (vote, reputation, stake)
            for voter in participating_voters:
                # Collect tuples of vote, reputation, and stake (uniform for now)
                collected_votes.append(
                    voter.vote(
                        this_question.all_context,
                        this_question.contention,
                        this_question.true_outcome,
                        use_reputation
                        )
                    )

            cumulative_true_votes = 0
            cumulative_false_votes = 0
            resolved_outcome = None
            for vote in collected_votes:
                # NOTE: We can consider adding superlinearity with stake (only relevant if non-uniform voting stake)
                if use_reputation:
                    # Voting with reputation and stake weighting
                    if vote[0] is True:
                        cumulative_true_votes += vote[1]*vote[2]  # Add: f(i) = reputation*stake for each vote
                    else:  # Vote is Flase
                        cumulative_false_votes += vote[1]*vote[2]  # Add: reputation*stake for each vote
                else:
                    # Only voting with stake (i.e. 1 to 1)
                    if vote[0] is True:
                        cumulative_true_votes += vote[2]  # Add: f(i) = stake for each vote
                    else:  # Vote is Flase
                        cumulative_false_votes += vote[2]  # Add: stake for each vote
            # Compute the _resolved outcome_ based on votes, and
            #  - Utilize reputation weights and stakes to resolve the outcome
            if (cumulative_true_votes > cumulative_false_votes):
                resolved_outcome = True
            elif (cumulative_true_votes < cumulative_false_votes):
                resolved_outcome = False
            else:
                # Record the indeterminate solution result for stats (should be unlikely)
                indeterminate_resolution += 1
                this_question.indeterminate_resolution = True
                logger.warning("Result indeterminate! Continuing to next question...")
                continue

            # Compute who voted correctly and adjust reputation
            #   - Regardless of the “true” outcome, adjust reputation according to votes and resolved outcome
            for voter, vote in zip(participating_voters, collected_votes):
                voter.update_reputation(vote[0], resolved_outcome, this_question.all_context)

            # Record if resolved outcome is not the presupposed one
            this_question.resolved_correctly = resolved_outcome is this_question.true_outcome
            if resolved_outcome is not this_question.true_outcome:
                incorrectly_resolved += 1
            # Log correctness stats for precision/recall/accuracy
            if resolved_outcome is True:
                if this_question.true_outcome is True:
                    # Resolved true, actually true.
                    true_positive += 1
                else:
                    # Resolved true, actually false.
                    false_positive += 1
            else:
                if this_question.true_outcome is False:
                    # Resolved false, actually false.
                    true_negative += 1
                else:
                    # Resolved false, actually true.
                    false_negative += 1
            logger.info("Resolved outcome agrees with expected: %s", resolved_outcome is this_question.true_outcome)
            # END question proceessing
        # Save simulation state
        if epoch_number % epochs_per_save == 0:
            with open(f"./{save_directory}/{datetime.datetime.now().isoformat()}.json", "x") as f:
                current_state = {
                    "random_seed": random_seed,
                    "progress": {
                        "total_questions": total_questions,
                        "total_aborted": total_aborted,
                        "incorrectly_resolved": incorrectly_resolved,
                        "indeterminate_resolution": indeterminate_resolution,
                        "accuracy": (true_positive + true_negative) / (true_positive + true_negative
                                                                       + false_positive + false_negative),
                        "precision": (true_positive) / (true_positive + false_positive),
                        "recall": (true_positive) / (true_positive + false_negative)
                    },
                    "parameters": {
                        "data_file": data_file,
                        "answering_population_count": answering_population_count,
                        "experience_domains": experience_domains,
                        "questions_per_epoch": questions_per_epoch,
                        "epochs": epochs,
                        "epochs_per_save": epochs_per_save,
                        "save_directory": save_directory,
                        "use_reputation": use_reputation,
                        "fixed_threshold": fixed_threshold,
                        "rep_c1": rep_c1,
                        "rep_c2": rep_c2,
                        "reputation_affinity": reputation_affinity,
                        # Additional parameters below.
                        "bassline_contention_center": bassline_contention_center,
                        "confidence_threshold": confidence_threshold,
                        "experience_boost": experience_boost,
                        "secondary_context_count": secondary_context_count
                    },
                    "answering_entites": [e.dump_state() for e in answering_parties],
                    "question_pool": question_pool.dump_state(),
                }
                f.write(json.dumps(current_state, indent=4))


if __name__ == "__main__":
    main()
