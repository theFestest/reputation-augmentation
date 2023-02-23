
import argparse
import json
import sys
import logging
import datetime
import random
import copy

from .players import AnsweringEntity, QuestionPool

# Make sure logging gets sent to the screen
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  # Change to warning for pending implementations

logger = logging.getLogger(__name__)

# TODO:
#   - be able to run without reputation for comparison
#   - include args/parameters in state dumps
#   - plot live results of counts required to hit threshold?
#   - include secondary context fields (could do more nested than 2 but increasingly rare if there is no overlap)
# Lower priority:
#   - implement state loading
#   - integrate time or location based context?
#   - implement program to load logged stats for analysis

# Data files derived from: https://en.wikipedia.org/wiki/Category:Main_topic_classifications


# Run with: python3.10 -m src.simulation, or similar
def main():
    # TODO: configure the simulation with arg parser for a default simulation run
    argparser = argparse.ArgumentParser()

    random_seed = datetime.datetime.now().isoformat()
    random.seed(a=random_seed)

    # Load context from full dataset in its own file (decide how many secondary facets)
    data_file = "data/context_data.json"
    data_file = "data/simple.json"
    with open(data_file) as f:
        data_set = json.load(f)
    context_set = data_set

    # PARAMETER
    answering_population_count = 100
    # PARAMETER
    experience_domains = 3
    # PARAMETER
    questions_per_epoch = 200  # 4000
    # PARAMETER
    epochs = 1
    # PARAMETER
    epochs_per_save = 1

    # Initialize statistics
    # - Overall stats
    total_questions = 0
    # - Per question stats
    incorrectly_resolved = 0
    indeterminate_resolution = 0
    participants_utilized = []
    majority_reputation = []
    minority_reputation = []

    # Initialize question pool from context_set
    question_pool = QuestionPool(context_set=context_set)

    answering_parties = []
    # Initialize the desired number of answering parties
    for _ in range(answering_population_count):
        answering_parties.append(
            AnsweringEntity(
                context_set,
                experience_domains,
            )
        )

    # Start running epochs
    for epoch_number in range(epochs):
        logger.info("Running epoch #%s", epoch_number)
        # Process the desired number of questions
        for question_number in range(questions_per_epoch):
            logger.info("Running question #%s in epoch #%s", question_number, epoch_number)
            # Generate a question from context
            this_question: QuestionPool.Question = question_pool.generate_question()
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
            while (available_reputation < this_question.req_confidence_theshold):
                if len(remaining_voters) > 0:
                    # TODO: make selection favor high reputation instead of being uniform
                    #   - this will be important for selecting the knowledgeable individuals
                    #   - maybe count=[count=floor(entity.rep/0.2) for each entity], so high rep occurs more
                    #   - use random.sample(iterable, COUNTS, k) to adjust distribution
                    selected_participant: AnsweringEntity = random.choice(remaining_voters)
                    remaining_voters.remove(selected_participant)
                    available_reputation += selected_participant.calculate_reputation(this_question.all_context)
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
                # participants_utilized.append(-1)
                # indeterminate_resolution += 1
                this_question.indeterminate_resolution = True
                this_question.aborted = True
                this_question.parties_used = None
                continue  # TODO: Aborting this question, so also update other ending stats!
            else:
                participants_utilized.append(len(participating_voters))
            # Record status of how many parties it took
            logger.info("Met confidence threshold with %s voters", participants_utilized[-1])

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
                    voter.vote(this_question.all_context, this_question.contention, this_question.true_outcome)
                    )

            cumulative_true_votes = 0
            cumulative_false_votes = 0
            resolved_outcome = None
            for vote in collected_votes:
                # NOTE: We can consider adding superlinearity with stake (only relevant if non-uniform stake)
                if vote[0] is True:
                    cumulative_true_votes += vote[1]*vote[2]  # Add: f(i) = reputation*stake for each vote
                else:  # Vote is Flase
                    cumulative_false_votes += vote[1]*vote[2]  # Add: reputation*stake for each vote
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
            this_question.parties_used = len(collected_votes)

            # Compute who voted correctly and adjust reputation
            #   - Regardless of the “true” outcome, adjust reputation according to votes and resolved outcome
            for voter, vote in zip(participating_voters, collected_votes):
                voter.update_reputation(vote[0], resolved_outcome, this_question.all_context)

            # Record if resolved outcome is not the presupposed one
            this_question.resolved_correctly = resolved_outcome is this_question.true_outcome
            if resolved_outcome is not this_question.true_outcome:
                incorrectly_resolved += 1
            logger.info("Resolved outcome agrees with expected: %s", resolved_outcome is this_question.true_outcome)
            # Increment total questions completed.
            total_questions += 1
            # END question proceessing
        # Save simulation state
        if epoch_number % epochs_per_save == 0:
            with open(f"./sim_states/{datetime.datetime.now().isoformat()}.json", "x") as f:
                current_state = {
                    "random_seed": random_seed,
                    "parameters": {
                        "data_file": data_file,
                        "answering_population_count": answering_population_count,
                        "experience_domains": experience_domains,
                        "questions_per_epoch": questions_per_epoch,
                        "epochs": epochs,
                        "epochs_per_save": epochs_per_save,
                    },
                    "answering_entites": [e.dump_state() for e in answering_parties],
                    "question_pool": question_pool.dump_state()
                }
                f.write(json.dumps(current_state, indent=4))


if __name__ == "__main__":
    main()
