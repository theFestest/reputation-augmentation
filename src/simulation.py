
import argparse
import json
import sys
import logging
import datetime
import random
import copy

from .players import AnsweringEntity, CreatingEntity, QuestionPool

# Make sure logging gets sent to the screen
logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # Change to warning for pending implementations

logger = logging.getLogger(__name__)

# Lower priority:
#   - implement state loading
#   - integrate time or location based context?
#   - implement program to load logged stats for analysis

# TODO: introduce statistics keeping
# TODO: include secondary context fields
# Derived from: https://en.wikipedia.org/wiki/Category:Main_topic_classifications
DEFAULT_CONTEXT = {
    "Academic disciplines": {},
    "Business": {},
    "Communication": {},
    "Concepts": {},
    "Culture": {},
    "Economy": {},
    "Education": {},
    "Energy": {},
    "Engineering": {},
    "Entertainment": {},
    "Entities": {},
    "Ethics": {},
    "Food and drink": {},
    "Geography": {},
    "Government": {},
    "Health": {},
    "History": {},
    "Human behavior": {},
    "Humanities": {},
    "Information": {},
    "Internet": {},
    "Knowledge": {},
    "Language": {},
    "Law": {},
    "Life": {},
    "Mass media": {},
    "Mathematics": {},
    "Military": {},
    "Nature": {},
    "People": {},
    "Philosophy": {},
    "Politics": {},
    "Religion": {},
    "Science": {},
    "Society": {},
    "Sports": {},
    "Technology": {},
    "Time": {},
    "Universe": {},
}


def main():

    # TODO: configure the simulation with arg parser for a default simulation run
    argparser = argparse.ArgumentParser()

    random_seed = datetime.datetime.now().isoformat()
    random.seed(a=random_seed)

    # TODO: load context from full dataset in its own file (decide how many secondary facets)
    context_set = DEFAULT_CONTEXT

    # PARAMETER
    answering_population_count = 100
    # PARAMETER
    preferential_domains = 10
    # PARAMETER
    questions_per_epoch = 4000
    # PARAMETER
    epochs = 1
    # PARAMETER
    epochs_per_save = 1

    # Initialize question pool from context_set
    question_pool = QuestionPool(context_set=context_set)

    answering_parties = []
    # Initialize the desired number of answering parties
    for _ in range(answering_population_count):
        answering_parties.append(
            AnsweringEntity(
                context_set,  # TODO: check that this is just a reference and not the whole data set
                preferential_domains,
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
                    #   - use random.sample(iterable, COUNTS, k) to adjust distribution
                    selected_participant: AnsweringEntity = random.choice(remaining_voters)
                    remaining_voters.remove(selected_participant)
                    available_reputation += selected_participant.calculate_reputation(this_question.all_context)
                    participating_voters.append(selected_participant)
                else:
                    # TODO: log aborting this question and keep stats
                    logger.info("Out of voters! Cannot reach threshold to evaluate this question.")
                    break  # Don't loop forever, just give up on this question.

            # TODO: log status of how many parties it took
            logger.info("Met confidence threshold with %s voters", len(participating_voters))

            # Allow selected entities to vote; collect votes
            #   - Each entity has a probability c to vote the “true” outcome [parameter]
            #       Key: we suppose some entities naturally have a higher probability
            #       of correctness (due to their knowledge of the context) and we hope
            #       reputation can be a tool to allow long term system behavior to favor
            #       this knowledge.
            #   - Each entity has a particular stake contributed when voting [parameter; these may just be uniform]
            question_evaluation: int = 0
            # TODO: check implemention of the resolution algorithm
            collected_votes: tuple[bool, float, float] = []  # (vote, reputation, stake)
            for voter in participating_voters:
                # Collect tuples of vote, reputation, and stake (uniform for now)
                # TODO: do we need to know contention of question? Contention is a parameter for voters?
                collected_votes.append(voter.vote(this_question.all_context))

            cumulative_true_votes = 0
            cumulative_false_votes = 0
            resolved_outcome = None
            for vote in collected_votes:
                # TODO: consider adding superlinearity with stake (only relevant if non-uniform stake)
                if vote[0] is True:
                    cumulative_true_votes += vote[1]*vote[2]  # Add: f(i) = reputation*stake for each vote
                else:  # Vote is Flase
                    cumulative_false_votes += vote[1]*vote[2]  # Add: reputation*stake for each vote
            # Compute the _resolved outcome_ based on votes, and
            #  - Utilize reputation weights and stakes to resolve the outcome
            if(cumulative_true_votes > cumulative_false_votes):
                resolved_outcome = True
            elif(cumulative_true_votes < cumulative_false_votes):
                resolved_outcome = False
            else:
                # TODO: log this result for stats (should be unlikely)
                logger.warning("Result indeterminate!")

            # Compute who voted correctly and adjust reputation
            #   - Regardless of the “true” outcome, adjust reputation according to votes and resolved outcome
            for voter, vote in zip(participating_voters, collected_votes):
                voter.update_reputation(vote[0], resolved_outcome, this_question.all_context)

            # TODO: log if resolved outcome is the presupposed one
            logger.info("Resolved outcome agrees with expected: %s", resolved_outcome is this_question.true_outcome)

            # END question proceessing
        # Save simulation state
        if epoch_number % epochs_per_save == 0:
            with open(f"./sim_states/{datetime.datetime.now().isoformat()}.json", "x") as f:
                current_state = {
                    "random_seed": random_seed,
                    "parameters": {},
                    "answering_entites": [e.dump_state() for e in answering_parties],
                    "question_pool": question_pool.dump_state()
                }
                f.write(json.dumps(current_state))


if __name__ == "__main__":
    main()
