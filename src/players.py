
import copy
import random
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class SimulationEntity(ABC):

    @abstractmethod
    def dump_state():
        pass

    @abstractmethod
    def load_state():
        pass


class AnsweringEntity(SimulationEntity):

    def __init__(self, context_set: dict, c1: float, c2: float, experience_domains_count: int = 1, exp_boost=0.1):

        # Use dictionary as a sparse vector of reputation for this identity
        #   - Track total contributions and total correct per domain
        self.sparse_rep: dict[str, tuple[int, int]] = {}  # (total, correct)

        # Count of questions this party has participated in
        self._participation_count = 0

        # Select experience_domains_count base context from context_set
        #   - these are primary domains that give a boost in probability of correctness
        self.knowledge_domains = random.sample(context_set.keys(), k=experience_domains_count)
        # Growth limit in sigmoid reputation function
        self._c1 = c1  # trying 3?
        # Growth rate in sigmoid reputation function
        self._c2 = c2  # if less than 1, it is very hard to reach threshold. Do (1 + rep) so min rep is 1?
        # PARAMETER: the probability increase granted by being knowledgable
        self._experience_boost = exp_boost

    def vote(self, q_context, inverse_contention: float, true_outcome: bool, rep: bool) -> tuple[bool, float, float]:
        # Choose an outcome based on, c, their inherent probability to align with MPPO for this question.
        #   - We can view the bassline c as the inverse_contention of the question. (~.5 high contention, ~1 low)
        #   - Each party will have some offset from this bassline. Experienced parties are assume to have higher.
        #   - Here we only report a vote and not a peer prediction because we care about the resolution
        #     process are are assuming that peer prediction produces truthful incentives.
        # Return (vote, reputation, stake)
        #  Vote: the proposition vote for this question for this voter
        #  Reputation: the reputation for this given question context
        #  Stake: the voting stake contributed, default to 1 for simpler analysis
        self._participation_count += 1
        domain_experience = False
        for domain in self.knowledge_domains:
            if domain in q_context:
                domain_experience = True
        # Vote based on a random float being LESS than the inverse contention (plus boost)
        #  Consider inverse_contention \in (0.5, 1] centered at ~0.75
        #   low chance that: random.random() > inverse_contention --> vote opposite true_outcome
        #   high chance that: random.random() < inverse_contention --> vote true_outcome.
        # Given the "true answer", select the opposite if voting against (unaligned)
        if domain_experience:
            aligned = random.random() < min(inverse_contention + self._experience_boost, 1)
        else:
            aligned = random.random() < inverse_contention
        # We are assuming that everyone has a preferred side.
        #  - Population level base contention with (imagined) "true value" sets the preferred side
        #  - Random selection for each party which side to take
        if not aligned:
            # Vote against true outcome
            vote = not true_outcome
        else:
            vote = true_outcome
        if rep:
            return (vote, self.calculate_confidence(q_context), 1)
        else:
            # Return None for rep to ensure this isn't used anywhere.
            return (vote, None, 1)

    # cache these to avoid a second call?
    # cache is only valid for one iteration: include question #, epoch # in args?
    def calculate_confidence(self, question_context, default_rep=1):
        # Project our reputation onto the question context and return confidence value
        # - Iterate each context domain in the question
        # - compute historical correctness (handle negatives?)
        # - append to list
        # - Return vector magnitude of this list
        rep_vector = []
        relevant_reputation = [self.sparse_rep[c] for c in question_context if c in self.sparse_rep.keys()]
        for total, correct in relevant_reputation:
            # Note: Handle negatives? For now just default to zero. (see below)
            rep_vector.append(max(correct, 0))  # dividing here creates an extra penalty and traps < 1.
        logger.debug("Reputation vector is: %s", rep_vector)
        magnitude = np.linalg.norm(rep_vector, ord=1)  # + default_rep  # if len(rep_vector) != 0 else default_rep
        # Use magnitude to evaluate in the sigmoid s(x) = c1(1/(1+exp(−x·c2))−1/2)
        #   - could handle negatives if our "projection" method allows it
        #   - this would permit negative contributions for persistent incorrectness
        adjusted_rep = self._c1*(1/(1+np.exp(-magnitude*self._c2))-1/2)
        logger.info("Voting with %s reputation.", 1+adjusted_rep)
        # NOTE: currently configured to only be able to increase vote weight, not decrease.
        #   - consider allowing reducing contribution if historically incorrect? needs projection changes
        #   - how to decide if incorrect enough to deduct instead?
        #   perhaps, sum of contributions in these domains is incorrect on avg?
        # breakpoint()
        return 1 + adjusted_rep

    def update_reputation(self, voted: bool, resolved_outcome: bool, question_context):
        # Increment +1 if agreeing with true result, -1 if disagreeing
        if voted is resolved_outcome:
            increment = +1
        else:
            increment = -1
        for domain in question_context:
            # (correct votes, all votes) per domain
            current_rep: tuple = self.sparse_rep.get(domain, (0, 0))
            self.sparse_rep[domain] = (current_rep[0] + 1, current_rep[1] + increment)

    def dump_state(self):
        return {
            "participation_count": self._participation_count,
            "reputation": self.sparse_rep,
            "knowledge_domains": self.knowledge_domains
            }

    def load_state(self, state_data: dict):
        self.sparse_rep = state_data['reputation']
        self.knowledge_domains = state_data['knowledge_domains']
        self._participation_count = state_data['participation_count']


class QuestionPool(SimulationEntity):

    class Question(SimulationEntity):
        def __init__(self,
                     primary: str,
                     secondary: list[str] = None,
                     confidence_threshold: float = 50.0,
                     contention_center: float = 0.7) -> None:
            self.primary_context = primary
            self.secondary_context = secondary if secondary is not None else []
            # PARAMETER: bassline "inverse" contention. ~.5 is high contention, ~1 is low.
            #   Assign dynamically (default = 0.7): uniform instead? random.uniform(0.51, 0.99)
            #   Currently: (.5, 1] centered at .7
            self.contention: int = max(0.51, min(random.gauss(mu=contention_center, sigma=.1), 1))
            self.true_outcome = random.choice([True, False])
            # PARAMETER: confidence threshold to be considered answerable
            #   Assign dynamically: is uniform better? or random.gauss(mu=30, sigma=5.0)?
            self.req_confidence_theshold: float = confidence_threshold
            # Result statistics
            self.aborted = False
            self.indeterminate_resolution = False
            self.parties_used: Optional[bool] = None
            self.resolved_correctly: Optional[bool] = None

        @property
        def all_context(self) -> list[str]:
            return [self.primary_context, *self.secondary_context]

        def __str__(self) -> str:
            # For now just return all of your context
            return str(self.all_context)

        def dump_state(self):
            return {
                "primary_context": self.primary_context,
                "secondary_context": self.secondary_context,
                "contention": self.contention,
                "true_outcome": self.true_outcome,
                "req_confidence": self.req_confidence_theshold,
                "aborted": self.aborted,
                "indeterminate_resolution": self.indeterminate_resolution,
                "parties_used": self.parties_used,
                "resolved_correctly": self.resolved_correctly
            }

        def load_state(self, state_data: dict):
            self.primary_context = state_data['primary_context']
            self.secondary_context = state_data['secondary_context']
            self.contention = state_data['contention']
            self.true_outcome = state_data['true_outcome']
            self.req_confidence_theshold = state_data['req_confidence']
            self.aborted = state_data['aborted']
            self.indeterminate_resolution = state_data['indeterminate_resolution']
            self.parties_used = state_data['parties_used']
            self.resolved_correctly = state_data['resolved_correctly']

    def __init__(self, context_set):
        self.context_set = copy.deepcopy(context_set)
        self.question_history = []

    def generate_question(self, secondary_count=2, confidence_threshold=50.0, contention_center=0.7) -> Question:
        # PARAMETER: secondary_count
        # NOTE: Realistically these will be somewhat biased toward certain domains
        # Select random domains from context lists
        primary = random.choice(list(self.context_set.keys()))
        secondary = []
        for _ in range((secondary_count)):
            # Pick secondary domains from the sub-collection at the primary domain
            #   if no sub-collection, pick another primary domain
            sub_collection = self.context_set.get(primary, {})
            choice = random.choice(list(sub_collection if len(sub_collection) != 0 else self.context_set.keys()))
            secondary.append(choice)
        new_question = QuestionPool.Question(
            primary=primary,
            # Assign secondary from nested sublists (after including this full data)!
            secondary=secondary,
            confidence_threshold=confidence_threshold,
            contention_center=contention_center
        )

        # Keep questions in order!
        self.question_history.append(new_question)
        return new_question

    def dump_state(self):
        return {
            "context_set": self.context_set,
            "question_history": [q.dump_state() for q in self.question_history]
        }

    def load_state(self, state_data: dict):
        self.context_set = state_data['context_set']
        # Should stay ordered?
        self.question_history = state_data['question_history']


# NOTE: may be unnecessary if they don't own any state.
#   May be more useful if allowing multi-party funding.
class CreatingEntity(SimulationEntity):

    def __init__(self):
        pass

    def dump_state(self):
        logger.warning("unimplemented")

    def load_state(self):
        logger.warning("unimplemented")
