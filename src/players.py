
import copy
import random
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SimulationEntity(ABC):

    @abstractmethod
    def dump_state():
        pass

    @abstractmethod
    def load_state():
        pass


class AnsweringEntity(SimulationEntity):

    def __init__(self, context_set: dict, preferential_context_count: int):

        # Use dictionary as a sparse vector of reputation for this identity
        #   - Track total contributions and total correct per domain
        self.sparse_reputation: dict[str, tuple[int, int]] = {}

        # Are these unnecessary?
        # TODO: select preferential_context_count base context from context_set
        #   select random dictionary keys
        # knowledge_domains = random.sample(context_set.keys(), k=preferential_context_count)
        # starting
        # TODO: Assign and store (relatively) high probability for answering in these domains?

    def vote(self, question_context) -> tuple[bool, float, float]:
        # Choosing an outcome based on, c, the inherent probability to align with MPPO (for this question?)
        #   - Here we only report a vote and not a peer prediction because we care about the resolution
        #     process are are assuming that peer prediction produces truthful incentives.
        # Return (vote, reputation, stake)
        # TODO: implement proper computed return
        # logger.warning("unimplemented, returning default tuple")
        return (random.choice([True, True, False]), self.calculate_reputation(question_context), 1)

    # TODO: cache these to avoid a second call? cache is only valid for one iteration?
    def calculate_reputation(self, question_context):
        # TODO: project our reputation onto the question context and return confidence value
        # logger.warning("unimplemented, returning integer weight")
        return 1

    def update_reputation(self, voted: bool, true_result: bool, question_context):
        # Increcment +1 if agreeing with true result, -1 if disagreeing
        if voted is true_result:
            increment = +1
        else:
            increment = -1
        for domain in question_context:
            # (correct votes, all votes) per domain
            current_rep: tuple = self.sparse_reputation.get(domain, (0, 0))
            self.sparse_reputation[domain] = (current_rep[0] + increment, current_rep[1] + 1)

    def dump_state(self):
        # TODO: implement
        return {}

    def load_state(self, state_data: dict):
        # TODO: implement
        pass


class QuestionPool(SimulationEntity):

    class Question(SimulationEntity):
        def __init__(self, primary: str, secondary: list[str] = None) -> None:
            self.primary_context = primary
            self.secondary_context = secondary if secondary is not None else []
            # PARAMETER: TODO: assign dynamically?
            self.contention: int = 0.7
            self.true_outcome = random.choice([True, False])
            # PARAMETER: TODO: assign dynamically?
            self.req_confidence_theshold: float = 30.0

        @property
        def all_context(self) -> list[str]:
            return [self.primary_context, *self.secondary_context]

        def __str__(self) -> str:
            # For now just return all of your context
            return str(self.all_context)

        def dump_state(self):
            # TODO: implement
            return self.__str__()

        def load_state(self, state_data: dict):
            # TODO: implement
            pass

    def __init__(self, context_set):
        self.context_set = copy.deepcopy(context_set)
        self.question_history = []

    def generate_question(self, secondary_count=0) -> Question:
        # TODO: realistically these will be somewhat biased toward certain domains
        # Select random domains from context lists
        new_question = QuestionPool.Question(
            primary=random.choice(list(self.context_set.keys())),
            # TODO: assign secondary from nested sublists!
            secondary=[random.choice(list(self.context_set.keys()))]
        )

        # Keep questions in order!
        self.question_history.append(new_question.dump_state())
        return new_question

    def dump_state(self):
        return {
            "context_set": self.context_set,
            "question_history": self.question_history
        }

    def load_state(self, state_data: dict):
        # TODO: implement
        pass


# TODO: may be unnecessary if they don't own any state.
#   May be more useful if allowing multi-party funding.
class CreatingEntity(SimulationEntity):

    def __init__(self):
        pass

    def dump_state(self):
        logger.warning("unimplemented")

    def load_state(self):
        logger.warning("unimplemented")
