class Image: pass

class Timeseries: pass

class AlignmentScores: pass


def get_llm_generated_answer(
    example: str | Image | Timeseries,
) -> str:
    """
    Args:
        example (str | Image | timeseries): The input example from which we want an LLM to generate some answer to a task,
          e.g., the emotion classification task.
    """
    raise NotImplementedError()


def isolate_individual_features(
    explanation: str
):
    """
    Args:
        explanation (str): The LLM-generated reasoning chain of why it gave a specific answer to an example.
        
    Returns:
        raw_atomic_claims (list[str]): A list of strings where each string is an isolated claim (includes relevant and irrelevant claims).
    """
    raise NotImplementedError()


def is_claim_relevant(
    example: str | Image | Timeseries,
    answer: str,
    atomic_claim: str,
) -> bool:
    """
    For a claim to be relevant, it must be:
        (1) Supported by the example.
        (2) Answers the question of why the LLM gave the answer it did for this specific example.

    Args:
        example (str | Image | timeseries): The input example from a dataset from which to distill the relevant features from.
        answer (str): The LLM-generated answer to the example.
        atomic_claim (str): A claim to check if it is relevant to the example.
    """
    raise NotImplementedError()


def distill_relevant_features(
    example: str | Image | Timeseries,
    answer: str,
    atomic_claims: list[str],
):
    """
    Args:
        example (str | Image | timeseries): The input example from a dataset from which to distill the relevant features from.
        answer (str): The LLM-generated answer to the example.
        raw_atomic_claims (list[str]): A list of strings where each string is an isolated claim (includes relevant and irrelevant claims).
    Returns:
        atomic_claims (list[str]): A list of strings where each string is a relevant claim.
    """
    raise NotImplementedError()


def calculate_expert_alignment_score(
    atomic_claims: list[str],
) -> AlignmentScores:
    """
    Computes the individual (and overall) alignment score of all the relevant atomic claims.

    Possibly needs a domain-independent aggregation function.
    Args:
        atomic_claims (list[str]): A list of strings where each string is a relevant claim.
    Returns:
        1. Alignment score of each individual atomic claims.
        2. Overall alignment score of all the atomic claims.
    """
    raise NotImplementedError()


