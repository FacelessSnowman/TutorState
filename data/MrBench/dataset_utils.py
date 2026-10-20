"""Extract flat examples directly from the current MRBench V2 records.

Never send an entire extracted example to a model. Construct judge inputs as::

    judge_input = {key: example[key] for key in (
        "conversation_history", "ground_truth_solution", "candidate_response")}

Default tutor input is only ``example["conversation_history"]``. If a reference
solution is intentionally supplied, construct that payload explicitly and use
it identically for both tutors. IDs, model names, labels and split metadata are
for analysis only. Dataset integrity/group audits remain separate from extraction.
"""
from copy import deepcopy

RATING_DIMENSIONS = (
    'Mistake_Identification', 'Mistake_Location', 'Revealing_of_the_Answer',
    'Providing_Guidance', 'Actionability', 'Coherence', 'Tutor_Tone', 'Humanlikeness',
)


def _validate_split(split):
    if split not in (None, 'train', 'validation', 'test'):
        raise ValueError(f'Invalid split {split!r}; expected train, validation, test, or None')


def _conversation(record):
    if not isinstance(record, dict):
        raise ValueError('Each conversation must be a dictionary')
    cid = record.get('conversation_id')
    if not isinstance(cid, str) or not cid.strip():
        raise ValueError('conversation_id must be a non-empty string')
    history = record.get('conversation_history')
    if not isinstance(history, str) or not history.strip():
        raise ValueError(f'{cid}: conversation_history must be a non-empty string')
    solution = record.get('Ground_Truth_Solution')
    if solution is None or (isinstance(solution, str) and solution.strip().casefold() in ('', 'not available')):
        solution = None
    return {'conversation_id': cid, 'conversation_history': history,
            'ground_truth_solution': deepcopy(solution)}


def _ratings(annotation, cid, model):
    if not isinstance(annotation, dict):
        raise ValueError(f'{cid}/{model}: annotation must be a dictionary with eight dimensions')
    ratings = deepcopy(annotation)
    if 'humanlikeness' in ratings:
        original = ratings.pop('humanlikeness')
        if 'Humanlikeness' in ratings and ratings['Humanlikeness'] != original:
            raise ValueError(f'{cid}/{model}: conflicting humanlikeness keys')
        ratings['Humanlikeness'] = original
    missing = [key for key in RATING_DIMENSIONS if key not in ratings]
    if missing:
        raise ValueError(f'{cid}/{model}: missing annotation dimensions: {missing}')
    return ratings


def get_judge_examples(records, include_labels=False, split=None):
    """Return one flat example per candidate; attach only an explicit split.

    With labels enabled, human_ratings is analysis/evaluation data. Build the
    three-field judge_input documented above; exclude response_source and labels.
    Candidate identity is (conversation_id, response_source); duplicates raise.
    """
    _validate_split(split)
    examples, seen = [], set()
    for record in records:
        context = _conversation(record)
        cid = context['conversation_id']
        candidates = record.get('anno_llm_responses')
        if not isinstance(candidates, dict):
            raise ValueError(f'{cid}: anno_llm_responses must be a dictionary')
        for model, candidate in candidates.items():
            pair = (cid, model)
            if pair in seen:
                raise ValueError(f'Duplicate judge response pair: conversation_id={cid!r}, response_source={model!r}')
            seen.add(pair)
            if not isinstance(candidate, dict):
                raise ValueError(f'{cid}/{model}: candidate must be a dictionary')
            response = candidate.get('response')
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f'{cid}/{model}: response must be a non-empty string')
            example = {**context, 'response_source': model, 'candidate_response': response}
            if include_labels:
                example['human_ratings'] = _ratings(candidate.get('annotation'), cid, model)
            if split is not None:
                example['split'] = split
            examples.append(example)
    return examples


def get_tutor_examples(records, include_ground_truth_solution=False, split=None):
    """Return one flat context per conversation, never candidates or annotations.

    include_ground_truth_solution is a deprecated compatibility argument and has
    no effect: the reference is always returned for offline evaluation. Callers
    explicitly choose whether to supply it to both tutors; by default send only
    example['conversation_history']. No dialogue is truncated or rewritten.
    """
    _validate_split(split)
    examples = []
    for record in records:
        example = _conversation(record)
        if split is not None:
            example['split'] = split
        examples.append(example)
    return examples
