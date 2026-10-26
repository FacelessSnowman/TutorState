from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    use_analyser: bool
    use_memory: bool
    use_planner: bool


CONDITIONS = {
    "baseline": Condition(
        use_analyser=False,
        use_memory=False,
        use_planner=False,
    ),
    "analyser": Condition(
        use_analyser=True,
        use_memory=False,
        use_planner=False,
    ),
    "memory": Condition(
        use_analyser=True,
        use_memory=True,
        use_planner=False,
    ),
    "full": Condition(
        use_analyser=True,
        use_memory=True,
        use_planner=True,
    ),
}