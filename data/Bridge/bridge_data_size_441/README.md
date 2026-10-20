# data_size_441

One JSON array per split. Dataset size is measured in **unique histories**.

| Split | Rows | Expert annotations |
| --- | --- | --- |
| train | 309 | 461 |
| validation | 66 | 96 |
| test | 66 | 103 |

## Structure

Each row has `history_id`, `lesson_topic`, `c_h`, `annotations`. Each annotation has `annotation_id`, `e_clean`, `z_what_clean`, `c_r_`. All original IDs, conversation text, topic strings and retained annotation bundles are unchanged.

Multiple annotations are alternative expert references for the same history. Keep their error label, action and response together. A 142-history subset can therefore contain more than 142 annotations.

## Splitting

Exact histories never cross splits within this version. Training and validation history counts are rounded to the nearest integer from 70% and 15%; test gets the remainder. Python random.Random(42) shuffles canonically sorted histories. The best of 1,000 candidates is chosen by summed squared deviations divided by expected annotation-level label counts for both cleaned-label families across all splits. This approximates balance without selecting one representative label, oversampling or dropping annotation variants.

The 441-history and filtered versions are split independently: the same history may have different split membership between versions. Do not mix their training and evaluation files. Separation is by exact c_h, not by broader tutoring session or semantic similarity.

## Run metadata

```json
{
  "seed": 42,
  "candidates": 1000,
  "selected_candidate": 74,
  "distribution_score": 7.877588462486645,
  "sizes": [
    309,
    66,
    66
  ]
}
```

## Annotation-level label distributions

### e_clean

| Label | train | validation | test | Total |
| --- | --- | --- | --- | --- |
| careless | 37 | 9 | 9 | 55 |
| diagnose | 91 | 19 | 18 | 128 |
| guess | 161 | 38 | 38 | 237 |
| imprecise | 33 | 7 | 8 | 48 |
| misinterpret | 34 | 6 | 5 | 45 |
| no_error | 13 | 1 | 4 | 18 |
| right-idea | 92 | 16 | 21 | 129 |

### z_what_clean

| Label | train | validation | test | Total |
| --- | --- | --- | --- | --- |
| AFFIRM | 20 | 4 | 4 | 28 |
| CORRECT | 31 | 8 | 3 | 42 |
| ENCOURAGE | 15 | 4 | 3 | 22 |
| EXPLAIN | 146 | 30 | 36 | 212 |
| HINT | 55 | 11 | 13 | 79 |
| PROBE | 143 | 28 | 33 | 204 |
| SIMPLIFY | 51 | 11 | 11 | 73 |

## Validation

- Saved arrays were reloaded and compared to the intended records.
- No source dataset or screening CSV was modified.
- All applicable history IDs, annotation IDs, cleaned labels, topic strings and expert responses were preserved.
