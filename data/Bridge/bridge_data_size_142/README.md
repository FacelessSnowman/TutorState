# data_size_142

One JSON array per split. Dataset size is measured in **unique histories**.

| Split | Rows | Expert annotations |
| --- | --- | --- |
| train | 99 | 150 |
| validation | 21 | 35 |
| test | 22 | 34 |

## Structure

Each row has `history_id`, `lesson_topic`, `c_h`, `annotations`. Each annotation has `annotation_id`, `e_clean`, `z_what_clean`, `c_r_`. All original IDs, conversation text, topic strings and retained annotation bundles are unchanged.

Multiple annotations are alternative expert references for the same history. Keep their error label, action and response together. A 142-history subset can therefore contain more than 142 annotations.

## Splitting

Exact histories never cross splits within this version. Training and validation history counts are rounded to the nearest integer from 70% and 15%; test gets the remainder. Python random.Random(42) shuffles canonically sorted histories. The best of 1,000 candidates is chosen by summed squared deviations divided by expected annotation-level label counts for both cleaned-label families across all splits. This approximates balance without selecting one representative label, oversampling or dropping annotation variants.

The 441-history and filtered versions are split independently: the same history may have different split membership between versions. Do not mix their training and evaluation files. Separation is by exact c_h, not by broader tutoring session or semantic similarity.

## Filter rule

Keep the union of histories marked KEEP in `previous_codex_classification` (the initial c_h-only screening) and histories manually marked KEEP in the current reviewed CSV. Do not keep automated REVIEW/EXCLUDE cases unless manually promoted to KEEP. All annotations belonging to a kept history are retained. In this source CSV there are no manual REVIEW or EXCLUDE values. If such values are added later, this builder stops for an explicit decision about their precedence.

For this run, the initial-check union and updated-check union select exactly the same histories.

```json
{
  "initial_automated_keep": 127,
  "manual_keep": 15,
  "overlap_initial_and_manual": 0,
  "filtered_histories": 142,
  "filtered_annotations": 219,
  "updated_automated_keep": 128,
  "same_selection_using_updated_checks": true
}
```

## Run metadata

```json
{
  "seed": 42,
  "candidates": 1000,
  "selected_candidate": 467,
  "distribution_score": 10.599668990623503,
  "sizes": [
    99,
    21,
    22
  ]
}
```

## Annotation-level label distributions

### e_clean

| Label | train | validation | test | Total |
| --- | --- | --- | --- | --- |
| careless | 21 | 5 | 4 | 30 |
| diagnose | 10 | 2 | 3 | 15 |
| guess | 58 | 14 | 13 | 85 |
| imprecise | 18 | 4 | 3 | 25 |
| misinterpret | 14 | 4 | 4 | 22 |
| no_error | 2 | 0 | 0 | 2 |
| right-idea | 27 | 6 | 7 | 40 |

### z_what_clean

| Label | train | validation | test | Total |
| --- | --- | --- | --- | --- |
| AFFIRM | 9 | 2 | 3 | 14 |
| CORRECT | 14 | 3 | 2 | 19 |
| ENCOURAGE | 3 | 0 | 0 | 3 |
| EXPLAIN | 52 | 9 | 16 | 77 |
| HINT | 15 | 4 | 5 | 24 |
| PROBE | 37 | 11 | 5 | 53 |
| SIMPLIFY | 20 | 6 | 3 | 29 |

## Validation

- Saved arrays were reloaded and compared to the intended records.
- No source dataset or screening CSV was modified.
- All applicable history IDs, annotation IDs, cleaned labels, topic strings and expert responses were preserved.
