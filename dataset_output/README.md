# dataset_output

Final merged Q&A dataset combining questions from the compiler pipeline and 10q generator. This is the primary input for LLM evaluation.

| File | Description |
|---|---|
| `original_questions.json` | Base questions before augmentation |
| `augmented_questions.json` | Questions after data augmentation (rephrasing, additional metadata) |
| `multi_turn_and_augmented_questions.json` | Combined file: augmented questions with multi-turn chains attached |

## Usage

`run_llm_eval.py` reads from `10q/final_qa_dataset.json` and `90q/random_questions_90.json` directly. The files here represent a merged/processed view used for broader experiments.
