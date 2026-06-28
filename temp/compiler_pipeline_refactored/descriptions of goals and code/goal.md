# Refactored File Goals

## 1. Data Preparation & Atomization

- Define and manage metadata (categorical, yearly, companies, ratings, folders, etc.).
- Generate a data CSV:
  - For each company, compute and populate row values.
  - Ensure output folder exists and write the CSV.
- Build and save a JSON schema describing each column’s metadata.
- List all concepts and their metadata.
- Validate that all concepts are present in the generated CSV.
- Iterate over CSV rows and columns to create *atoms* (individual data points).
- Save the atoms as a JSON file for downstream use.

---

## 2. Tree Generation, Binding, and Analysis

- Define allowed operators and financial concepts.
- Randomly generate a symbolic expression tree (with error handling and retries):
  - Use rules for depth, operator selection, and derived concepts.
  - Validate generated trees for correctness and uniqueness.
- Save the generated tree and associated metadata.
- Read atoms and tree files.
- Bind the symbolic tree to real data:
  - Map leaves to specific atoms (cells) in the data.
  - Expand derived concepts and instantiate aggregates and operators.
- Analyze the bound tree semantically:
  - For each node, generate a human-readable description (Meaning).
  - Build an annotated tree with semantic info at each node.
- Evaluate the tree numerically using the atom values.
- Generate a natural language question describing the expression.
- Save the resulting bundle (expression, semantic info, question, answer) to a JSON file.
- Optionally, generate a visualization of the bound tree.

---

## 3. Question Generation & Dataset Creation

- Read the CSV and atoms data.
- Use the tree generation and analysis pipeline to create multiple questions:
  - For each question, generate a tree, bind it, analyze it, and evaluate it.
  - Compile all information into a CSV line.
- Write the final dataset as a CSV file.