## File 1
### Code
- Define metadata
    - categorial
    - yearly
    - comapnies
    - ratings
    - folders
- For each company, generate row
  - cal the values and put it in the row
- make sure folder exists, and write csv
- build json schema and save to file
  - for each column, generate its metadata info

### Desc
- generate data csv
- generate schema json

## File 2
### Code
- list concepts and concept metadata
- get csv from File 1
- validate that all the concepts are included in the csv
- iterate over rows, and concept columns in table
  - for each cell, create an *atom*
- save file

### Desc
From the csv, create the atoms json that will be used in File 4

## File 3
### Code
- define the operators allowed
- define the list of financial concepts
- generate tree (and error, if failed to generate)
  - try to generate the tree
    - if depth == 0:
      - return {"kind": "leaf", "semantic_type_in": ["amount"]}
    - with probability 0.35, create either a leaf, or a _"derived\_concept"_ subtree
      - filter for permissible _"derived\_concept"_ subtrees, based on the depth
      - if there are any, with probability given in args, create a _"derived\_concept"_ subtree
        - randomly choose a permissible one
        - return `{ "kind": "derived_concept", "name": name}`
      - else, create leaf.
        - return `{"kind": "leaf", "semantic_type_in": ["amount"]}`
    - choose one operator from `TIME_AGG_OPS` and `AMOUNT_BINARY_OPS`
    - build build left and right subtrees, with depth-=1
      - if operator is sum or diff, the left and right calls are identical to the current (using build_amount_tree)
      - if operator is mul, the left is identical to current. 
      - the right is a call to another function
        - if depth is 0, returns `{"kind": "leaf", "semantic_type_in": ["rate"]}`
        - choose operator from ("ratio", "growth")
        - if ratio, then make right and left as above.
        - if growth, then make the right and left as above, but force not to use `TIME_AGG_OPS`
    - return the node info
      - if in `("min", "max", "avg")`, return `{"kind": "time_agg","op": op,}`
      - if in ` ("sum", "diff", "mul")`, return `{"kind": "node","op": op,"left": left,"right": right,}`
  - afterwards, try validating the tree
    - check the root node.
      - if it is one of ` {"leaf", "time_agg", "derived_concept"}`, all good. return
      - otherwise, continue validating tree
        - make tree into a string
          - the string expands each derived concept into its nodes/components
        - at the same time, do the same for each of the derive concepts
        - checks if the tree string is equal to any of the dervie concept strings.
        - continues for left and right subnodes
  - keep trying, up to 200 times, until a valid tree is found
- save tree + some stats and the args used to a json file

### Desc
- generate a tree
- generates it randomly, and then checks if it is valid
- continues until it is valid
- the generation rules are above

## File 4
### Code
- reads atoms json from file 2
- reads tree from file 3
- compiles output
  - creates AtomIndex for fast atom lookup
  - instansiate a tree
    - if leaf
      - Picks a random entity (e.g., "Apple"), period (e.g., "2021"), and concept (e.g., "revenue")—unless the context (BindEnv) already specifies these.
      - Filters the atoms to those matching the chosen entity, period, and concept.
      - Randomly selects one atom from the filtered list.
      - Returns a Leaf node with the key of the selected atom.
    - if derived concept
      - Looks up the formula for the derived concept in the registry.
      - Recursively instantiates each argument of the formula (which could themselves be leaves or other derived concepts), using the same entity and period for all arguments.
      - Builds the expression tree for the formula (e.g., sum(revenue, cost_of_goods_sold)).
      - Wraps the expanded tree in a DerivedExpr node (preserving the name).
    - if time aggregate
      - For min/max:
        - Picks a random entity and concept.
        - Finds all periods for which that entity and concept exist.
        - Selects a random contiguous window of periods (at least 2).
        - For each period in the window, creates a leaf bound to the entity, concept, and that period.
        - Builds a left-nested binary tree of min or max nodes over these leaves.
      - For avg:
        - Same as above, but builds a sum over the leaves, then divides by the number of periods (using a ratio node with a Literal for the count).
    - For sum, diff, ratio:
      - Picks a shared entity and period (unless already specified in the context).
      - Recursively instantiates the left and right subtrees with this shared context.
      - Returns a Node with the operator and the two instantiated subtrees.
    - For mul:
      - Picks a shared entity and period, but allows the right subtree to have a different concept (e.g., multiplying an amount by a rate).
      - Recursively instantiates left and right subtrees with appropriate contexts.
      - Returns a Node for multiplication.
    - For growth:
      - Picks a single entity and concept, and two distinct periods.
      - Instantiates the left and right subtrees for the same entity/concept but different periods.
      - Returns a Node for growth.
  - After recursively processing all nodes, the result is a tree where:
    - All leaves are bound to specific atoms (cells in the data).
    - All derived concepts are expanded.
    - All time aggregates are replaced by explicit trees over real periods.
    - All binary operations are instantiated with concrete subtrees.
  - runs the semantic part
    - Create the SemanticAnalyzer
    - For each node in the tree, analyze does:
      - Looks up the corresponding Atom using its key.
      - Creates a Meaning object with:
      - if node is a leaf
        - kind="leaf_metric"
        - semantic_type (e.g., "amount" or "rate")
        - text (e.g., "Revenue for Apple in 2021")
        - entity, period, unit, concept, label, etc. from the atom.
        - Returns an AnalysisResult with this meaning, no children, and depth 0.
      - If the node is a Literal:
        - Creates a Meaning object with:
        - kind="literal_scalar"
        - semantic_type="ratio"
        - text as the literal value.
        - Returns an AnalysisResult with this meaning, no children, and depth 0.
      - If the node is a DerivedExpr:
        - Recursively analyzes the inner expression (expr.expr).
        - Wraps the result in a new Meaning with:
        - kind="derived_metric"
        - semantic_type from the inner meaning.
        - text like "gross profit for Apple in 2021".
        - Other fields from the inner meaning and the derived concept name.
        - Returns an AnalysisResult with this meaning, the inner result as its child, and the correct depth.
      - If the node is a Node (operator):
        - Recursively analyzes the left and right children.
        - Depending on the operator (sum, diff, ratio, mul, growth, min, max), calls a specialized method:
        - _analyze_sum, _analyze_diff, etc.
        - These methods:
          - Check the semantic types and context of the children.
          - Combine their meanings into a new Meaning object that describes the operation in human terms (e.g., "sum of revenue and expenses for Apple in 2021").
          - Set fields like kind, semantic_type, text, entity, period, etc.
          - Returns an AnalysisResult with this meaning, the two children, and the correct depth.
      - returns the resultant tree
- evaluate the numerical value
  - recursively traverse the tree
  - It recursively traverses the expression tree.
  - For a Leaf, it looks up the atom’s value in the atoms dictionary.
  - For a Literal, it returns the literal value.
  - For a DerivedExpr, it evaluates the inner expression.
  - For a Node (operator), it recursively evaluates the left and right children, then applies the operator:
  - "sum": adds the two values.
  - "diff": subtracts right from left.
  - "ratio": divides left by right (with zero-division check).
  - "mul": multiplies the two values.
  - "growth": computes growth rate as (left - right) / right (with zero-division check).
  - "min"/"max": returns the minimum/maximum of t
- saves it
### Desc
- Take a symbolic, typed expression tree (from File 3) and a set of spreadsheet-like atoms (from File 2).
- Bind the symbolic tree to real data:
  - Each leaf is mapped to a specific cell (atom) in the data, using random but context-consistent choices for entity, period, and concept.
  - Derived concepts are expanded into their formulas, recursively binding their arguments.
  - Time aggregates (min, max, avg) are instantiated over random contiguous windows of periods for a chosen entity and concept.
  - Binary operators (sum, diff, ratio, mul, growth) are instantiated by recursively binding their left and right subtrees, sharing context as appropriate.
- After instantiation, the tree is fully concrete: all leaves are bound to real atoms, all derived concepts are expanded, and all aggregates are explicit.
- Analyze the concrete tree semantically:
  - For each node, generate a human-readable description (Meaning) of what it represents, combining information from its children as needed.
  - Build an annotated tree (AnalysisResult) that mirrors the structure of the expression, with semantic information at each node.
- Evaluate the tree numerically:
  - Recursively compute the value of the expression, applying the correct operation at each node and using the actual values from the atoms.
- Generate a natural language question describing the expression, using the semantic analysis.
- Save the resulting bundle (expression, semantic info, question, answer) to a JSON file.
- Optionally, generate a visualization of the bound tree.

## File 5
### Code
- read csv from file 1
- make atoms data using file 2
- feed to file 4 to create atoms var
- create index, analyzer, evaluator and renderer from file 4
- generate n question
  - for each question, generate a tree (unless an exception occured, in that case it retries)
  - generate a tree using file 3
  - generate info on the treefrom file 4
  - compile the info into a csv line
- write csv