# Bakeoff Scoring v1

Use hidden private evals and runtime evidence as the primary decision source. Public benchmark reputation is not a scoring dimension.

## Scoring dimensions

1. **Hidden coding final verified success**  
   Highest weight because this repo already optimizes for verified coding outcomes, not one-shot style.

2. **Hidden coding repair-loop success**  
   Matters because the finished agent loop includes bounded repair and should be judged on final verified recovery, not only initial patch quality.

3. **Source-grounded truthfulness**  
   Unsupported confident answers are a direct mismatch for the repo’s truthfulness design.

4. **Hidden exact symbolic correctness**  
   Exactness matters because the repo already routes exact tasks deterministically and should reward models that cooperate with that discipline.

5. **Abstention precision**  
   A model that guesses instead of asking for a tool or source is operationally worse even when it sounds fluent.

6. **Retrieval-grounded QA**  
   Retrieval-aware answering matters, but only with evidence and citations.

7. **Runtime fit / stability**  
   A strong-looking candidate that is operationally fragile is not a main-path winner.

8. **Structured output reliability**  
   The agent loop depends on machine-readable outputs and narrow diffs.

9. **Cost / latency per verified success**  
   Cost and speed matter, but only in relation to verified success.

10. **Deployment fit for the Linux-first workstation path**  
    The chosen path must be realistic for the intended Phase A and near-term execution environment.

## Tie-break rules

- If two candidates are within five weighted points, prefer the one with better runtime fit and lower cost per verified success.
- Do not give the current custom base a sentimental tie-break advantage.
- Do not give the 14B slot an automatic advantage for being larger.
