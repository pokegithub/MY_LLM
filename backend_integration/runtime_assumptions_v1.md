# Runtime Assumptions v1

## Primary target environment

- Linux-first workstation path
- roughly 48 GB class GPU memory / RTX 6000-class execution target
- local Python process with direct model loading

## Non-target environments for first integration

- RTX 2050 4 GB: dev/spec/validation only, not the first real backend execution target
- serious cloud / multi-GPU: later if needed, not the first integration target
- Windows: not the primary first backend path

## Dependency assumptions

- repo Python environment can load the backend runtime dependencies needed for one local dense model path
- the chosen candidate model family must be compatible with the target Linux-first workstation setup
- no broad runtime abstraction layer is assumed for the first integration

## Model loading assumptions

- one selected bakeoff candidate model is loaded locally
- tokenizer/model artifacts are accessible from local disk
- loading must fail cleanly when the model path is missing, invalid, or incompatible

## Output format assumptions

- backend emits one structured candidate at a time
- candidate output must map to the existing `Candidate` and `FileEdit` contract
- prose-only answers are not sufficient for coding solve

## Risks if assumptions are violated

- wrong environment -> backend instability or unusable latency
- wrong output format -> malformed candidate failures dominate
- missing local artifacts -> blocked/unverified solve path
- trying to support Windows, service backends, and local backends at once -> avoidable complexity before the first truthful path is proven
