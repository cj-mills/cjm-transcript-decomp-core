# cjm-transcript-decomp-core

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

A frontend-agnostic core for the transcript decomposition workflow — composes isolated capability workers (forced alignment, VAD, graph storage) into a headless pipeline that decomposes transcription run manifests into a VAD-aligned context-graph spine with traceable provenance, and a CLI as its first driver.

## Modules

- **`cjm_transcript_decomp_core.alignment`** — Pure forced-alignment logic (no capability calls): map FA words back to character spans in the original text, assign words to VAD chunks by timestamp, and build one text segment per VAD chunk. Extracted from the page-centric ForcedAlignmentService (Tier-1 logic).
- **`cjm_transcript_decomp_core.cli`** — The CLI driver — the decomposition core's first (and currently only) frontend.
- **`cjm_transcript_decomp_core.graph`** — Graph-spine EXTENSION + skeptical-lens verification (stage 5, CR-18 revolution 2). Decomp no longer creates a Document: it RECOMPUTES the transcription-emitted root's deterministic node ids from the consumed manifest (no search), verifies the root exists, and attaches the fine Segment spine under the existing AudioSegment nodes — PART_OF to the owning rendition, STARTS_WITH per rendition (the coarse-seam jump anchor), source-wide NEXT. Each Segment carries the audio TimeSlice ref plus per-transcriber CharSlice refs into the Transcript nodes (the D4/P10 framing, finally expressible). Commit goes through the layer's idempotent extend_graph.
- **`cjm_transcript_decomp_core.models`** — Lean data shapes for the transcript-decomposition pipeline: in-core mirrors of the forced-alignment / VAD / text DTOs (no FastHTML deps), run configuration, the committed graph-segment carrier, and the decomposition run manifest (proto-bundle).
- **`cjm_transcript_decomp_core.pipeline`** — The headless decomposition pipeline (stage 5: decomp is an EXTENDER). Load a transcription run manifest, verify the transcription-emitted graph root exists (the graph begins at transcription), then per source per pipeline-segment run VAD + per-transcriber forced alignment, build one aligned segment per VAD chunk with per-transcriber text variants, and attach the fine spine under the existing AudioSegment nodes via the layer's idempotent extend_graph — with HITL approval seams between alignment, commit, and the next source.

## API

### `cjm_transcript_decomp_core.alignment`

- `assign_words_to_chunks` _function_ — Assign each FA word to a VAD chunk by timestamp overlap.
- `build_segments_from_alignment` _function_ — Build a TextSegment per VAD chunk by grouping words by chunk assignment.
- `map_fa_words_to_text` _function_ — Map forced-alignment words back to character spans in the original text.
- `tier1_alignment_checks` _function_ — Tier-1 deterministic pre-filters for the alignment-review seam (no AI).

### `cjm_transcript_decomp_core.cli`

- `build_parser` _function_ — Build the CLI parser (subcommands: run).
- `load_capabilities` _function_ — Discover manifests + load each requested capability (default instance).
- `main` _function_ — CLI entry point (console script: `cjm-transcript-decomp-core`).
- `run_command` _function_ — Execute the `run` subcommand: extend transcription-run manifest(s) with the fine spine.

### `cjm_transcript_decomp_core.graph`

- `SourceVerification` _class_ — Skeptical-lens verification of one Source's fine-spine extension under a
- `build_extension_payload` _function_ — Build the fine-spine EXTENSION payload (pure; no capability calls).
- `resolve_root_ids` _function_ — Recompute the transcription-emitted root node ids from manifest data.
- `verify_source` _function_ — Verify a Source's committed extension via server-side AGGREGATES (D13/D19).

### `cjm_transcript_decomp_core.models`

- `DecompConfig` _class_ — Configuration for one transcript-decomposition run.
- `DecompManifest` _class_ — Durable record of one decomposition run (proto-bundle; see CR-20).
- `DecompSegment` _class_ — One fine spine segment (stage 5: shared audio-side skeleton + per-transcriber variants).
- `DecompSourceRecord` _class_ — Record of one Source whose fine spine this run committed (stage 5:
- `FAWord` _class_ — One word-level forced-alignment result (segment-local times).
- `SegmentVariant` _class_ — One transcriber's text + char range for one fine segment (stage 5).
- `TextSegment` _class_ — A text segment produced by alignment, before graph commit.
- `VADChunk` _class_ — A voice-activity time range within one pipeline segment (segment-local).
- `new_run_id` _function_ — Generate a unique, sortable decomposition run id.

### `cjm_transcript_decomp_core.pipeline`

- `build_alignment_composition` _function_ — Build the whole-source M×(VAD ∥ T×FA) composition (D8 fan-in, stage-5 variants).
- `collect_capability_info` _function_ — Record capability identity + data-DB pointers for the run manifest (provenance).
- `confirm_seam` _function_ — HITL approval seam in its cheapest viable form (log + optional CLI prompt).
- `decomp_replay_handlers` _function_ — The decomp core's replay vocabulary (DEC 426658f1, replay stays DOMAIN-OWNED).
- `decompose_source` _function_ — Decompose one source into aligned fine segments with per-transcriber variants.
- `fa_words_from_result` _function_ — Normalize a typed forced-alignment result into FA words (pure; stage 3).
- `load_source_manifest` _function_ — Load + lightly validate a transcription-core run manifest.
- `run_decomp` _function_ — Extend every source in a transcription run manifest with its fine spine.
- `submit_and_wait` _function_ — Submit one capability job, wait for it, and return its result (raise on failure).
- `vad_chunks_from_result` _function_ — Normalize a typed VAD result into segment-local VAD chunks.

## Dependencies

**Depends on:** `cjm-capability-primitives`, `cjm-context-graph-layer`, `cjm-context-graph-primitives`, `cjm-substrate`, `cjm-transcript-graph-schema`
**Used by:** `cjm-transcript-decomp-tui`
