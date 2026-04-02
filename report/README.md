# Report

Final project report for the TensorStore checkpoint loading system.

## Contents

- `main.tex` — Report source
- `main.pdf` — Compiled PDF
- `chapters/` — Individual chapter source files
- `bib/` — Bibliography
- `figures/` — Report figures

## Compiling

```bash
cd report/final_report
latexmk -pdf main.tex
```

Or use the provided script:

```bash
uv run python -B ~/.agents/skills/latex-paper-en/scripts/compile.py report/final_report/main.tex
```

## Report Structure

The report is organised around the central thesis that LLM checkpoint loading should be driven by adaptive heuristics rather than a fixed backend policy. Key chapters:

- **Methodology** — Experimental platform, model ladder, cache-control procedure, benchmark design
- **Results** — Rust-layer backend comparison, Python binding overhead, vLLM integration
- **Discussion** — Why no backend is universally best, the architectural turnaround, practical lessons
- **Conclusion** — Research questions answered, contributions, limitations
