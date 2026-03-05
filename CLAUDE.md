# CLAUDE.md

Claude should use `AGENTS.md` in this repository as the canonical operating guide.

## Required Behavior
When creating or updating analyses in `analysis/*`, follow the mandatory workflow in `AGENTS.md`:
1. `analysis_standards.py init` (for new analyses)
2. update `analysis_manifest.json` and `key_findings.txt`
3. `analysis_standards.py validate`
4. `analysis_standards.py score --apply-gate-penalty`
5. ensure `result_card.md` is present

## Required Implication Lines
Every `key_findings.txt` must include:
- `business_implication: ...`
- `scientific_implication: ...`

## References
- `AGENTS.md`
- `ANALYSIS_STANDARDS.md`
- `standards/analysis_manifest_template.json`

