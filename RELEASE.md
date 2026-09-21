# Release record

Source commit: `8eb6a0ca288793e187509f16027efc0f0a9aad5b`.
Prepared: 21 September 2026.

Included: every tracked Python module and test, configs, schemas, pyproject.toml, uv.lock,
both runtime annotation rubrics, and the teacher-selection specification referenced by the code.

Excluded: Git history, agent instructions, development logs, website deployment files,
raw data, generated outputs, credentials, model weights and local caches.

Portability edits:

- Replaced two personal workbook defaults with relative paths under `data/`.
- Changed one test to locate its source file relative to the checkout.
- Added plain-language release documentation and a data/cache ignore file.

All other implementation files and the runtime rubrics retain their original content.
Earlier experimental modules are kept so shared dependencies and tests remain intact.
The README distinguishes those modules from the retained model.

No licence has been assigned by this packaging step.

## Verification

- All 196 selected original files are present; only the two files listed above differ.
- Ruff passed for source and tests.
- The README's portable test selection passed: 146 tests.
- A full-suite attempt was stopped after failures due to missing Git history and retained private
  run artefacts. Those prerequisites are not bundled and the suite is not reported as passing.
- All Python files parse. The release scan found no row-data file formats, personal home paths,
  private-key headers or long Hugging Face/OpenAI-style credential literals.
- No remote repository was created or published.
