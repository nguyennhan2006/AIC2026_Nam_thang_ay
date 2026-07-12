# AIC2026_Nam_thang_ay

Developing a multimodal video retrieval and question-answering system capable of single-frame retrieval, visual sequence retrieval within scenes or video segments, and answering questions based on video content.

## Data section

All metadata produced in this project goes through a single versioned
contract before being stored or indexed.

- **Full documentation (features + usage guide, Vietnamese):**
  [`docs/data_section.md`](docs/data_section.md)
- **Quick reference of conventions (English):**
  [`schemas/README.md`](schemas/README.md)
- **Portable JSON Schema contract:**
  [`contracts/keyframe.schema.json`](contracts/keyframe.schema.json)

### Quick start

```bash
pip install -e .                              # Python >= 3.11, pydantic >= 2.12
python -m unittest discover -s tests -v      # 8 contract tests
python scripts/export_schemas.py             # regenerate JSON Schema contract
```

```python
from schemas import Keyframe
kf = Keyframe.model_validate_json(raw_json)  # validate pipeline output
```
