# Transformers and sequence learning

Encoder-decoder transformer translation examples and the separate homework versions.

| Python file | Original source |
| --- | --- |
| [translation_intro/encoder.py](translation_intro/encoder.py) | `week9/encoder.py` |
| [translation_intro/decoder.py](translation_intro/decoder.py) | `week9/decoder.py` |
| [translation_intro/main.py](translation_intro/main.py) | `week9/main.py` |
| [translation_homework/encoder.py](translation_homework/encoder.py) | `Week10/Homework2/encoder.py` |
| [translation_homework/decoder.py](translation_homework/decoder.py) | `Week10/Homework2/decoder.py` |
| [translation_homework/main.py](translation_homework/main.py) | `Week10/Homework2/main.py` |
| [translation_homework/download_eng_spa.py](translation_homework/download_eng_spa.py) | `Week10/Homework2/download_eng_spa.py` |
| [translation_homework/hw2_evaluate.py](translation_homework/hw2_evaluate.py) | `Week10/Homework2/hw2_evaluate.py` |
| [translation_homework_variant/encoder.py](translation_homework_variant/encoder.py) | `Week10/HW2/encoder.py` |
| [translation_homework_variant/decoder.py](translation_homework_variant/decoder.py) | `Week10/HW2/decoder.py` |
| [translation_homework_variant/main.py](translation_homework_variant/main.py) | `Week10/HW2/main.py` |
| [translation_homework_variant/submission/main.py](translation_homework_variant/submission/main.py) | `Week10/HW2/submission/main.py` |

Original implementations and unfinished work are preserved. Notebook exports retain code cell order and code comments; lecture markdown and outputs are omitted.

The source versions are kept separate:

- `translation_intro/` comes from `week9/` and embeds its examples in code.
- `translation_homework/` comes from `Week10/Homework2/`. Its scripts expect `data.csv` and `test_translation_data_100.csv` in the working directory. The original downloader is included; CSV data and trained weights are excluded.
- `translation_homework_variant/` comes from `Week10/HW2/`. Its top-level `main.py` embeds translation pairs. The nested `submission/main.py` expects the CSV files, imports the parent encoder and decoder modules, and retains its original absolute output paths.

Run each version from its own directory so its local `encoder` and `decoder` imports resolve. The submission's import and path assumptions are preserved for later work.
