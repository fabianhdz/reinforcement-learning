# Transformers

Encoder-decoder models for English-Spanish translation.

- [Basic translation](translation_intro/): self-contained training examples.
- [CSV translation](translation_homework/): dataset download, training, and evaluation scripts.
- [Translation variants](translation_homework_variant/): additional training configurations.

Run each model from its own directory so the local encoder and decoder imports resolve. CSV-based scripts expect `data.csv` and `test_translation_data_100.csv`. The extended variant in `translation_homework_variant/submission/main.py` also needs its parent modules on the Python path and its output paths configured.
