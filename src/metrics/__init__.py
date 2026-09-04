"""Метрики для оценки пригодности генерации музыки для игровых систем.

Группы (docs/metrics.md):
  1. distributional.py — FAD, KAD (расстояния между распределениями)
  2. clap_score.py      — соответствие тексту (CLAP Score)
  3. structure.py        — структура: self-similarity, дрейф темпа/тональности
  4. game_criteria.py    — игровые критерии: качество шва, попадание в такт
  codec_floor.py          — пол-метрика (нижняя граница через round-trip кодека)
"""
