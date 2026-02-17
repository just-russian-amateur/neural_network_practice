import tensorflow as tf
import random
from pathlib import Path

'''
Файл с глобальными параметрами для модели
'''

SEED: int = 42  # фиксируем seed для воспроизводимости
NUM_WORDS: int = 4  # количество слов/классов
FILES_PER_WORD: int = 50  # количество файлов на каждый класс
TRAIN_PER_WORD: int = 25  # количество файлов для тренировки на класс
BATCH_SIZE: int = 16  # размер батча

MODEL_PATH: str = "model_best.h5"  # путь для сохранения модели
DATASET_DIR: Path = Path("dataset")  # директория для сохранения датасета
RESULT_DIR: Path = Path("result")  # директория для сохранения результатов

random.seed(SEED)
tf.random.set_seed(SEED)
