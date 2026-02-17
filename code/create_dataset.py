import tensorflow as tf
import random
from pathlib import Path
import shutil
from typing import List, Tuple

from config import NUM_WORDS, FILES_PER_WORD, DATASET_DIR, TRAIN_PER_WORD, SEED, BATCH_SIZE


def upload_dataset() -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], List[str]]:
    """
    Загружает и подготавливает датасет.
    Случайно выбирает NUM_WORDS классов, сохраняет по FILES_PER_WORD файлов каждого класса в dataset/<label>,
    делит на тренировочный и тестовый наборы.

    Returns:
        train_set: Tuple[List[str], List[int]] - пути к тренировочным файлам и их метки
        test_set: Tuple[List[str], List[int]] - пути к тестовым файлам и их метки
        selected_labels: List[str] - список выбранных классов
    """
    data_dir: Path = Path("data/mini_speech_commands_extracted/mini_speech_commands")

    # если датасета нет, скачиваем
    if not data_dir.exists():
        print("Скачивание датасета...")
        tf.keras.utils.get_file(
            "mini_speech_commands.zip",
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            cache_dir=".",
            cache_subdir="data"
        )

    all_labels: List[str] = sorted([item.name for item in data_dir.iterdir() if item.is_dir()])
    selected_labels: List[str] = random.sample(all_labels, NUM_WORDS)
    print(f"Выбранные классы: {selected_labels}")

    label_to_index: dict = {label: i for i, label in enumerate(selected_labels)}

    # создаём папки dataset/<label>
    for label in selected_labels:
        (DATASET_DIR / label).mkdir(parents=True, exist_ok=True)

    train_files, test_files = [], []
    train_labels, test_labels = [], []

    for label in selected_labels:
        files: List[Path] = list((data_dir / label).glob("*.wav"))
        files = random.sample(files, FILES_PER_WORD)

        # сохраняем копии файлов в dataset/<label> с новым именем
        for i, f in enumerate(files, 1):
            target_path: Path = DATASET_DIR / label / f"{label}{i}.wav"
            shutil.copy(f, target_path)

        # добавляем пути к файлам в списки
        train_files.extend([str(DATASET_DIR / label / f"{label}{i + 1}.wav") for i in range(TRAIN_PER_WORD)])
        test_files.extend([str(DATASET_DIR / label / f"{label}{i + 1 + TRAIN_PER_WORD}.wav") for i in range(TRAIN_PER_WORD)])

        train_labels.extend([label_to_index[label]] * TRAIN_PER_WORD)
        test_labels.extend([label_to_index[label]] * TRAIN_PER_WORD)

    print(f"Train size: {len(train_files)}, Test size: {len(test_files)}")
    return (train_files, train_labels), (test_files, test_labels), selected_labels


def load_audio(file_path: str, label: int) -> Tuple[tf.Tensor, int]:
    """
    Загружает WAV-файл и возвращает его waveform фиксированной длины + метку.

    Args:
        file_path: путь к WAV-файлу
        label: метка класса файла

    Returns:
        waveform: tf.Tensor с аудио сигналом длины 16000
        label: метка класса
    """
    audio_binary: tf.Tensor = tf.io.read_file(file_path)
    waveform, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)
    desired_length: int = 16000
    waveform_length: tf.Tensor = tf.shape(waveform)[0]
    waveform = tf.cond(
        waveform_length < desired_length,
        lambda: tf.pad(waveform, [[0, desired_length - waveform_length]]),
        lambda: waveform[:desired_length]
    )
    return waveform, label


def get_mel_spectrogram(waveform: tf.Tensor) -> tf.Tensor:
    """
    Преобразует waveform в mel spectrogram.

    Args:
        waveform: tf.Tensor аудио сигнала

    Returns:
        mel_spectrogram: tf.Tensor [time, freq, 1]
    """
    sample_rate: int = 16000
    stft = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(stft)
    num_spectrogram_bins = tf.shape(spectrogram)[-1]
    num_mel_bins = 40
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        sample_rate,
        20.0,
        4000.0
    )
    mel_spectrogram = tf.tensordot(spectrogram, mel_weight_matrix, 1)
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mel_spectrogram = mel_spectrogram[..., tf.newaxis]

    return mel_spectrogram


def preprocess(file_path: str, label: int) -> Tuple[tf.Tensor, int]:
    """
    Объединяет загрузку аудио и преобразование в mel spectrogram.

    Args:
        file_path: путь к WAV-файлу
        label: метка класса

    Returns:
        mel_spectrogram: tf.Tensor спектрограммы
        label: метка класса
    """
    waveform, label = load_audio(file_path, label)
    mel_spec = get_mel_spectrogram(waveform)

    return mel_spec, label


def create_dataset(train_set: Tuple[List[str], List[int]], 
                   test_set: Tuple[List[str], List[int]]) -> Tuple[Tuple[int,int,int], tf.data.Dataset, tf.data.Dataset]:
    """
    Создает tf.data.Dataset для обучения и тестирования

    Args:
        train_set: кортеж (пути к файлам, метки)
        test_set: кортеж (пути к файлам, метки)

    Returns:
        input_shape: форма входа модели (H, W, C)
        train_ds: Dataset для обучения
        test_ds: Dataset для тестирования
    """
    train_files, train_labels = train_set
    test_files, test_labels = test_set

    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_files), tf.constant(train_labels)))
    test_ds = tf.data.Dataset.from_tensor_slices((tf.constant(test_files), tf.constant(test_labels)))

    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.shuffle(100, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # выводим форму входа для модели
    for spec, _ in train_ds.take(1):
        input_shape = spec.shape[1:]
        print(f"Input shape: {input_shape}")

    return input_shape, train_ds, test_ds
