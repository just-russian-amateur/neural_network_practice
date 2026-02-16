import tensorflow as tf
import numpy as np
import os
import random
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Tuple, Optional

# ==============================
# Глобальные параметры
# ==============================
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

# ==============================
# Подготовка и сохранение датасета
# ==============================
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

        # делим на тренировочные и тестовые части
        train_part = files[:TRAIN_PER_WORD]
        test_part = files[TRAIN_PER_WORD:TRAIN_PER_WORD + TRAIN_PER_WORD]

        # добавляем пути к файлам в списки
        train_files.extend([str(DATASET_DIR / label / f"{label}{i+1}.wav") for i in range(TRAIN_PER_WORD)])
        test_files.extend([str(DATASET_DIR / label / f"{label}{i+1+TRAIN_PER_WORD}.wav") for i in range(TRAIN_PER_WORD)])

        train_labels.extend([label_to_index[label]] * TRAIN_PER_WORD)
        test_labels.extend([label_to_index[label]] * TRAIN_PER_WORD)

    print(f"Train size: {len(train_files)}, Test size: {len(test_files)}")
    return (train_files, train_labels), (test_files, test_labels), selected_labels

# ==============================
# Аудио обработка
# ==============================
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
        num_mel_bins, num_spectrogram_bins, sample_rate, 20.0, 4000.0
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

# ==============================
# Dataset
# ==============================
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
        print("Input shape:", input_shape)

    return input_shape, train_ds, test_ds

# ==============================
# CNN модель
# ==============================
def create_cnn(input_shape: Tuple[int,int,int]) -> tf.keras.Model:
    """
    Создает и компилирует CNN модель для классификации.

    Args:
        input_shape: форма входных данных (H, W, C)

    Returns:
        model: tf.keras.Model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(8, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(NUM_WORDS)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()
    return model

# ==============================
# Сохраняем предсказанные файлы
# ==============================
def save_classified_files(test_files: List[str], y_pred: List[int], selected_labels: List[str]) -> None:
    """
    Копирует тестовые файлы в папки result/<predicted_label> согласно предсказаниям модели.

    Args:
        test_files: список путей к тестовым файлам
        y_pred: предсказанные классы
        selected_labels: список выбранных классов
    """
    RESULT_DIR.mkdir(exist_ok=True)
    for label in selected_labels:
        (RESULT_DIR / label).mkdir(exist_ok=True)

    for file_path, pred_idx in zip(test_files, y_pred):
        pred_label = selected_labels[pred_idx]
        filename = Path(file_path).name
        shutil.copy(file_path, RESULT_DIR / pred_label / filename)

# ==============================
# Оценка и результат
# ==============================
def evaluate_and_save_results(model: tf.keras.Model, 
                              test_ds: tf.data.Dataset, 
                              test_files: List[str], 
                              selected_labels: List[str], 
                              history: Optional[tf.keras.callbacks.History]=None) -> None:
    """
    Оценивает модель на тестовой выборке, выводит графики, confusion matrix, ошибки I/II рода,
    и копирует файлы в result/<predicted_label>.

    Args:
        model: обученная модель
        test_ds: tf.data.Dataset для тестирования
        test_files: список путей к тестовым файлам
        selected_labels: список выбранных классов
        history: история обучения (для графиков), по умолчанию None
    """
    # Графики accuracy/loss
    if history is not None:
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'], label='train acc')
        plt.plot(history.history['val_accuracy'], label='val acc')
        plt.title('Accuracy')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.legend()
        plt.show()

    # Предсказания
    y_true, y_pred = [], []
    for spectrograms, labels in test_ds:
        logits = model.predict(spectrograms, verbose=0)
        preds = np.argmax(logits, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=selected_labels, yticklabels=selected_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Сохраняем файлы по предсказанным папкам
    save_classified_files(test_files, y_pred, selected_labels)

    # Ошибки I и II рода
    print("\nОшибки I и II рода:")
    for i in range(NUM_WORDS):
        FP = cm[:, i].sum() - cm[i, i]
        FN = cm[i, :].sum() - cm[i, i]
        print(f"Класс '{selected_labels[i]}': FP={FP}, FN={FN}")

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)

    while True:
        mode = 0
        while mode not in [1, 2]:
            try:
                mode = int(input("Выберите режим:\n1. Обучение модели\n2. Тестирование модели\nВаш выбор: "))
            except:
                continue

        if mode == 1:
            # Обучение
            train_set, test_set, selected_labels = upload_dataset()
            input_shape, train_ds, test_ds = create_dataset(train_set, test_set)
            model = create_cnn(input_shape)

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )

            history = model.fit(
                train_ds, validation_data=test_ds,
                epochs=25,
                callbacks=[early_stop]
            )

            # Сохраняем лучшую модель
            model.save(MODEL_PATH)
            print(f"Модель сохранена: {MODEL_PATH}")

            evaluate_and_save_results(model, test_ds, test_set[0], selected_labels, history)
            break

        else:  # mode == 2
            # Тестирование
            if not os.path.exists(MODEL_PATH):
                print(f"Модель {MODEL_PATH} не найдена! Сначала обучите модель.")
                continue  # повторный выбор режима

            train_set, test_set, selected_labels = upload_dataset()
            _, _, test_ds = create_dataset(train_set, test_set)

            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Модель {MODEL_PATH} загружена")

            evaluate_and_save_results(model, test_ds, test_set[0], selected_labels)
            break
