import tensorflow as tf
import numpy as np
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Optional


from config import NUM_WORDS, RESULT_DIR


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
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='train acc')
        plt.plot(history.history['val_accuracy'], label='val acc')
        plt.title('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.legend()
        plt.savefig("./graphs/loss.png")

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
    plt.savefig("./graphs/confusion_matrix.png")

    # Сохраняем файлы по предсказанным папкам
    save_classified_files(test_files, y_pred, selected_labels)

    # Ошибки I и II рода
    print("\nОшибки I и II рода:")
    for i in range(NUM_WORDS):
        FP = cm[:, i].sum() - cm[i, i]
        FN = cm[i, :].sum() - cm[i, i]
        print(f"Класс '{selected_labels[i]}': FP={FP}, FN={FN}")
        