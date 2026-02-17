import tensorflow as tf
import os

from config import MODEL_PATH
from create_dataset import upload_dataset, create_dataset
from cnn import create_cnn
from plot_graphs import evaluate_and_save_results


if __name__ == "__main__":
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
        else:
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
