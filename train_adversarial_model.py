import os
import numpy as np
import tensorflow as tf
import cv2
from keras._tf_keras.keras.callbacks import CSVLogger
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
NUM_EPOCHS = 100
INPUT_SHAPE = (64, 64, 1)
NUM_CLASSES = 7
BASE_PATH = './trained_models/'
FER_PATH = 'Datasets/FER-2013/train'
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
label_to_index = {label: i for i, label in enumerate(EMOTION_LABELS)}

def preprocess_input(x):
    x = x.astype('float32') / 255.0
    x = x - 0.5
    x = x * 2.0
    return x

def load_fer2013_from_folders(base_path):
    images, labels = [], []
    for emotion in EMOTION_LABELS:
        folder = os.path.join(base_path, emotion)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if not file.lower().endswith(('.jpg', '.png')):
                continue
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            img = preprocess_input(img)
            images.append(np.expand_dims(img, axis=-1))
            labels.append(label_to_index[emotion])

    images = np.array(images)
    labels = np.array(labels)
    y = np.zeros((labels.size, NUM_CLASSES))
    y[np.arange(labels.size), labels] = 1
    return train_test_split(images, y, test_size=0.2, stratify=labels, random_state=42)

from cnn import mini_XCEPTION

def main():
    x_train, x_val, y_train, y_val = load_fer2013_from_folders(FER_PATH)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    model = mini_XCEPTION(INPUT_SHAPE, NUM_CLASSES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    os.makedirs(BASE_PATH, exist_ok=True)
    log_file = os.path.join(BASE_PATH, 'fer2013_adv_training.log')
    csv_logger = CSVLogger(log_file, append=False)

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    classifier = TensorFlowV2Classifier(
        model=model,
        nb_classes=NUM_CLASSES,
        input_shape=INPUT_SHAPE,
        loss_object=loss_object,
        clip_values=(-1.0, 1.0)
    )

    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=3e-2,
        eps_step=5e-3,
        max_iter=40
    )

    steps_per_epoch = len(x_train) // BATCH_SIZE
    best_val_loss = np.inf

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)

        epoch_loss = 0.0
        epoch_acc = 0.0
        for step in range(steps_per_epoch):
            x_batch, y_batch = next(gen)

            num_adv = BATCH_SIZE // 4

            x_batch_adv = attack.generate(x_batch[:num_adv])

            x_combined = np.concatenate([x_batch[num_adv:], x_batch_adv], axis=0)
            y_combined = np.concatenate([y_batch[num_adv:], y_batch[:num_adv]], axis=0)

            metrics = classifier._model.train_on_batch(x_combined, y_combined)

            epoch_loss += metrics[0]
            epoch_acc += metrics[1]

            if (step + 1) % 50 == 0 or (step + 1) == steps_per_epoch:
                print(f" Step {step+1}/{steps_per_epoch} - loss: {metrics[0]:.4f} - acc: {metrics[1]:.4f}")

        epoch_loss /= steps_per_epoch
        epoch_acc /= steps_per_epoch
        print(f"Epoch {epoch+1} summary: loss = {epoch_loss:.4f}, accuracy = {epoch_acc:.4f}")

        val_loss, val_acc = classifier._model.evaluate(x_val, y_val, verbose=0)
        print(f" Validation - loss: {val_loss:.4f} - acc: {val_acc:.4f}")

        csv_logger.on_epoch_end(epoch, {'loss': epoch_loss, 'accuracy': epoch_acc, 'val_loss': val_loss, 'val_accuracy': val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(os.path.join(BASE_PATH, 'mini_xception_adv_best.h5'))
            print(f" Modelo salvo no epoch {epoch+1} com val_loss {val_loss:.4f}")


    print("\nAvaliando no conjunto limpo:")
    clean_preds = classifier.predict(x_val)
    clean_acc = np.mean(np.argmax(clean_preds, axis=1) == np.argmax(y_val, axis=1))
    print(f"Acurácia limpa: {clean_acc * 100:.2f}%")

    print("\nAvaliando com PGD adversarial:")
    x_val_adv = attack.generate(x_val, y_val)
    adv_preds = classifier.predict(x_val_adv)
    adv_acc = np.mean(np.argmax(adv_preds, axis=1) == np.argmax(y_val, axis=1))
    print(f"Acurácia adversarial: {adv_acc * 100:.2f}%")

    model.save(os.path.join(BASE_PATH, 'mini_xception_adv_final.h5'))
    print(f"\nModelo salvo em: {BASE_PATH}")

if __name__ == "__main__":
    main()
