import numpy as np
import tensorflow as tf
import os


# Load the training and test sets
def load_data(
    train_dir,
    test_dir,
    img_height=64,
    img_width=64,
    batch_size=16,
    label_mode="categorical",
):
    train_set = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode=label_mode,
    )

    test_set = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode=label_mode,
    )

    return train_set, test_set


# Build the CNN model
def build_cnn(img_height=64, img_width=64, class_names=2):
    cnn = tf.keras.models.Sequential(
        [
            ### Data Preprocessing
            tf.keras.layers.Input(shape=(img_height, img_width, 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            ### Step 1 - Convolution
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
            ### Step 2 - Pooling
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            ### Adding a second convolutional layer
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            ### Step 3 - Flattening
            tf.keras.layers.Flatten(),
            ### Step 4 - Full Connection
            tf.keras.layers.Dense(units=128, activation="relu"),
            ### Step 5 - Output Layer
            tf.keras.layers.Dense(units=len(class_names), activation="softmax"),
        ]
    )

    return cnn


# Compile and train the model
def train_cnn(cnn, train_set, test_set, epochs=20):
    cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history = cnn.fit(train_set, validation_data=test_set, epochs=epochs)

    return history


# Print training results
def evaluate_training(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    print("\nTraining Results:")
    print(f"Final Training Accuracy: {acc[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_acc[-1]:.4f}")
    print(f"Final Training Loss: {loss[-1]:.4f}")
    print(f"Final Validation Loss: {val_loss[-1]:.4f}")


def make_prediction(cnn, img_path, class_names):
    # Load and preprocess the image
    test_image = tf.keras.utils.load_img(img_path, target_size=(64, 64))
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    # Make prediction
    preds = cnn.predict(test_image)
    preds_class = np.argmax(preds)
    preds_label = class_names[preds_class]
    confidence = preds[0][preds_class]

    print(f"\nPrediction Result for {img_path}:")
    print(f"Predicted Class: {preds_label}")
    print(f"Confidence Score: {confidence:.4f}")


def main():
    # Ask user for directory inputs
    print("Welcome to the Multiclass Classification CNN Model!\n")

    training_set_path = input("Please enter the path to the training dataset: ")
    test_set_path = input("Please enter the path to the test dataset: ")

    # Check if the directories exist
    if not os.path.exists(training_set_path):
        print(f"Error: Training directory '{training_set_path}' does not exist.")
        return
    if not os.path.exists(test_set_path):
        print(f"Error: Test directory '{test_set_path}' does not exist.")
        return

    # Ask user for number of epochs, default to 20 if invalid
    epochs_input = input("Please enter the number of epochs (default is 20): ")
    try:
        epochs = int(epochs_input) if epochs_input.strip() else 20
    except ValueError:
        print("Invalid input. Using default of 20 epochs.")
        epochs = 20

    # Load the datasets
    train_set, test_set = load_data(training_set_path, test_set_path)

    # Build the CNN model
    cnn = build_cnn(img_height=64, img_width=64, class_names=train_set.class_names)

    # Train the CNN
    history = train_cnn(cnn, train_set, test_set, epochs=epochs)

    # Evaluate training results
    evaluate_training(history)

    # Ask for the full path to the prediction image after training is complete
    predict_image_path = input(
        "\nPlease enter the full path to the image you want to predict (e.g., /path/image.jpg): "
    )

    # Check if the prediction image exists
    if not os.path.exists(predict_image_path):
        print(f"Error: Image file '{predict_image_path}' does not exist.")
        return

    # Make prediction for the specified image
    make_prediction(cnn, predict_image_path, class_names=train_set.class_names)


if __name__ == "__main__":
    main()
