import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import argparse
import datetime

def load_dataset(dataset_dir='dataset'):
    print("Loading dataset...")

    landmarks = []
    labels = []

    # Define the expected order of classes
    expected_classes = [chr(i) for i in range(65, 91)] + ["ADD", "SPACE", "DELETE", "NOTHING"]
    available_classes = [d for d in expected_classes if os.path.isdir(os.path.join(dataset_dir, d))]

    if not available_classes:
        print("No class directories found in the dataset folder.")
        return np.array([]), np.array([]), []

    print(f"Found {len(available_classes)} classes: {available_classes}")

    # Create class to index mapping
    class_map = {cls: idx for idx, cls in enumerate(available_classes)}

    # Load data from each class directory
    for class_name in available_classes:
        class_dir = os.path.join(dataset_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]

        if len(files) == 0:
            print(f"Warning: No data files found for class {class_name}. Skipping.")
            continue

        print(f"Loading {len(files)} samples for class {class_name}")

        for file in files:
            try:
                landmark_data = np.load(os.path.join(class_dir, file))
                landmarks.append(landmark_data)
                labels.append(class_map[class_name])
            except Exception as e:
                print(f"Error loading file {file}: {e}")

    X = np.array(landmarks)
    y = np.array(labels)

    print(f"Dataset loaded: {X.shape[0]} samples, {len(available_classes)} classes")
    return X, y, available_classes

def preprocess_data(X, y):
    """Preprocess the data for training"""
    print("Preprocessing data...")

    # Flatten the landmarks array while preserving the sequence structure
    # Each hand has 21 landmarks with 3 values (x, y, z)
    X = X.reshape(X.shape[0], -1)

    # Convert labels to one-hot encoding
    y = to_categorical(y)

    # Split data manually instead of using sklearn
    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split into training (80%) and validation (20%)
    split_idx = int(0.8 * X.shape[0])
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    return X_train, X_val, y_train, y_val

def build_model(input_shape, num_classes):
    """Build and compile the model"""
    print("Building model...")

    model = Sequential([
        # First dense layer
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),

        # Second dense layer
        Dense(64, activation='relu'),
        Dropout(0.3),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

def build_lstm_model(input_shape, num_classes):
    """Build and compile an LSTM model for potentially better sequence learning"""
    print("Building LSTM model...")

    # Reshape input for LSTM (samples, timesteps, features)
    # We'll treat the 21 landmarks as timesteps, with 3 features (x,y,z) each
    reshaped_input = (21, 3)

    model = Sequential([
        # LSTM layer
        LSTM(128, input_shape=reshaped_input, return_sequences=True),
        Dropout(0.5),

        # Second LSTM layer
        LSTM(64),
        Dropout(0.3),

        # Dense hidden layer
        Dense(64, activation='relu'),
        Dropout(0.3),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=150, batch_size=32, model_type='dense'):
    """Train the model with early stopping and learning rate reduction"""
    print(f"Training {model_type} model...")

    # First, verify shapes and adjust if necessary
    num_classes = y_train.shape[1]  # Get actual number of classes from data

    # If model output doesn't match target shape, rebuild the final layer
    if model.output_shape[1] != num_classes:
        print(f"Adjusting model output layer: {model.output_shape[1]} → {num_classes}")
        # Get all layers except the last one
        if model_type == 'dense':
            # Rebuild model with correct output dimension
            input_shape = X_train.shape[1]
            model = Sequential([
                # First dense layer
                Dense(128, activation='relu', input_shape=(input_shape,)),
                Dropout(0.5),

                # Second dense layer
                Dense(64, activation='relu'),
                Dropout(0.3),

                # Output layer with correct number of classes
                Dense(num_classes, activation='softmax')
            ])
        else:  # LSTM model
            # Rebuild LSTM model with correct output dimension
            model = Sequential([
                # LSTM layer
                LSTM(128, input_shape=(21, 3), return_sequences=True),
                Dropout(0.5),

                # Second LSTM layer
                LSTM(64),
                Dropout(0.3),

                # Dense hidden layer
                Dense(64, activation='relu'),
                Dropout(0.3),

                # Output layer with correct number of classes
                Dense(num_classes, activation='softmax')
            ])

        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()

    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/{model_type}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
    ]

    # For LSTM model, reshape the input data
    if model_type == 'lstm':
        X_train = X_train.reshape(X_train.shape[0], 21, 3)
        X_val = X_val.reshape(X_val.shape[0], 21, 3)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    return history

def save_model(model, filename='sign_language_model.h5'):
    """Save the trained model"""
    print(f"Saving model to {filename}...")
    model.save(filename)
    print("Model saved successfully!")

def evaluate_model(model, X_val, y_val, model_type='dense'):
    """Evaluate the model on validation data"""
    print("Evaluating model...")

    # Check for shape mismatch and fix if needed
    num_classes = y_val.shape[1]  # Get actual number of classes from data

    # If model output doesn't match target shape, rebuild the final layer
    if model.output_shape[1] != num_classes:
        print(f"Adjusting model output layer for evaluation: {model.output_shape[1]} → {num_classes}")
        # Get all layers except the last one
        if model_type == 'dense':
            # Rebuild model with correct output dimension
            input_shape = X_val.shape[1]
            new_model = Sequential([
                # First dense layer
                Dense(128, activation='relu', input_shape=(input_shape,)),
                Dropout(0.5),

                # Second dense layer
                Dense(64, activation='relu'),
                Dropout(0.3),

                # Output layer with correct number of classes
                Dense(num_classes, activation='softmax')
            ])

            # Try to copy weights from original model for layers that match
            for i in range(len(new_model.layers) - 1):  # All except the last layer
                new_model.layers[i].set_weights(model.layers[i].get_weights())

            # Compile the model
            new_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            model = new_model
        else:  # LSTM model
            # Similar approach for LSTM model
            new_model = Sequential([
                # LSTM layer
                LSTM(128, input_shape=(21, 3), return_sequences=True),
                Dropout(0.5),

                # Second LSTM layer
                LSTM(64),
                Dropout(0.3),

                # Dense hidden layer
                Dense(64, activation='relu'),
                Dropout(0.3),

                # Output layer with correct number of classes
                Dense(num_classes, activation='softmax')
            ])

            # Try to copy weights from original model for layers that match
            for i in range(len(new_model.layers) - 1):  # All except the last layer
                new_model.layers[i].set_weights(model.layers[i].get_weights())

            # Compile the model
            new_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            model = new_model

    # For LSTM model, reshape the input data
    if model_type == 'lstm':
        X_val = X_val.reshape(X_val.shape[0], 21, 3)

    # Evaluate
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    return loss, accuracy

def plot_training_history(history):
    """Plot training history"""
    print("Plotting training history...")

    # Create directory for plots
    os.makedirs('plots', exist_ok=True)

    # Plot accuracy
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    try:
        plt.savefig('plots/training_history.png')
        print("Training plot saved to plots/training_history.png")
    except Exception as e:
        print(f"Could not save plot: {e}")
        print("Continuing without saving the plot...")

def data_augmentation(X, y, augmentation_factor=2):
    """Perform simple data augmentation by adding noise to existing samples"""
    print(f"Augmenting data by factor of {augmentation_factor}...")

    # Get original shape
    original_samples = X.shape[0]

    # Create arrays for augmented data
    X_augmented = np.zeros((original_samples * augmentation_factor, X.shape[1]))
    y_augmented = np.zeros((original_samples * augmentation_factor, y.shape[1]))

    # Copy original data
    X_augmented[:original_samples] = X
    y_augmented[:original_samples] = y

    # Create augmented samples
    for i in range(1, augmentation_factor):
        # Add random noise to create new samples
        noise = np.random.normal(0, 0.01, X.shape)
        X_augmented[i*original_samples:(i+1)*original_samples] = X + noise
        y_augmented[i*original_samples:(i+1)*original_samples] = y

    print(f"Data augmented from {original_samples} to {X_augmented.shape[0]} samples")
    return X_augmented, y_augmented

def main():
    try:
        parser = argparse.ArgumentParser(description="Train Sign Language Recognition Model")
        parser.add_argument('--epochs', type=int, default=150, help="Number of training epochs")  # Changed from 50 to 150
        parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training")
        parser.add_argument('--model-type', choices=['dense', 'lstm'], default='dense',
                          help="Model architecture to use")
        parser.add_argument('--augment', action='store_true', help="Use data augmentation")
        parser.add_argument('--augment-factor', type=int, default=2,
                          help="Data augmentation factor (if --augment is used)")

        args = parser.parse_args()

        # Load dataset
        X, y, classes = load_dataset()

        if X.shape[0] == 0:
            print("No data found. Please run data collection first.")
            return

        # Preprocess data
        X_train, X_val, y_train, y_val = preprocess_data(X, y)

        # Augment data if requested
        if args.augment:
            X_train, y_train = data_augmentation(X_train, y_train, args.augment_factor)

        # Build model
        if args.model_type == 'dense':
            model = build_model(X_train.shape[1], len(classes))
        else:  # LSTM model
            model = build_lstm_model(X_train.shape[1], len(classes))

        # Train model
        history = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_type=args.model_type
        )

        # Evaluate model
        evaluate_model(model, X_val, y_val, model_type=args.model_type)

        # Plot training history
        try:
            plot_training_history(history)
        except Exception as e:
            print(f"Error plotting training history: {e}")
            print("Continuing without plotting...")

        # Save model
        save_model(model)

        print("\nTraining complete! You can now run the application with:")
        print("python app.py --run")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()