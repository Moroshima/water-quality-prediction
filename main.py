import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from typing import Literal
import os

os.environ["KERAS_BACKEND"] = "torch"

mode: Literal["train", "evaluate", "predict"] = "train"
train_epochs = 20
train_batch_size = 5

csv_path = "preprocess/data/water_quality_data_interpolated.csv"
features = ["pH值", "溶解氧", "电导率", "浑浊度", "氨氮", "耗氧量"]

df = pd.read_csv(
    (csv_path),
    parse_dates=[0],
)  # 格式化采样日期列为 Timestamp
df.set_index("采样日期", inplace=True)  # 将采样日期列设置为索引列
# 将 DataFrame 索引提取去重，得到 datetime64[s] 数组
date_array = np.unique(df.index.to_numpy().astype("datetime64[s]"))

for index, feature in enumerate(features):
    # Download and load the dataset.
    # fpath = keras.utils.get_file(
    #     "moving_mnist.npy",
    #     "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
    # )
    fpath = f"water_quality_{index}.npz"
    dataset = np.load(fpath)["arr_0"]

    # Swap the axes representing the number of frames and number of data samples.
    dataset = np.swapaxes(dataset, 0, 1)
    # We'll pick out all of the examples and use those.
    dataset = dataset[: (len(date_array) - 20 + 1), ...]
    # Add a channel dimension since the images are grayscale.
    dataset = np.expand_dims(dataset, axis=-1)

    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(0.9 * dataset.shape[0])]
    val_index = indexes[int(0.9 * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    # Normalize the data to the 0-1 range.
    train_dataset = train_dataset / 255
    val_dataset = val_dataset / 255

    # We'll define a helper function to shift the frames, where
    # `x` is frames 0 to n - 1, and `y` is frames 1 to n.
    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, 1 : data.shape[1], :, :]
        return x, y

    # Apply the processing function to the datasets.
    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)

    # Inspect the dataset.
    print(f"{feature} Training Dataset Shapes: {x_train.shape}, {y_train.shape}")
    print(f"{feature} Validation Dataset Shapes: {x_val.shape}, {y_val.shape}")

    # Construct a figure on which we will visualize the images.
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))

    # Plot each of the sequential images for one random data example.
    data_choice = np.random.choice(range(len(train_dataset)), size=1)[0]
    for idx, ax in enumerate(axes.flat):
        ax.imshow(
            np.squeeze(train_dataset[data_choice][idx]), cmap="gray", vmin=0, vmax=1
        )
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")

    # FOR DEBUG Print information and display the figure.
    # print(f"Displaying frames for {feature} example {data_choice}.")
    # plt.show()

    match mode:
        case "train":
            # Construct the input layer with no definite frame size.
            inp = layers.Input(shape=(None, *x_train.shape[2:]))

            # We will construct 3 `ConvLSTM2D` layers with batch normalization,
            # followed by a `Conv3D` layer for the spatiotemporal outputs.
            x = layers.ConvLSTM2D(
                filters=64,
                kernel_size=(5, 5),
                padding="same",
                return_sequences=True,
                activation="relu",
            )(inp)
            x = layers.BatchNormalization()(x)
            x = layers.ConvLSTM2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                return_sequences=True,
                activation="relu",
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.ConvLSTM2D(
                filters=64,
                kernel_size=(1, 1),
                padding="same",
                return_sequences=True,
                activation="relu",
            )(x)
            x = layers.Conv3D(
                filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
            )(x)

            # Next, we will build the complete model and compile it.
            model = keras.models.Model(inp, x)

            model.summary()

            model.compile(
                loss=keras.losses.binary_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=[keras.metrics.RootMeanSquaredError()],
            )

            # Define some callbacks to improve training.
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10
            )
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=5
            )

            # Define modifiable training hyperparameters.
            epochs = train_epochs
            batch_size = train_batch_size

            # Fit the model to the training data.
            history = model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping, reduce_lr],
            )

            print(f"{feature} Model training history: {history.history}")

            np.save(
                (f"{index}_model_training_history.npy"),
                history.history,
            )

            # Save the model to disk.
            model.save(
                (f"water_quality_{index}_prediction.keras"),
            )  # The file needs to end with the .keras extension

        case "evaluate":
            model = keras.models.load_model(f"water_quality_{index}_prediction.keras")

            val_loss, val_rsme = model.evaluate(x_val, y_val)

            y_true = y_val.reshape(y_val.shape[0], -1)
            y_pred_raw = model.predict(x_val)
            y_pred = y_pred_raw.reshape(y_pred_raw.shape[0], -1)
            metric = keras.metrics.R2Score()
            metric.update_state(y_true, y_pred)
            val_r2 = metric.result()
            print(
                f"{feature} Validation Loss: {val_loss}, RSME: {val_rsme}, R2 Score: {val_r2}"
            )
        case "predict":
            model = keras.models.load_model(f"water_quality_{index}_prediction.keras")

            # Select a random example from the validation dataset.
            example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]

            # Pick the first/last ten frames from the example.
            frames = example[:10, ...]
            original_frames = example[10:, ...]

            # Predict a new set of 10 frames.
            for _ in range(10):
                # Extract the model's prediction and post-process it.
                new_prediction = model.predict(np.expand_dims(frames, axis=0))
                new_prediction = np.squeeze(new_prediction, axis=0)
                predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

                # Extend the set of prediction frames.
                frames = np.concatenate((frames, predicted_frame), axis=0)

            plt.rcParams["font.sans-serif"] = ["Noto Sans SC"]
            # Construct a figure for the original and new frames.
            fig, axes = plt.subplots(3, 10, figsize=(20, 6))

            # Plot the original frames.
            for idx, ax in enumerate(axes[0]):
                ax.imshow(np.squeeze(example[idx]), cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"Input {idx + 1}")
                ax.axis("off")
            for idx, ax in enumerate(axes[1]):
                ax.imshow(np.squeeze(example[idx + 10]), cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"GT {idx + 11}")
                ax.axis("off")

            # Plot the new frames.
            new_frames = frames[10:, ...]
            for idx, ax in enumerate(axes[2]):
                ax.imshow(np.squeeze(new_frames[idx]), cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"Prediction {idx + 11}")
                ax.axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出顶部空间

            # 添加大标题（使用suptitle而不是title）
            plt.suptitle(
                f"Water Quality Prediction for {feature}",
                fontsize=16,
                y=0.98,  # 控制标题位置
            )

            # Save the figure.
            plt.tight_layout()
            plt.savefig(
                f"graphs/water_quality_{index}_{feature}_prediction.png",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close()

    print("finished.")
