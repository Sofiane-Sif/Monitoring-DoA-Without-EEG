
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger


def plot_pred_hist(lstm_model, device, test_loader, test_labels, y_unscale, criterion, seq_length_sec, fig_path, files_suffix):
    logger.info("Plotting predictions histogram")

    #TODO do it without test_labels

    lstm_model.eval()
    with torch.no_grad():

        test_losses = list()
    
        scaled_preds = torch.tensor([], device=device)

        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            batch_pred = lstm_model(sequences)

            scaled_preds = torch.cat((scaled_preds, batch_pred))

        test_loss = criterion(scaled_preds.flatten(), test_labels)

        y_true = np.round(y_unscale(np.array(test_labels.cpu()))).astype(int)
        y_pred = np.round(y_unscale(np.array(scaled_preds.cpu()).flatten())).astype(int)


        data = pd.DataFrame({"ytrue": y_true, "ypred": y_pred})

        fig, (ax1, ax2) = plt.subplots(figsize=(12, 4), ncols=2)
        sns.histplot(data=pd.melt(data), ax=ax1, x="value", hue="variable", bins=30)
        sns.histplot(data=data, x="ytrue", y="ypred", ax=ax2, bins=30)
        ax2.plot([0, 100], [0, 100], ls="dotted", alpha=0.7)
        fig.tight_layout()

        fig.savefig(fig_path.joinpath(Path(f"lstm_pred_plot_{files_suffix}.png")), dpi=300)


def plot_sessions_preds(lstm_model, device, df_test, seq_length_sec, fig_path, files_suffix):
    logger.info("Plotting test sessions predictions")

    lstm_model.eval()
    with torch.no_grad():

        i = 0
        examples_square_count = 2
        fig, axes = plt.subplots(nrows=examples_square_count, ncols=examples_square_count, figsize=(16, 8))

        for _, sg in df_test.groupby("rec_id"):
            if (len(sg.index) < seq_length_sec) or (i >= examples_square_count**2):
                break

            bis_preds = [np.nan] * (seq_length_sec-1)

            tensor_list = []
            for j in range(len(sg.index) - seq_length_sec + 1):
                subgroup = sg.iloc[j:j+seq_length_sec]
                subgroup_tensor = torch.tensor((subgroup[["CO₂fe","CO₂mi","FC","PNIm","PNIs","PNIs", "SpO₂"]]).values).to(device)
                tensor_list.append(subgroup_tensor)

            model_input = torch.stack(tensor_list)
            predictions = lstm_model(model_input)

            bis_preds += [p.item() for p in predictions]

            ax = axes[i%examples_square_count][i//examples_square_count]

            ax.plot(sg["BIS"].values, c="black", label="True BIS")
            ax.plot(sg["CO₂fe"].values, linestyle='--',c="green", label="CO₂fe")
            ax.plot(sg["CO₂mi"].values, linestyle='--',c="gray", label="CO₂mi")
            ax.plot(sg["FC"].values, linestyle='--',c="yellow", label="FC")
            ax.plot(sg["PNIm"].values, linestyle='--',c="purple", label="PNIm")
            ax.plot(sg["PNIs"].values, linestyle='--',c="blue", label="PNIs")
            ax.plot(sg["PNId"].values, linestyle='--',c="orange", label="PNId")
            ax.plot(sg["SpO₂"].values, linestyle='--',c="pink", label="SpO₂")
            ax.plot(bis_preds, c="red", label="Predicted BIS")
            ax.legend()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("scaled values")
            i+=1

        plt.title("Predicted BIS values")
        plt.tight_layout()
        plt.savefig(fig_path.joinpath(Path(f"lstm_random_preds_{files_suffix}.png")), dpi=300)


def plot_learning_curves(y_unscale, train_losses, val_losses, fig_path, files_suffix):
    plt.figure(figsize=(6, 4))
    plt.plot(y_unscale(np.array(train_losses)))
    plt.plot(y_unscale(np.array(val_losses)), label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (MAE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path.joinpath(Path(f"lstm_training_validation_loss_{files_suffix}.png")))
    plt.close()