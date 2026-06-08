"""Affichage des courbes d'apprentissage (loss et accuracy, train + validation)."""

import matplotlib

matplotlib.use("Agg")  # backend sans affichage : la figure est sauvegardée en PNG
import matplotlib.pyplot as plt  # noqa: E402


def plot_learning_curves(history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Courbe 1 : loss (train ET validation)
    ax1.plot(history["train_loss"], label="training loss")
    ax1.plot(history["val_loss"], label="validation loss", linestyle="--")
    ax1.set_title("Loss")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Courbe 2 : accuracy (train ET validation)
    ax2.plot(history["train_acc"], label="training acc")
    ax2.plot(history["val_acc"], label="validation acc", linestyle="--")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
