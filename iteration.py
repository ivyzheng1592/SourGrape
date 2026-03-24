import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import SourGrapeDataset
from hyper_params import HyperParams
from model import LSTMRegressor, Seq2SeqRegressor
def iterate_once() -> None:
    # Training configuration.
    hp = HyperParams()

    # Reproducibility.
    torch.manual_seed(hp.seed)

    # Dataset and dataloaders.
    dataset = SourGrapeDataset(condition=hp.condition)
    train_ratio, val_ratio, test_ratio = hp.data_split_ratio
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_ratio, val_ratio, test_ratio],
        generator=torch.Generator().manual_seed(hp.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=hp.batch_size, shuffle=False)

    # Device setup.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model selection.
    if hp.model_type == "seq2seq":
        model = Seq2SeqRegressor(
            input_size=len(dataset.vocab),
            output_len=dataset.trajectory_len,
        )
    else:
        model = LSTMRegressor(
            input_size=len(dataset.vocab),
            output_size=dataset.trajectory_len,
        )
    model.to(device)

    # Optimization setup.
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    loss_fn = nn.MSELoss()

    # Optionally resume from a checkpoint.
    if hp.resume_path:
        checkpoint = torch.load(hp.resume_path, map_location=device)
        model.load_state_dict(checkpoint)

    # Output directory for artifacts.
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop.
    history = {"train_loss": [], "val_loss": [], "test_loss": []}
    for epoch in range(1, hp.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = eval_one_epoch(model, val_loader, device, loss_fn)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch={epoch}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        # Save a checkpoint every 5 epochs.
        if epoch % 5 == 0:
            ckpt_path = out_dir / f"model_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # Final test evaluation.
    test_loss = eval_one_epoch(model, test_loader, device, loss_fn)
    history["test_loss"].append(test_loss)
    print(f"test_loss={test_loss:.6f}")

    # Save a prediction plot for a single test sample.
    if len(test_ds) > 0:
        sample_idx = 0
        sample = test_ds[sample_idx]
        x = sample["x"].unsqueeze(0).to(device)
        y = sample["y"].to(device)
        model.eval()
        with torch.no_grad():
            pred = model(x).squeeze(0).cpu().tolist()
        word = "".join(dataset.id_to_char[i] for i in sample["x"].tolist())
        pred_path = out_dir / "prediction_vs_target.png"
        save_prediction_plot(word, y.cpu().tolist(), pred, str(pred_path))

    # Save a single loss plot for this run.
    plot_path = out_dir / "loss_curve.png"
    save_loss_plot(history, str(plot_path))

    # Save artifacts.
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(dataset.vocab, f, ensure_ascii=True, indent=2)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    iterate_once()
