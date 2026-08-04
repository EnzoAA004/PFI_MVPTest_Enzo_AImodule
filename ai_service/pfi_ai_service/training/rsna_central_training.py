from __future__ import annotations

import hashlib, json, math, os, random, shutil, subprocess, sys, time, zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
import timm


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2026
    image_size: int = 224
    crop_size: int = 256
    batch_size: int = 32
    num_workers: int = 2
    max_epochs: int = 12
    patience: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    model_name: str = "efficientnet_b0"
    pretrained: bool = True


CLASS_NAMES = ["Normal/Mild", "Moderate", "Severe"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifests(split_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_path = split_root / "train_manifest.csv"
    validation_path = split_root / "validation_manifest.csv"
    summary_path = split_root / "split_summary.json"
    required = [train_path, validation_path, summary_path, split_root / "internal_test_manifest.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Faltan artefactos del Notebook 54:\n- " + "\n- ".join(missing))
    train = pd.read_csv(train_path, dtype={"study_id": str, "series_id": str})
    validation = pd.read_csv(validation_path, dtype={"study_id": str, "series_id": str})
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("leakageDetected") is not False or summary.get("officialTestAccessed") is not False:
        raise RuntimeError("El split del Notebook 54 no está aprobado.")
    if set(train.study_id) & set(validation.study_id):
        raise RuntimeError("Fuga de study_id entre train y validation.")
    return train, validation, summary


def ensure_local_subset(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    local_root: Path,
    competition: str,
    kaggle_token: str,
) -> None:
    series_ids = set(train.series_id.astype(str)) | set(validation.series_id.astype(str))
    train_images = local_root / "train_images"
    coordinates_csv = local_root / "train_label_coordinates.csv"

    def ready() -> bool:
        if not coordinates_csv.is_file() or not train_images.is_dir():
            return False
        for series_id in list(series_ids)[:50]:
            matches = list(train_images.glob(f"*/{series_id}"))
            if not matches or not any(matches[0].glob("*.dcm")):
                return False
        return True

    if ready():
        return

    free = shutil.disk_usage("/content").free
    if free < 38 * 1024**3:
        raise RuntimeError(f"Espacio local insuficiente: {free/1024**3:.1f} GiB libres.")

    download_root = Path("/content/rsna_kaggle_training")
    local_root.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    os.environ["KAGGLE_API_TOKEN"] = kaggle_token
    try:
        subprocess.check_call([
            executable, "competitions", "download", competition,
            "--path", str(download_root), "--force",
        ])
    finally:
        os.environ.pop("KAGGLE_API_TOKEN", None)

    archives = sorted(download_root.glob("*.zip"))
    if not archives:
        raise RuntimeError("No se encontró el ZIP descargado.")
    archive_path = archives[0]
    root_resolved = local_root.resolve()
    extracted = 0
    started = time.time()

    def selected(name: str) -> Path | None:
        normalized = PurePosixPath(name)
        if normalized.is_absolute() or ".." in normalized.parts:
            raise RuntimeError(f"Ruta insegura en ZIP: {name!r}")
        if normalized.name == "train_label_coordinates.csv":
            return Path(normalized.name)
        if "train_images" not in normalized.parts:
            return None
        index = normalized.parts.index("train_images")
        parts = normalized.parts[index:]
        if len(parts) < 4 or parts[2] not in series_ids:
            return None
        return Path(*parts)

    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative = selected(member.filename)
            if relative is None:
                continue
            destination = (local_root / relative).resolve()
            if destination != root_resolved and root_resolved not in destination.parents:
                raise RuntimeError("Extracción fuera del root local.")
            if destination.exists() and destination.stat().st_size == member.file_size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            extracted += 1
            if extracted % 5000 == 0:
                print(f"Archivos extraídos: {extracted}")

    shutil.rmtree(download_root, ignore_errors=True)
    if not ready():
        raise RuntimeError("El subconjunto local quedó incompleto.")
    print({"selectedFilesExtracted": extracted, "minutes": round((time.time()-started)/60, 2)})


def attach_coordinates(manifest: pd.DataFrame, local_root: Path) -> pd.DataFrame:
    coordinates = pd.read_csv(
        local_root / "train_label_coordinates.csv",
        dtype={"study_id": str, "series_id": str},
    )
    rows = coordinates.loc[
        coordinates.condition.astype(str).str.strip().eq("Spinal Canal Stenosis")
    ].copy()
    rows["instance_number"] = rows.instance_number.astype(int)
    rows = rows.sort_values(
        ["study_id", "series_id", "level", "instance_number"]
    ).drop_duplicates(["study_id", "series_id", "level"])
    samples = manifest.merge(
        rows[["study_id", "series_id", "level", "instance_number", "x", "y"]],
        on=["study_id", "series_id", "level"],
        how="left",
        validate="one_to_one",
    )
    if samples.instance_number.isna().any():
        raise RuntimeError(f"Faltan {int(samples.instance_number.isna().sum())} coordenadas.")
    samples["severity_code"] = samples.severity.map(CLASS_TO_INDEX).astype(int)
    samples["local_series_path"] = samples.apply(
        lambda row: str(local_root / "train_images" / row.study_id / row.series_id),
        axis=1,
    )
    if not samples.local_series_path.map(lambda value: Path(value).is_dir()).all():
        raise RuntimeError("Faltan series locales.")
    return samples.reset_index(drop=True)


def _files(series_path: Path) -> list[Path]:
    files = list(series_path.glob("*.dcm"))
    if not files:
        raise RuntimeError(f"Serie sin DICOM: {series_path}")
    return sorted(files, key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)


def _read(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(path)
    image = dataset.pixel_array.astype(np.float32)
    return image * float(getattr(dataset, "RescaleSlope", 1.0)) + float(getattr(dataset, "RescaleIntercept", 0.0))


def _crop(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    h, w = image.shape
    half = size // 2
    left, top = int(round(x)) - half, int(round(y)) - half
    right, bottom = left + size, top + size
    pads = (max(0, -left), max(0, -top), max(0, right-w), max(0, bottom-h))
    if any(pads):
        image = np.pad(image, ((pads[1], pads[3]), (pads[0], pads[2])), mode="edge")
        left, right = left + pads[0], right + pads[0]
        top, bottom = top + pads[1], bottom + pads[1]
    return image[top:bottom, left:right]


def _uint8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99])
    high = max(high, low + 1)
    return (np.clip((image-low)/(high-low), 0, 1) * 255).round().astype(np.uint8)


def build_cache(samples: pd.DataFrame, cache_root: Path, split: str, cfg: TrainConfig) -> None:
    target = cache_root / split
    target.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(samples.iterrows(), total=len(samples), desc=f"cache {split}"):
        level = str(row.level).replace("/", "_")
        destination = target / f"{row.study_id}_{row.series_id}_{level}.npy"
        if destination.is_file():
            continue
        files = _files(Path(row.local_series_path))
        lookup = {int(path.stem): index for index, path in enumerate(files) if path.stem.isdigit()}
        instance = int(row.instance_number)
        center = lookup.get(instance)
        if center is None:
            nearest = min(lookup, key=lambda value: abs(value-instance))
            center = lookup[nearest]
        indices = [max(0, center-1), center, min(len(files)-1, center+1)]
        channels = []
        for index in indices:
            image = _uint8(_crop(_read(files[index]), row.x, row.y, cfg.crop_size))
            tensor = torch.from_numpy(image).float()[None, None]
            resized = F.interpolate(tensor, (cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False)[0, 0]
            channels.append(resized.clamp(0, 255).byte().numpy())
        np.save(destination, np.stack(channels), allow_pickle=False)


class CachedDataset(Dataset):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, frame: pd.DataFrame, cache_root: Path, split: str, augment: bool):
        self.frame, self.root, self.split, self.augment = frame.reset_index(drop=True), cache_root, split, augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        level = str(row.level).replace("/", "_")
        image = torch.from_numpy(np.load(self.root / self.split / f"{row.study_id}_{row.series_id}_{level}.npy", allow_pickle=False)).float() / 255
        if self.augment:
            if random.random() < 0.5:
                image = torch.flip(image, [2])
            if random.random() < 0.3:
                image = TF.rotate(image, random.uniform(-5, 5))
        return (image-self.mean)/self.std, int(row.severity_code)


def loaders(train: pd.DataFrame, validation: pd.DataFrame, cache_root: Path, cfg: TrainConfig):
    train_ds = CachedDataset(train, cache_root, "train", True)
    validation_ds = CachedDataset(validation, cache_root, "validation", False)
    counts = train.severity_code.value_counts().reindex(range(3), fill_value=0).sort_index()
    weights = train.severity_code.map({i: 1/math.sqrt(c) for i, c in counts.items()}).to_numpy()
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
    train_loader = DataLoader(train_ds, cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, validation_loader, counts


def metrics(targets, predictions):
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "severe_recall": float(recall_score(targets, predictions, labels=[2], average="macro", zero_division=0)),
    }


def train(
    train_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    cache_root: Path,
    checkpoint_root: Path,
    run_root: Path,
    manifest_hashes: dict,
    cfg: TrainConfig,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere GPU T4 o superior.")
    device = torch.device("cuda")
    set_seed(cfg.seed)
    train_loader, validation_loader, counts = loaders(train_samples, validation_samples, cache_root, cfg)
    try:
        model = timm.create_model(cfg.model_name, pretrained=cfg.pretrained, in_chans=3, num_classes=3)
    except Exception:
        model = timm.create_model(cfg.model_name, pretrained=False, in_chans=3, num_classes=3)
    model.to(device)
    class_weights = torch.tensor((counts.sum()/(3*counts.astype(float))).to_numpy(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler("cuda")
    history, best_score, best_epoch, stale = [], -1.0, 0, 0
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    def epoch(loader, training):
        model.train(training)
        loss_sum, targets, predictions = 0.0, [], []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for images, labels in tqdm(loader, leave=False):
                images, labels = images.to(device), labels.to(device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, labels)
                if training:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                loss_sum += float(loss.detach().cpu()) * len(labels)
                targets += labels.detach().cpu().tolist()
                predictions += logits.argmax(1).detach().cpu().tolist()
        result = metrics(targets, predictions)
        result["loss"] = loss_sum / len(loader.dataset)
        return result

    for number in range(1, cfg.max_epochs + 1):
        train_result = epoch(train_loader, True)
        validation_result = epoch(validation_loader, False)
        score = 0.5*validation_result["macro_f1"] + 0.3*validation_result["severe_recall"] + 0.2*validation_result["balanced_accuracy"]
        scheduler.step(score)
        row = {"epoch": number, **{f"train_{k}": v for k,v in train_result.items()}, **{f"validation_{k}": v for k,v in validation_result.items()}, "selection_score": score}
        history.append(row)
        checkpoint = {
            "schemaVersion": "pfi.rsna-central-stenosis-checkpoint.v1",
            "epoch": number,
            "modelName": cfg.model_name,
            "modelStateDict": model.state_dict(),
            "classNames": CLASS_NAMES,
            "config": cfg.__dict__,
            "metrics": row,
            "manifestHashes": manifest_hashes,
            "governance": {"humanReviewRequired": True, "notClinicalDiagnosis": True, "officialTestAccessed": False, "internalTestAccessed": False},
        }
        torch.save(checkpoint, checkpoint_root / "last_checkpoint.pt")
        if score > best_score:
            best_score, best_epoch, stale = score, number, 0
            torch.save(checkpoint, checkpoint_root / "best_checkpoint.pt")
        else:
            stale += 1
        pd.DataFrame(history).to_csv(run_root / "training_history.csv", index=False)
        print(json.dumps(row, indent=2))
        if stale >= cfg.patience:
            break
    return {"bestEpoch": best_epoch, "bestSelectionScore": best_score, "epochsCompleted": len(history)}
