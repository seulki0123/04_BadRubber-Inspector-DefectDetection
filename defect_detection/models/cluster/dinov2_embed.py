import torch
import torchvision.transforms as T
from PIL import Image

# -----------------------------
# Model
# -----------------------------
_models = {}

def load_model(model_name="dinov2_vitb14", device=None):
    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cache_key = (model_name, str(target_device))
    if cache_key not in _models:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        _models[cache_key] = model.eval().to(target_device)
    return _models[cache_key]


# -----------------------------
# Transform
# -----------------------------
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    )
])


# -----------------------------
# Single image embedding
# -----------------------------
def get_embedding(image, device=None):

    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(device=target_device)

    img = Image.fromarray(image).convert("RGB")
    img = transform(img).unsqueeze(0).to(target_device)

    with torch.inference_mode():
        feat = model(img)

    feat = feat.squeeze()

    feat = torch.nn.functional.normalize(feat, dim=-1)

    return feat

# -----------------------------
# Batch embedding
# -----------------------------
def get_embeddings_batch(images, device=None):
    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(device=target_device)

    imgs = []
    for img_np in images:
        img = Image.fromarray(img_np).convert("RGB")
        img = transform(img)
        imgs.append(img)

    imgs = torch.stack(imgs).to(target_device, non_blocking=True)

    with torch.inference_mode():
        feats = model(imgs)

    feats = torch.nn.functional.normalize(feats, dim=1)

    return feats
