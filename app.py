"""
🧬 FusionFaceGAN — İnsan Yüzü Üretim Arayüzü
================================================
Orijinal mimari ile eğitilmiş yüz üretim modeli.

Kullanım:
  1. best_generator.pth dosyasını bu klasöre kopyalayın
  2. pip install -r requirements.txt
  3. python app.py
  4. http://localhost:7860
"""

import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
import os
import time

from model import FusionGenerator

# ============================================================
# Konfigürasyon
# ============================================================

MODEL_PATH = "best_generator.pth"
LATENT_DIM = 128
G_BASE_CH = 64
NUM_CHANNELS = 3
IMAGE_SIZE = 64
UPSCALE_SIZE = 512  # 64→512 yüzler için daha büyük

# ============================================================
# Model Yükleme
# ============================================================

def load_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"🔧 Cihaz: {device_name}")

    model = FusionGenerator(
        latent_dim=LATENT_DIM,
        g_base_ch=G_BASE_CH,
        num_channels=NUM_CHANNELS
    ).to(device)

    fid, epoch = "N/A", "N/A"

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if "generator_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["generator_state_dict"])
            fid = checkpoint.get("fid", checkpoint.get("best_fid", "N/A"))
            epoch = checkpoint.get("epoch", "N/A")
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Model yüklendi: Epoch {epoch}, FID: {fid}")
    else:
        print(f"⚠️ Model bulunamadı: {MODEL_PATH}")

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    model_size = os.path.getsize(MODEL_PATH) / (1024*1024) if os.path.exists(MODEL_PATH) else 0

    info = {
        "device": device_name, "fid": fid, "epoch": epoch,
        "params": f"{total_params:,}", "size_mb": f"{model_size:.1f}",
    }
    return model, device, info


model, device, model_info = load_model()

# ============================================================
# Üretim Fonksiyonları
# ============================================================

def upscale(tensor_img, size=UPSCALE_SIZE):
    img = (tensor_img * 0.5 + 0.5).clamp(0, 1)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    img = F.interpolate(img, size=(size, size), mode="bicubic", align_corners=False)
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))


def generate_single(seed, upscale_on):
    t = time.time()
    if seed == -1:
        seed = np.random.randint(0, 999999)
    torch.manual_seed(seed)
    z = torch.randn(1, LATENT_DIM, 1, 1, device=device)
    with torch.no_grad():
        fake = model(z)
    size = UPSCALE_SIZE if upscale_on else IMAGE_SIZE
    img = upscale(fake[0], size)
    ms = (time.time() - t) * 1000
    return img, f"Seed: {seed} | {size}x{size} | {ms:.1f}ms | {model_info['device']}", seed


def generate_batch(num, seed, upscale_on):
    t = time.time()
    if seed == -1:
        seed = np.random.randint(0, 999999)
    torch.manual_seed(seed)
    z = torch.randn(int(num), LATENT_DIM, 1, 1, device=device)
    with torch.no_grad():
        fakes = model(z)
    size = UPSCALE_SIZE if upscale_on else IMAGE_SIZE
    imgs = [upscale(fakes[i], size) for i in range(fakes.size(0))]
    ms = (time.time() - t) * 1000
    return imgs, f"Seed: {seed} | {int(num)} yüz | {ms:.1f}ms"


def generate_face(seed, upscale_on):
    """Tek bir yüz üretir."""
    t = time.time()
    if seed == -1:
        seed = np.random.randint(0, 999999)
    torch.manual_seed(seed)
    z = torch.randn(1, LATENT_DIM, 1, 1, device=device)
    with torch.no_grad():
        fake = model(z)
    size = UPSCALE_SIZE if upscale_on else IMAGE_SIZE
    img = upscale(fake[0], size)
    ms = (time.time() - t) * 1000
    return img, f"Seed: {seed} | {size}x{size} | {ms:.1f}ms"


def get_model_info():
    return f"""
## 🧬 FusionFaceGAN — Model Bilgileri

| Özellik | Değer |
|---------|-------|
| **Mimari** | FusionFaceGAN (Orijinal) |
| **Dataset** | CelebA (50K yüz görüntüsü) |
| **Görüntü Boyutu** | 64×64 RGB |
| **Latent Boyut** | {LATENT_DIM} |
| **Parametreler** | {model_info['params']} |
| **Model Boyutu** | {model_info['size_mb']} MB |
| **En İyi FID** | {model_info['fid']} |
| **Eğitim Epoch** | {model_info['epoch']} |
| **Cihaz** | {model_info['device']} |

## 🏗️ Orijinal Mimari Bileşenler

### 1. Multi-Scale Fusion Generator
Her çözünürlük seviyesinden (8×8, 16×16, 32×32, 64×64) RGB çıktı üretilir
ve öğrenilebilir ağırlıklarla birleştirilir. Bu sayede her seviyedeki
detay korunur.

### 2. Adaptive Channel Attention (ACA)
Her kanal için önem skoru hesaplar. Gamma warm-up ile eğitimin başında
nötr, ilerledikçe etkili hale gelir.

### 3. Progressive Residual Upsample (PRU)
Bilinear upsample + çift konvolüsyon + skip connection + ACA.
ConvTranspose2d'nin checkerboard artefaktlarını önler.

### 4. Dual-Path Discriminator
Uzaysal yol (klasik konvolüsyon) + Frekans yolu (Haar Wavelet).
Sahte görüntülerdeki frekans anomalilerini yakalar.

### 5. Eğitim Teknikleri
- Hinge Loss + Feature Matching Loss
- DiffAugment (color + translation + cutout)
- EMA (Exponential Moving Average)
- TTUR (Two Time-Scale Update Rule)
- Mixed Precision (FP16) — T4 Tensor Core
- Cosine Annealing LR Scheduler

```
Generator Akış:
  z(128) → 4×4×512
         → PRU → 8×8×256  → toRGB₈
         → PRU → 16×16×128 → toRGB₁₆
         → PRU → 32×32×64  → toRGB₃₂
         → PRU → 64×64×64  → toRGB₆₄
  Final = Σ wᵢ · toRGBᵢ  (öğrenilebilir ağırlıklar)
```
"""


# ============================================================
# Gradio Arayüzü
# ============================================================

custom_css = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #e66465 0%, #9198e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em; font-weight: 800; margin-bottom: 0.2em;
}
.subtitle { text-align: center; color: #888; font-size: 1.1em; margin-bottom: 1.5em; }
footer {visibility: hidden}
"""

with gr.Blocks(
    css=custom_css,
    title="FusionFaceGAN — Yüz Üretici",
    theme=gr.themes.Soft(primary_hue="orange", secondary_hue="purple", neutral_hue="slate")
) as demo:

    gr.HTML("""
        <div class="main-title">🧬 FusionFaceGAN</div>
        <div class="subtitle">
            Orijinal Mimari ile İnsan Yüzü Üretimi &nbsp;|&nbsp;
            CelebA Dataset &nbsp;|&nbsp; T4 GPU Eğitimli
        </div>
    """)

    with gr.Tabs():

        # Tab 1: Yüz Üretme
        with gr.TabItem("🧑 Yüz Üretme"):
            gr.Markdown("Rastgele bir insan yüzü üretin. Aynı seed ile aynı yüzü tekrar elde edebilirsiniz.")
            with gr.Row():
                with gr.Column(scale=1):
                    seed_f = gr.Number(label="Seed (-1 = Rastgele)", value=-1, precision=0, info="Aynı seed = aynı yüz")
                    up_f = gr.Checkbox(label="Büyüt (64→512px)", value=True)
                    btn_f = gr.Button("🧑 Yüz Üret", variant="primary", size="lg")
                    btn_r = gr.Button("🔀 Rastgele", variant="secondary")
                    info_f = gr.Textbox(label="Bilgi", interactive=False)
                with gr.Column(scale=2):
                    out_f = gr.Image(label="Üretilen Yüz", type="pil", height=500)

            btn_f.click(generate_face, [seed_f, up_f], [out_f, info_f])
            btn_r.click(lambda u: generate_face(-1, u), [up_f], [out_f, info_f])

        # Tab 2: Model Bilgisi
        with gr.TabItem("📊 Model Bilgisi"):
            gr.Markdown(get_model_info())

    gr.HTML("""
        <div style="text-align:center; padding:20px; color:#666; font-size:0.85em;">
            FusionFaceGAN — Orijinal Mimari ile İnsan Yüzü Üretimi<br>
            CelebA Dataset | T4 GPU | Mixed Precision Eğitim | FID: """ + str(model_info['fid']) + """
        </div>
    """)


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧬 FusionFaceGAN Yüz Üretici")
    print("="*50)
    print(f"Model: {MODEL_PATH}")
    print(f"Cihaz: {model_info['device']}")
    print(f"FID: {model_info['fid']}")
    print("="*50)
    print("🌐 http://localhost:7860")
    print("="*50 + "\n")

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)