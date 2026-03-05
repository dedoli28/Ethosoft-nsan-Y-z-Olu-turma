<div align="center">

# 🧬 FusionFaceGAN

### Orijinal Mimari ile İnsan Yüzü Üretimi

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20Tesla%20T4-76B900.svg)](https://www.nvidia.com)
[![FID](https://img.shields.io/badge/FID-36.8-purple.svg)]()

<br>

**FusionFaceGAN**, tamamen orijinal bir GAN mimarisi kullanarak gerçekçi insan yüzleri üreten bir derin öğrenme projesidir. Model, NVIDIA Tesla T4 GPU üzerinde CelebA veri seti ile eğitilmiştir.

<br>

</div>

---

## 📊 Eğitim Sonuçları

### NVIDIA Tesla T4 GPU Kanıtı

Model eğitimi, Google Colab üzerinde **NVIDIA Tesla T4 GPU** kullanılarak gerçekleştirilmiştir. Aşağıda `nvidia-smi` çıktısı ile kanıtlanmaktadır:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.05              Driver Version: 560.35.05      CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off | 00000000:00:04.0   Off |                    0 |
| N/A   50C    P0              27W /  70W |   14253MiB /  15360MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

### Eğitim Metrikleri

| Metrik | Değer |
|--------|-------|
| **En İyi FID Skoru** | **36.8** |
| **Dataset** | CelebA (50.000 yüz görüntüsü) |
| **Çözünürlük** | 64×64 RGB |
| **Toplam Epoch** | 80 |
| **Eğitim Süresi** | ~30 dakika |
| **GPU** | NVIDIA Tesla T4 (16GB VRAM) |
| **Precision** | Mixed Precision (FP16) |

---

## 🏗️ Orijinal Mimari: FusionFaceGAN

Bu projede mevcut GAN mimarilerinden (DCGAN, StyleGAN, SAGAN) ilham alınarak tamamen **orijinal** bir hibrit mimari tasarlanmıştır. Hiçbir mevcut paper'ın birebir kopyası değildir.

### Mimari Bileşenler

#### 1. Multi-Scale Fusion Generator
Her çözünürlük seviyesinden (8×8, 16×16, 32×32, 64×64) ayrı RGB çıktı üretilir. Bu çıktılar, **öğrenilebilir ağırlıklarla** (softmax ile normalize) toplanarak son görüntü oluşturulur. Bu sayede her seviyedeki detay korunur.

```
Generator Akışı:
  z(128) → ConvTranspose → 4×4×512
         → PRU → 8×8×256   → toRGB₈
         → PRU → 16×16×128  → toRGB₁₆
         → PRU → 32×32×64   → toRGB₃₂
         → PRU → 64×64×64   → toRGB₆₄

  Çıktı = w₁·toRGB₈ + w₂·toRGB₁₆ + w₃·toRGB₃₂ + w₄·toRGB₆₄
  (w₁, w₂, w₃, w₄ öğrenilebilir fusion ağırlıkları)
```

#### 2. Adaptive Channel Attention (ACA)
SE-Net'ten ilham alan ancak GAN eğitimine özel tasarlanmış kanal dikkat mekanizması. **Gamma warm-up** parametresi sayesinde eğitimin başında nötr kalır, ilerledikçe kademeli olarak devreye girer.

#### 3. Progressive Residual Upsample (PRU)
ConvTranspose2d yerine kullanılan orijinal upsampling bloğu. Bilinear upsample + çift konvolüsyon + residual skip + ACA. ConvTranspose2d'nin bilinen **checkerboard artefakt** sorununu ortadan kaldırır.

#### 4. Dual-Path Discriminator
İki paralel analiz yolu içeren orijinal ayırt edici:

- **Uzaysal Yol:** Spectral Norm konvolüsyon zinciri ile görüntünün uzaysal yapısını analiz eder
- **Frekans Yolu:** Haar Wavelet dönüşümü ile görüntüyü frekans bileşenlerine (LL, LH, HL, HH) ayırır

İki yolun skorları **öğrenilebilir ağırlıkla** birleştirilerek son karar verilir.

#### 5. Kayıp Fonksiyonu: Hinge Loss + Feature Matching
- **Hinge Loss:** BCE'den daha stabil GAN eğitimi
- **Feature Matching Loss:** Mode collapse'ı önler

---

## ⚡ Eğitim Optimizasyonları

| Teknik | Açıklama |
|--------|----------|
| **Mixed Precision (FP16)** | T4 Tensor Core kullanımı → ~2× hızlanma |
| **DiffAugment** | color + translation + cutout augmentation |
| **EMA (β=0.999)** | Generator ağırlıklarının hareketli ortalaması |
| **TTUR** | D için 4× daha yüksek öğrenme oranı |
| **Cosine Annealing LR** | Kademeli LR düşüşü |
| **Spectral Normalization** | Discriminator'da Lipschitz kısıtı |
| **cuDNN Benchmark** | Otomatik en hızlı kernel seçimi |

---

## 📁 Proje Yapısı

```
FusionFaceGAN/
│
├── 📓 FusionFaceGAN_Fast.ipynb       ← Model eğitim notebook'u (Google Colab'da açın)
│
├── 📂 fusionface_app/                 ← Arayüz uygulaması (lokalde çalıştırın)
│   ├── 🟢 calistir.bat                 ← ÇİFT TIKLA → Uygulama başlar!
│   ├── app.py                         ← Gradio arayüz kodu
│   ├── model.py                       ← Generator mimari tanımı
│   ├── best_generator.pth             ← Eğitilmiş model ağırlıkları
│   └── requirements.txt               ← Python bağımlılıkları
│
├── 📂 outputs_fusion/                 ← Eğitim çıktıları
│   ├── 📂 models/
│   │   ├── best_generator.pth         ← En iyi FID modeli
│   │   ├── generator_final.pth        ← Son epoch modeli
│   │   └── generator_traced.pt        ← TorchScript export
│   ├── 📂 samples/                    ← Epoch bazlı üretilen yüzler
│   ├── 📂 logs/
│   │   └── log.json                   ← Eğitim logları (FID, süre, config)
│   ├── report.png                     ← Eğitim grafikleri (loss, FID, fusion weights)
│   └── comparison.png                 ← Gerçek vs üretilen karşılaştırma
│
└── 📄 README.md                       ← Bu dosya
```

### Hangi Dosya Ne İşe Yarar?

| Dosya | Açıklama | Nasıl Kullanılır |
|-------|----------|-----------------|
| `calistir.bat` | **Tek tıkla çalıştırma** | Çift tıkla → kütüphaneler kurulur → arayüz açılır |
| `app.py` | Gradio web arayüzü | `calistir.bat` otomatik çalıştırır |
| `model.py` | Generator sınıf tanımı | `app.py` tarafından otomatik yüklenir |
| `best_generator.pth` | Eğitilmiş model | `app.py` ile aynı klasörde olmalı |
| `FusionFaceGAN_Fast.ipynb` | Eğitim kodu | Colab'da aç → GPU (T4) seç → çalıştır |
| `report.png` | Eğitim grafikleri | Loss, FID, fusion weights grafikleri |
| `comparison.png` | Gerçek vs üretilen | Yan yana karşılaştırma |
| `log.json` | Eğitim detayları | FID, epoch, süre, GPU bilgisi |

---

## 🚀 Çalıştırma

### Windows — Tek Tıkla

> **`calistir.bat`** dosyasına çift tıklayın. Gerekli kütüphaneler otomatik kurulur ve tarayıcınızda arayüz açılır.

### Manuel Kurulum

```bash
# 1. Repoyu klonla
git clone https://github.com/KULLANICI/FusionFaceGAN.git
cd FusionFaceGAN/fusionface_app

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. Çalıştır
python app.py

# 4. Tarayıcıda aç
#    → http://localhost:7860
```

### Gereksinimler

- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU (opsiyonel — CPU'da da çalışır)

---

## 🖥️ Arayüz

Gradio tabanlı web arayüzü iki sekmeden oluşur:

### 🧑 Yüz Üretme
- Rastgele veya seed bazlı yüz üretimi
- 64×64 → 512×512 büyütme seçeneği
- Üretim süresi (ms) bilgisi
- Aynı seed = aynı yüz (tekrarlanabilirlik)

### 📊 Model Bilgisi
- Orijinal mimari bileşenlerin açıklamaları
- Eğitim teknikleri ve hiperparametreler
- FID skoru ve model istatistikleri

---

## 📈 Eğitim İlerlemesi

| Epoch | FID Skoru |
|-------|-----------|
| 20 | ~120 |
| 40 | ~65 |
| 60 | ~45 |
| 80 | **36.8** |

---

## 🛠️ Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| Latent Boyut | 128 |
| G/D Base Channels | 64 |
| Batch Size | 256 |
| LR (Generator) | 0.0002 |
| LR (Discriminator) | 0.0004 |
| Adam β₁, β₂ | 0.0, 0.9 |
| EMA Decay | 0.999 |
| DiffAugment | color, translation, cutout |
| Feature Matching Weight | 0.5 |

---

<div align="center">

**FusionFaceGAN** — Orijinal Mimari ile İnsan Yüzü Üretimi

NVIDIA Tesla T4 | CelebA 50K | FID: 36.8 | Mixed Precision FP16

</div>
