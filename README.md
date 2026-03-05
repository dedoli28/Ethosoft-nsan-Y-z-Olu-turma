# 🧬 FusionFaceGAN — İnsan Yüzü Üretim Arayüzü

Orijinal mimari ile eğitilmiş insan yüzü üretim modeli.

## 📁 Dosya Yapısı

```
fusionface_app/
├── app.py              # Gradio arayüzü
├── model.py            # FusionGenerator mimari tanımı
├── best_generator.pth  # Eğitilmiş model (Colab'dan indirin)
├── requirements.txt    # Bağımlılıklar
└── README.md
```

## 🚀 Kurulum & Çalıştırma

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:7860
```

## 🏗️ Orijinal Mimari: FusionFaceGAN

| Bileşen | Açıklama |
|---------|----------|
| **Multi-Scale Fusion Generator** | Her seviyeden RGB çıktı + öğrenilebilir fusion |
| **Adaptive Channel Attention (ACA)** | Gamma warm-up ile kanal dikkati |
| **Progressive Residual Upsample (PRU)** | Checkerboard-free upsampling |
| **Dual-Path Discriminator** | Uzaysal + Haar Wavelet frekans analizi |

## 📊 Eğitim

- **Dataset:** CelebA (50K yüz, 64×64)
- **GPU:** NVIDIA Tesla T4 (Mixed Precision FP16)
- **FID:** 36.8
- **Epoch:** 80
- **Teknikler:** Hinge Loss, DiffAugment, EMA, TTUR, Cosine Annealing
