
Copy

@echo off
echo ============================================
echo    FusionFaceGAN - Yukleme ve Baslatma
echo ============================================
echo.

echo [1/2] Kutuphaneler yukleniyor...
pip install -r requirements.txt -q
echo.

echo [2/2] Arayuz baslatiliyor...
echo Tarayicinizda http://localhost:7860 adresini acin.
echo Kapatmak icin bu pencereyi kapatin.
echo.
start http://localhost:7860
python app.py

pause