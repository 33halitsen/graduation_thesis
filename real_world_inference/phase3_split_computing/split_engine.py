import torch
import torch.nn as nn
import os
from collections import OrderedDict


# --- 1. HATA ÖNLEYİCİ KUKLA SINIFLAR ---
class SplitComputingV5Net(nn.Module):
    def __init__(self):
        super().__init__()


class SplitComputingV6Net(nn.Module):
    def __init__(self):
        super().__init__()


class ImageComplexityEstimator(nn.Module):
    def __init__(self):
        super().__init__()


import __main__

__main__.SplitComputingV5Net = SplitComputingV5Net
__main__.SplitComputingV6Net = SplitComputingV6Net
__main__.ImageComplexityEstimator = ImageComplexityEstimator

# --- 2. DOSYA YOLLARI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # split.py'nin bulunduğu klasör

# Ağaç yapına göre model yolları (faz3 içinden ana dizine çıkıyoruz)
V5_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../v5/results (1)/best_v5_quantized_zlib.pth")
)
V6_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../v6/results/best_v6_dynamic_zlib.pth")
)

OUTPUT_DIR = os.path.join(BASE_DIR, "parcalanmis_modeller")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 3. PARÇALAMA MOTORU ---
def split_model_weights(model_path, model_name):
    print(f"\n[{model_name}] Parçalama işlemi başlatılıyor...")

    if not os.path.exists(model_path):
        print(f"HATA: Dosya bulunamadı -> {model_path}")
        return

    try:
        loaded_data = torch.load(model_path, map_location="cpu")

        if isinstance(loaded_data, nn.Module):
            full_state_dict = loaded_data.state_dict()
        elif isinstance(loaded_data, dict):
            full_state_dict = loaded_data
        else:
            print("Bilinmeyen model formatı!")
            return

        encoder_state = OrderedDict()
        decoder_state = OrderedDict()

        # GERÇEK KATMAN İSİMLERİNE GÖRE AYIRMA MANTIĞI
        for key, value in full_state_dict.items():
            # 'valve', 'encoder' veya 'inspector' içerenler Sunucuya (Encoder)
            if (
                key.startswith("valve")
                or key.startswith("encoder")
                or key.startswith("inspector")
            ):
                encoder_state[key] = value
                print(f" -> Encoder'a eklendi: {key}")
            # 'decoder' içerenler İstemciye (Decoder)
            elif key.startswith("decoder"):
                decoder_state[key] = value
                print(f" -> Decoder'a eklendi: {key}")
            else:
                print(
                    f"Uyarı: '{key}' nereye ait olduğu anlaşılamadı, varsayılan olarak Encoder'a ekleniyor."
                )
                encoder_state[key] = value

        enc_path = os.path.join(OUTPUT_DIR, f"{model_name}_encoder.pth")
        dec_path = os.path.join(OUTPUT_DIR, f"{model_name}_decoder.pth")

        torch.save(encoder_state, enc_path)
        torch.save(decoder_state, dec_path)

        print(f"\nBAŞARILI! Model ikiye bölündü.")
        print(f" -> Encoder Parametre Sayısı: {len(encoder_state)}")
        print(f" -> Decoder Parametre Sayısı: {len(decoder_state)}")

    except Exception as e:
        print(f"Hata oluştu: {str(e)}")


# --- 4. ÇALIŞTIRMA ---
if __name__ == "__main__":
    split_model_weights(V5_PATH, "V5_Standart")
    split_model_weights(V6_PATH, "V6_Mufettisli")
