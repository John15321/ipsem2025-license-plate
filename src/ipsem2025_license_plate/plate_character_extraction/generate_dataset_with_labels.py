import cv2
import pytesseract
import os
import argparse
import shutil

def recognize_text(image_path):
    # Wczytaj obraz
    image = cv2.imread(image_path)

    # OCR - rozpoznanie tekstu
    custom_config = r'--oem 3 --psm 10'  # Tryb dla pojedynczego znaku
    text = pytesseract.image_to_string(image, config=custom_config)
    
    return text.strip()

def create_ocr_dataset(folder_path, output_folder):
    """
    Funkcja tworzy zbiór danych OCR, przenosząc obrazy do nowego folderu
    i zapisując rozpoznane litery w osobnych plikach .txt w folderze labels.
    """
    dataset = []
    
    # Tworzymy główny folder ocr-dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Tworzymy foldery images i labels w obrębie ocr-dataset
    images_folder = os.path.join(output_folder, "images")
    labels_folder = os.path.join(output_folder, "labels")
    
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    
    image_counter = 1  # Licznik dla nazw plików (np. 1.png, 2.png, ...)
    
    # Przechodzimy przez wszystkie foldery w głównym folderze
    for folder_name in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder_name)
        
        # Sprawdzamy, czy folder zawiera pliki .png
        if os.path.isdir(folder_path_full):
            for filename in os.listdir(folder_path_full):
                if filename.endswith(".png"):
                    image_path = os.path.join(folder_path_full, filename)
                    
                    # Rozpoznanie tekstu z obrazu
                    recognized_text = recognize_text(image_path)
                    
                    if recognized_text:
                        # Tworzenie unikatowej nazwy pliku
                        new_filename = f"{image_counter}.png"
                        new_image_path = os.path.join(images_folder, new_filename)
                        
                        # Przenosimy obraz do folderu images
                        shutil.copy(image_path, new_image_path)
                        
                        # Zapisujemy rozpoznany tekst w pliku .txt
                        label_filename = f"{image_counter}.txt"
                        label_path = os.path.join(labels_folder, label_filename)
                        
                        with open(label_path, "w") as label_file:
                            label_file.write(recognized_text)
                        
                        # Zwiększamy licznik dla kolejnego obrazu
                        image_counter += 1
    
    print(f"Zbiór uczący zapisano do {output_folder}")
    print(f"Obrazy zostały przeniesione do folderu: {images_folder}")
    print(f"Pliki tekstowe zostały zapisane w folderze: {labels_folder}")

def main():
    # Ustawienie domyślnych ścieżek
    default_input_path = "OUTPUT_SINGLE"  # Domyślny folder z obrazami
    default_output_path = "ocr-dataset"  # Domyślny folder wyjściowy
    
    # Konfiguracja argumentów wiersza poleceń
    parser = argparse.ArgumentParser(description="OCR Dataset Creator")
    parser.add_argument(
        "--input", 
        type=str, 
        default=default_input_path, 
        help="Ścieżka do folderu wejściowego z obrazami (domyślnie: 'OUTPUT_SINGLE')"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=default_output_path, 
        help="Ścieżka do folderu wyjściowego (domyślnie: 'ocr-dataset')"
    )
    
    # Parsowanie argumentów
    args = parser.parse_args()
    
    # Wywołanie funkcji do tworzenia zbioru OCR
    create_ocr_dataset(args.input, args.output)

if __name__ == "__main__":
    main()