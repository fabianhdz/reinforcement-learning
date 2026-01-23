import kagglehub

# Download latest version
path = kagglehub.dataset_download("lonnieqin/englishspanish-translation-dataset")

print("Path to dataset files:", path)