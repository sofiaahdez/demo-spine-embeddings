import kagglehub

# Download latest version
path = kagglehub.dataset_download("stefanoleone992/fifa-20-complete-player-dataset")

print("Path to dataset files:", path)