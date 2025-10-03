import kagglehub

# Download latest version
path = kagglehub.dataset_download("pinxau1000/radioml2018")

print("Path to dataset files:", path)