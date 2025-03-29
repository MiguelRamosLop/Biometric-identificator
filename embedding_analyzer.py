import numpy as np

# Load the .npy file
file_path = "embeddings.npy"  # Replace with your file's path if different
embeddings = np.load(file_path, allow_pickle=True).item()

# Print the entire dictionary (might be long if you have many images)
print("Full embeddings dictionary:")
print(embeddings)

# Print just the keys (filenames)
print("\nList of image filenames:")
print(list(embeddings.keys()))

# Print a specific embedding (e.g., for the first image)
first_image = list(embeddings.keys())[0]

# Check the shape and type of an embedding
print(f"Shape of {first_image}'s embedding: {embeddings[first_image].shape}")
print(f"Type of {first_image}'s embedding: {type(embeddings[first_image])}")