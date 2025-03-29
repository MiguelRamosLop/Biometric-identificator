import os
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import warnings

# Disable warnings
warnings.filterwarnings("ignore")

class ImageEmbedder:
    def __init__(self, input_folder, output_file="embeddings.npy"):
        """
        Initialize the ImageEmbedder with the input folder and output file.

        Args:
            input_folder (str): Path to the folder containing images.
            output_file (str): Path to save the calculated embeddings.
        """
        self.input_folder = input_folder
        self.output_file = output_file
        self.imgbeddings = imgbeddings()

    def calculate_embeddings(self):
        """
        Calculate embeddings for all images in the input folder and save them to the output file.
        """
        embeddings = {}
        for file_name in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = Image.open(file_path)
                    embedding = self.imgbeddings.to_embeddings(img)[0]
                    embeddings[file_name] = embedding
                    print(f"Processed: {file_name}")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

        # Save embeddings to a file
        np.save(self.output_file, embeddings)
        print(f"Embeddings saved to {self.output_file}")