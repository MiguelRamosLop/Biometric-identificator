import cv2
import os

class FaceDetector:
    def __init__(self, detector_algorithm, output_prefix="Detected_face"):
        """
        Initialize the FaceDetector with a Haar Cascade file and output file prefix.
        """
        self.haar_cascade = cv2.CascadeClassifier(detector_algorithm)
        self.output_prefix = output_prefix
        self.face_id_counters = {"targets": 1, "prospects": 1}  # Separate counters for each folder_id

    def detect_faces_in_folder(self, folder_path, folder_id, output_dir="."):
        """
        Detect faces in all images within the given folder and save cropped face images to the output directory.

        Args:
            folder_path (str): Path to the folder containing images.
            folder_id (str): Identifier for the folder (e.g., "targets" or "prospects").
            output_dir (str): Directory to save the cropped face images.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Set the output prefix based on the folder_id
        if folder_id == "targets":
            self.output_prefix = "Targeted_face"
        elif folder_id == "prospects":
            self.output_prefix = "Prospected_face"
        else:
            self.output_prefix = "Detected_face"

        # Ensure the folder_id has a counter initialized
        if folder_id not in self.face_id_counters:
            self.face_id_counters[folder_id] = 1

        # Iterate through all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self._detect_faces_in_image(file_path, folder_id, output_dir)

    def _detect_faces_in_image(self, image_path, folder_id, output_dir):
        """
        Detect faces in a single image and save cropped face images to the output directory.

        Args:
            image_path (str): Path to the input image.
            folder_id (str): Identifier for the folder (e.g., "targets" or "prospects").
            output_dir (str): Directory to save the cropped face images.
        """
        # Read the image as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to read image file '{image_path}'. Skipping.")
            return

        # Configure detection parameters based on folder_id
        if folder_id == "targets":
            scaleFactor = 1.1
            minNeighbors = 1
            minSize = (40, 40)
        elif folder_id == "prospects":
            scaleFactor = 1.05
            minNeighbors = 1
            minSize = (60, 60)

        # Detect faces in the image
        faces = self.haar_cascade.detectMultiScale(
            img,
            scaleFactor=scaleFactor, # Parameter specifying how much the image size is reduced at each image scale.
            minNeighbors=minNeighbors, # Parameter specifying how many neighbors each candidate rectangle should have to retain it.
            minSize=minSize # Minimum possible object size. Objects smaller than this are ignored.
        )

        # Process each detected face
        for x, y, w, h in faces:
            cropped_image = img[y:y + h, x:x + w]
            target_file_name = f"{output_dir}/{self.output_prefix}_{self.face_id_counters[folder_id]}.png"
            cv2.imwrite(target_file_name, cropped_image)
            self.face_id_counters[folder_id] += 1  # Increment the counter for the specific folder_id

        print(f"Processed '{image_path}': Detected and saved {len(faces)} face(s).")