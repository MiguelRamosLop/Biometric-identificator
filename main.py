from detector import FaceDetector
from embedder import ImageEmbedder
from migrator import DatabaseMigrator
from comparator import Comparator

def main():
    db_url = 'postgresql://postgres:12345@localhost:5432/Faces'
    
    # Step 1: Detect faces in input images
    detector = FaceDetector("haarcascade_frontalface_default.xml")
    input_folder = "input_images"
    detected_faces_folder = "prospects"  # Folder to save cropped faces
    print("Detecting prospects...")
    detector.detect_faces_in_folder(input_folder, folder_id="prospects", output_dir=detected_faces_folder)

    # Step 2: Detect faces in target images (apply the same algorithm)
    input_target_folder = "raw_targets"
    target_faces_folder = "targets"  # Folder to save cropped faces
    print("Manipulating targets...")
    detector.detect_faces_in_folder(input_target_folder, folder_id="targets", output_dir=target_faces_folder)

    # Step 2: Calculate embeddings for target faces
    embedder = ImageEmbedder(input_folder="targets", output_file="targets_embeddings.npy")
    print("Calculating embeddings for targets...")
    embedder.calculate_embeddings()

    # Step 3: Calculate embeddings for detected faces
    embedder = ImageEmbedder(input_folder=detected_faces_folder, output_file="prospects_embeddings.npy")
    print("Calculating embeddings for prospects...")
    embedder.calculate_embeddings()

    migrator = DatabaseMigrator(db_url)
    # Step 4: Migrate targets embeddings to the database
    print("Migrating targets embeddings to the database...")
    migrator.migrate_embeddings(embeddings_file="targets_embeddings.npy", table_name="targets")
    # Step 5: Migrate prospects embeddings to the database
    print("Migrating prospects embeddings to the database...")
    migrator.migrate_embeddings(embeddings_file="prospects_embeddings.npy", table_name="prospects")

    print("Pipeline completed successfully.")
    
    # Step 6: Find matches between targets and prospects using database tables
    comparator = Comparator(db_url=db_url, similarity_threshold=0.86)
    print("Finding matches between targets and prospects...")
    matches = comparator.find_matches(
        targets_table="targets",
        prospects_table="prospects"
    )

    # Print matches
    if matches:
        print("Matches found:")
        for target_file, prospect_file, similarity in matches:
            print(f"Target: {target_file}, Prospect: {prospect_file}, Similarity: {similarity:.2f}")
    else:
        print("No matches found.")
    
if __name__ == "__main__":
    main()