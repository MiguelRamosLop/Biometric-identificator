import psycopg2
import numpy as np

class DatabaseMigrator:
    def __init__(self, db_url):
        """
        Initialize the DatabaseMigrator with the database connection URL.

        Args:
            db_url (str): PostgreSQL database connection URL.
        """
        self.db_url = db_url

    def migrate_embeddings(self, embeddings_file, table_name="pictures"):
        """
        Migrate embeddings from a file to the database.

        Args:
            embeddings_file (str): Path to the .npy file containing embeddings.
            table_name (str): Name of the database table to insert records into.
        """
        # Load embeddings from the .npy file
        embeddings = np.load(embeddings_file, allow_pickle=True).item()

        # Connect to the database
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()

        try:
            # Insert each embedding into the database
            for file_name, embedding in embeddings.items():
                cur.execute(
                    f'INSERT INTO {table_name} (identificator, embedding) VALUES (%s, %s)',
                    (file_name, embedding.tolist())
                )
                print(f"Inserted: {file_name}")

            # Commit the transaction
            conn.commit()
            print("All embeddings migrated successfully.")
        except Exception as e:
            print(f"Error during migration: {e}")
            conn.rollback()
        finally:
            # Close the connection
            cur.close()
            conn.close()

# Example usage:
if __name__ == "__main__":
    db_url = 'postgresql://postgres:12345@localhost:5432/Faces'
    migrator = DatabaseMigrator(db_url)
    migrator.migrate_embeddings(embeddings_file="target_embeddings.npy", table_name="targets")