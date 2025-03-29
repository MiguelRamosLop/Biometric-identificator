import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import mahalanobis

class Comparator:
    def __init__(self, db_url, similarity_threshold):
        """
        Initialize the Comparator with a database connection URL and similarity threshold.

        Args:
            db_url (str): PostgreSQL database connection URL.
            similarity_threshold (float): Threshold for cosine similarity to consider a match.
        """
        self.db_url = db_url
        self.similarity_threshold = similarity_threshold
    
    def euclidean_distance(vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    def manhattan_distance(vec1, vec2):
        return np.sum(np.abs(vec1 - vec2))

    def dot_product(vec1, vec2):
        return np.dot(vec1, vec2)

    def jaccard_similarity(vec1, vec2):
        intersection = np.sum(np.minimum(vec1, vec2))
        union = np.sum(np.maximum(vec1, vec2))
        return intersection / union

    def mahalanobis_distance(vec1, vec2, cov_matrix):
        return mahalanobis(vec1, vec2, np.linalg.inv(cov_matrix))

    def fetch_embeddings(self, table_name):
        """
        Fetch embeddings from the specified database table.

        Args:
            table_name (str): Name of the database table to fetch embeddings from.

        Returns:
            dict: A dictionary with file names as keys and embeddings as values.
        """
        embeddings = {}
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()

        try:
            cur.execute(f"SELECT identificator, embedding FROM {table_name}")
            rows = cur.fetchall()
            for file_name, embedding in rows:
                embeddings[file_name] = np.array(eval(embedding))
        except Exception as e:
            print(f"Error fetching embeddings from table '{table_name}': {e}")
        finally:
            cur.close()
            conn.close()

        return embeddings

    def find_matches(self, targets_table, prospects_table):
        """
        Find matches between targets and prospects based on cosine similarity.

        Args:
            targets_table (str): Name of the database table containing target embeddings.
            prospects_table (str): Name of the database table containing prospect embeddings.

        Returns:
            list: A list of matches in the format [(target_file, prospect_file, similarity_score), ...].
        """
        # Fetch embeddings from the database
        targets_embeddings = self.fetch_embeddings(targets_table)
        prospects_embeddings = self.fetch_embeddings(prospects_table)

        matches = []
        for target_file, target_embedding in targets_embeddings.items():
            for prospect_file, prospect_embedding in prospects_embeddings.items():
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    [target_embedding], [prospect_embedding]
                )[0][0]
                if similarity >= self.similarity_threshold:
                    matches.append((target_file, prospect_file, similarity))

        return matches