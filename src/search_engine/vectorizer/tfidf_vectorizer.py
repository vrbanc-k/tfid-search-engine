from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, lil_array, vstack
from .abstract_vectorizer import AbstractVectorizer
from .count_vectorizer import CountVectorizer

class TfidfVectorizer(AbstractVectorizer):
    """Represents a class responisble for calculating TF-IDF scores based on the document-term matrix"""
    

    def __init__(self) -> None:
        """Initializes the class variables"""

        # Initialize an instance of CountVectorizer
        self.count_vectorizer = CountVectorizer()
        # Initialize the TF-IDF matrix to None
        self.tf_idf_matrix : csr_matrix = None

    
    def __calculate_tfidf_scores(self) -> None:
        """Calculates the TF-IDF scores based on the document-term matrix"""

        # Calculate the number of terms per documents (sum of the rows in the document-term matrix)
        num_of_terms_per_documents = self.count_vectorizer.document_term_matrix.sum(axis=1, dtype=int)

        # Calculate the number of occurances for each term (sum of columns in binary representation of document-term matrix)
        num_of_term_occurances = (self.count_vectorizer.document_term_matrix > 0).astype(int).sum(axis=0, dtype=int)
        inverse_document_frequencies = lil_array(1 / num_of_term_occurances)

        # Calculate the TF-IDF matrix
        self.tf_idf_matrix = csr_matrix(self.count_vectorizer.document_term_matrix / num_of_terms_per_documents).multiply(inverse_document_frequencies)

        
    def fit(self, data: list[tuple[str, str]]) -> None:
        """Generates (or updates) TF-IDF matrix based on the provided data

        Args:
            data (list[tuple[str, str]]): The data on which to update the TF-IDF matrix. 
        """

        # Generate (or update) the document-term matrix based on the provided data
        self.count_vectorizer.fit(data=data)

        # Calculate the TF-IDF scores
        self.__calculate_tfidf_scores()
    

    def get_term_column_mapping(self) -> dict[str, int]:
        """Retrieve term to matrix column mapping 

        Returns:
            dict[str, int]: Dictionary mapping terms to matrix column indices
        """
        return self.count_vectorizer.term_column_mapping if self.count_vectorizer is not None else {} 
    

    def get_document_row_mapping(self) -> dict[str, int]:
        """Retrieve document to matrix row mapping 

        Returns:
            dict[str, int]: Dictionary mapping documents to matrix row indices
        """
        return self.count_vectorizer.document_row_mapping if self.count_vectorizer is not None else {}