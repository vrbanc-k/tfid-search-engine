from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, vstack
from .abstract_vectorizer import AbstractVectorizer


class CountVectorizer(AbstractVectorizer):
    """Represents a class that builds document-term matrix bases on the provided data"""

    def __init__(self) -> None:
        """Initializes the class variables"""

        # Initialize a dictionary that will map terms to column indexes in the sparse matrix
        self.term_column_mapping : dict[str, int] = {}
        # Initialize a dictionary that will map documents to rows in the sparse matrix
        self.document_row_mapping : dict[str, int] = {}
        # Initialize a document term matrix as None
        self.document_term_matrix : csr_matrix = None


    def fit(self, data: list[tuple[str, str]]) -> None:
        """Generates (or updates) the document-term matrix based on the provided data

        Args:
            data (list[tuple[str, str]]): Data on which to build the document-term matrix. The assumed format of the data is: ('document_content', 'document_name')
        """
        word_column_indices = []
        sparse_matrix_data = []
        row_index_pointers = [0]
        for data_record in data:
            # Extract the document name
            document_name = data_record[1]
            # If the document with the same name is already indexed, skip it
            if document_name in self.document_row_mapping:
                continue
    
            # Set the row index of the data record
            self.document_row_mapping.setdefault(document_name, len(self.document_row_mapping))

            # Tokenize the document data into lowercased words
            tokenized_data = word_tokenize(data_record[0].lower())

            for word in tokenized_data:
                # Determine the column index of the word
                word_column_index = self.term_column_mapping.setdefault(word, len(self.term_column_mapping))
                word_column_indices.append(word_column_index)
                # Note that the word has appeared in the document
                sparse_matrix_data.append(1)
            # Note at which index does the matrix row end
            row_index_pointers.append(len(word_column_indices))

        # If there is no data that needs to be saved, return from the function 
        # (this is the case when all the documents that are sent to the function are already indexed)
        if len(sparse_matrix_data) == 0:
            return
        
        if self.document_term_matrix is None:
            # Create the sparse matrix that contains the number of occurances of particular words in documents
            self.document_term_matrix = csr_matrix((sparse_matrix_data, word_column_indices, row_index_pointers), dtype=int)
        else:
            # Create a sparse matrix out of the newly added rows
            additional_rows = csr_matrix((sparse_matrix_data, word_column_indices, row_index_pointers), dtype=int)
            # Handle the mismatching matrix sizes (only the number of columns matters)
            if additional_rows.shape[1] > self.document_term_matrix.shape[1]:
                self.document_term_matrix.resize((self.document_term_matrix.shape[0], additional_rows.shape[1]))
            elif additional_rows.shape[1] < self.document_term_matrix.shape[1]:
                additional_rows.resize((additional_rows.shape[0], self.document_term_matrix.shape[1]))
            # Vertically stack the newly added rows onto existing document term matrix
            self.document_term_matrix = vstack([self.document_term_matrix, additional_rows])