from nltk.tokenize import word_tokenize
import numpy as np
from .vectorizer.tfidf_vectorizer import TfidfVectorizer


class TfidfSearchEngine:
    """Represents a class that uses TF-IDF scores to return the most relevant documents for particular terms"""

    def __init__(self) -> None:
        """Initializes the class variables"""

        print(f'{'-'*5} Initializing the search engine {'-'*5}')
        # Initialize a TF-IDF vactorizer
        self.tfidf_vectorizer = TfidfVectorizer()

    def check_if_document_is_indexed(self, document_name: str) -> bool:
        """Checks whether the document with the provided name is already indexed in the search engine

        Args:
            document_name (str): The name of the document

        Returns:
            bool: Whether the document is indexed
        """
        # Check whether the document with the provided name exists in the document-row mapping doctionary
        if document_name in self.tfidf_vectorizer.get_document_row_mapping():
            return True
        return False


    def index(self, data: list[tuple[str, str]]) -> None:
        """Indexes the provided data into underyling data model

        Args:
            data (list[tuple[str, str]]): The data that will be indexed. Assumption is that the data is a list of tuples that have the follwoing structure: ('document_data', 'document_name')
        """
        print(f'- Indexing the document(s) into the search engine ...')
        # Fit the TF-IDF vectorizer with the provided data
        self.tfidf_vectorizer.fit(data=data)
        print(f'- The document(s) are indexed. Searching can be performed now.')
        print('-'*41)

    
    def search(self, search_query: str, num_of_results = 10, num_of_results_to_skip = 0) -> list[str]:
        """Performs a search over indexed documents

        Args:
            search_query (str): The search query
            num_of_results (int, optional): A number of the most relevant documents that need to be returned. Defaults to 10.
            num_of_results_to_skip (int, optional): A number of the most relevant documents to skip. Defaults to 0.

        Returns:
            list[str]: The 'num_of_results' most relevant documents
        """
        # Split the search query into words
        search_query_words = word_tokenize(search_query.lower())

        # Get the most relevant documents for the specified search words
        return self.__get_most_relevant_documents_for_terms(search_query_words, num_of_documents=num_of_results, num_of_documents_to_skip=num_of_results_to_skip)

    
    def __get_most_relevant_documents_for_terms(self, terms: list[str], num_of_documents, num_of_documents_to_skip) -> list[str]:
        """Computes a list of the most relevant documents for the given terms

        Args:
            terms (list[str]): Terms for which to find most revevant documents
            num_of_documents (_type_): Number of the most relevant documents that need to be returned
            num_of_documents_to_skip (_type_): Number of the most relevant documents that need to be skipped

        Returns:
            list[str]: List of the most relevant documents
        """
        # For each term, extract the column index in the TF-IDF matrix of the TF-IDF vectorizer
        term_column_mapping = self.tfidf_vectorizer.get_term_column_mapping()
        column_indices = [term_column_mapping.get(term) for term in terms if term in term_column_mapping]

        # Extract only the specified columns from the TF-IDF sparse matrix and sum the rows
        row_sums = np.array(self.tfidf_vectorizer.tf_idf_matrix[:, column_indices].sum(axis=1)).flatten()

        # Find the indices of rows with the highest sums - those are the indices of the most relevant documents
        starting_index = - (num_of_documents + num_of_documents_to_skip)
        ending_index = len(row_sums) if num_of_documents_to_skip ==0 else -num_of_documents_to_skip
        most_relevant_document_indices = np.argsort(row_sums)[starting_index:ending_index][::-1] # Slice the result depending on the provided arguments

        # Extract the names of the most relevant documents
        document_row_mapping = list(self.tfidf_vectorizer.get_document_row_mapping().keys())
        document_names = [document_row_mapping[doc_index] for doc_index in most_relevant_document_indices]

        return document_names


