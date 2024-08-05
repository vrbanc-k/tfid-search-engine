from abc import ABC, abstractmethod

class AbstractVectorizer(ABC):
    """Represents a base abstract class for vecorizer classes"""

    @abstractmethod
    def fit(self, data: list[tuple[str, str]]) -> None:
        """Populates underlying data model based on the provided data

        Args:
            data (list[tuple[str, str]]): Data on which the data model will be fitted. The assumed format of the data is: ('document_content', 'document_name')
        """
        pass