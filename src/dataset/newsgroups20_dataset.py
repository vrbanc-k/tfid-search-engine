import os
from sklearn.datasets import fetch_20newsgroups

class NewsGroups20Dataset:
    """Represents a 20 Newsgroups dataset wrapper"""


    def __init__(self) -> None:
        """Loads the 20 Newsgroups dataset"""
        
        print(f'{'-'*5} Dataset loading {'-'*5}')
        print(f'- Started loading the 20 Newsgroup dataset ...')
        print(f'- (Note: It might take some time if it is loaded for the first time because the dataset will be downloaded from the internet)')
        try:
            # Load the dataset
            newsgroup_dataset = fetch_20newsgroups(data_home='../temp', subset='all', shuffle=False, remove=(), download_if_missing=True)

            # Save the data and filenames
            self._data : list[str] = newsgroup_dataset.data
            self._filenames : list[str] = [self.construct_record_name(filename) for filename in newsgroup_dataset.filenames]
            print('- Dataset loading succeeded!')
        except:
            print(f'- Error fetching the 20 Newsgroup dataset from the internet! The dataset will be initialized as empty.')
            self._data = []
            self._filenames = []
        print('-'*27)


    def construct_record_name(self, file_path: str) -> str:
        """Constructs dataset record name based on the file path.The dataset record name is constructed by concatenating filename and parent directory.

        Args:
            file_path (str): The file path from which to counstruct the dataset record name

        Returns:
            str: The dataset record name
        """
        # Get the base name (file name) from the path
        filename = os.path.basename(file_path)
        
        # Get the parent directory from the path
        parent_folder = os.path.basename(os.path.dirname(file_path))
        
        return f'{parent_folder}/{filename}'
    
    @property
    def dataset(self) -> list[tuple[str, str]]:
        """Returns dataset records

        Returns:
            list[tuple[str, str]]: List of database records whose structure is as follows: ('data', 'name')
        """
        return zip(self._data, self._filenames)
        


    
