"""
Used for processing all pile data and sharding/converting it
"""
import os
import tarfile
from abc import ABC, abstractmethod
from glob import glob
import shutil
import random
import zstandard
from pathlib import Path
from the_pile.datasets import *

datasets = [
    # Academic
    (PubMedCentralDataset(), 2.  ),
    (ArXivDataset()        , 2.  ),
    (FreeLawDataset()      , 1.5 ),
    (USPTODataset()        , 2.  ),
    (PubMedDataset()       , 2.  ),
    (PhilPapersDataset()   , 2.  ),
    (ExPorterDataset()     , 2.  ),

    # General internet
    (OpenWebText2Dataset() , 2.  ),
    (StackExchangeDataset(), 2.  ),
    (WikipediaDataset()    , 3.  ),

    # Prose
    (BibliotikDataset()    , 1.5 ),
    (GutenbergDataset()    , 2.5 ),
    (BookCorpusDataset()   , 1.5 ),

    # Github
    (GithubDataset()       , 1.  ),

    # Dialogue
    (UbuntuIRCDataset()    , 2.  ),
    (HackerNewsDataset()   , 2.  ),
    (EuroParlDataset()     , 2.  ),
    (YTSubtitlesDataset()  , 2.  ),
    (OpensubtitlesDataset(), 1.5 ),

    # Misc
    (DMMathDataset()       , 2.  ),
    (EnronEmailsDataset()  , 2.  ),

]

class PileDownloader(ABC):

    def _extract_tar(self,path):
        path=str(path)
        output_path = path.replace(".tar.gz","")
        with tarfile.open(path, "r:gz") as dataset_tar:
            print(f'Extracting files from tar {path}...')
            dataset_tar.extractall(output_path)
        return output_path
     

    def _extract_zstd(self, path):
        path = str(path)
        output_path = path.replace(".gz","")
        os.system(f"zstd -d {path}")
        os.remove(path)
        return output_path
            

    def extract_all(self,folder=None):
        """extracts dataset and moves to the correct data dir if necessary"""
        if not folder:
            folder = "./components"
        all_compressed = get_compressed_files(folder)
        all_paths = []
        for f in all_compressed:
            extension = f.suffix
            new_path = ""
            if "zst" in extension:
                new_path = self._extract_zstd(f)
            elif "gz" in extension:
                new_path = self._extract_tar(f)
            else:
                print("Could not decompress file")
            all_paths.append(new_path)
        return all_paths
        

    def download_all(self):
        for dataset,_ in datasets:
            dataset._download()

def get_compressed_files(folder):
    p = Path(folder)
    print(p)
    zst_files = list(p.glob("*/*.zst"))
    tar_files = list(p.glob("*/*.gz"))
    files = zst_files+tar_files
    return files


def prepare_pile_data():
    d = PileDownloader()
    #d.download_all()
    all_paths = d.extract_all(folder="/ml_data/the-pile/components/")
    print(all_paths)
    return all_paths

if __name__ == "__main__":
    prepare_pile_data()