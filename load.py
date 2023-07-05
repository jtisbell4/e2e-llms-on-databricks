import argparse
import os
import shutil

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

VECTOR_DB_DIR = "/dbfs/taylors_cool_app/numpy_chroma_index"


def load_data(data_dir: str):
    """Splits, loads, and embeds a directory of PDF files into a Chroma vector
    database

    Args:
        data_dir (str): Location of PDF files
    """

    loader = PyPDFDirectoryLoader(data_dir)
    pages = loader.load_and_split()

    os.environ[
        "OPENAI_API_KEY"
    ] = "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"
    embeddings = OpenAIEmbeddings()  # noqa
    db = Chroma.from_documents(
        pages, embeddings, persist_directory=VECTOR_DB_DIR
    )

    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR)

    db.persist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory")
    args = parser.parse_args()

    data_dir = args.directory

    load_data(data_dir=data_dir)
