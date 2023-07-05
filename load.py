import argparse
import os
import shutil

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory")
args = parser.parse_args()

data_dir = args.directory

vector_db_dir = "/dbfs/taylors_cool_app/numpy_chroma_index"

loader = PyPDFDirectoryLoader(data_dir)
pages = loader.load_and_split()

os.environ[
    "OPENAI_API_KEY"
] = "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"
embeddings = OpenAIEmbeddings()  # noqa
db = Chroma.from_documents(pages, embeddings, persist_directory=vector_db_dir)


if os.path.exists(vector_db_dir):
    shutil.rmtree(vector_db_dir)

db.persist()
