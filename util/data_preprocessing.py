
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import  NLTKTextSplitter
from langchain_core.documents import Document
from datetime import datetime

import uuid
import re
import nltk

def preprocess_text(text):
    """ Remove newlines and normalize spaces
        :param text [str]: text data
        :return [dictionary]: metadata extracted from raw text
    """

    text = text.replace("\n", " ")
    text = " ".join(text.split())  # Normalize spaces
    return text


def extract_metadata(pdf_text, pdf_file_name):
    """ Extract metadata from pdf raw text
        :param pdf_text [str]: pdf raw text
        :param pdf_file_name [str]: name of pdf file
        :return [str]: pre processed text
    """
    metadata = {
        'title': pdf_file_name,
        'date': '',  
        'court': '',  
        'petitioner': 'Info unavailable', # Default
        'respondent': 'Info unavailable', # Default
        'bench': 'Info unavailable',
        'citation': '',
    }

    # Regex patterns 
    date_pattern = r"DATE OF JUDGMENT(\d{2}/\d{2}/\d{4})"
    court_pattern = r"http://JUDIS.NIC.IN ([\w\s-]+) Page"
    petitioner_pattern = r"PETITIONER:\s*(.*?)\s*Vs\."
    respondent_pattern = r"RESPONDENT:\s*(.*?)\s*DATE OF JUDGMENT"
    benches_pattern = r"(BENCH:\s*(.*?))\s*CITATION:"
    citation_pattern = r"CITATION:\s*(.*?)\s*ACT:"


    # Extracting date
    date_match = re.search(date_pattern, pdf_text)
    if date_match:
        metadata['date'] = datetime.strptime(date_match.group(1), '%d/%m/%Y').strftime('%Y-%m-%d')

    # Extracting court name
    court_match = re.search(court_pattern, pdf_text)
    if court_match:
        metadata['court'] = court_match.group(1).strip()

    # Extracting petitioner
    petitioner_match = re.search(petitioner_pattern, pdf_text, re.DOTALL)
    if petitioner_match:
          metadata['petitioner']  = petitioner_match.group(1).strip()
        

    # Extracting respondent
    respondent_match = re.search(respondent_pattern, pdf_text, re.DOTALL)
    if respondent_match:
        metadata['respondent'] = respondent_match.group(1).strip()

    # Extracting bench members
    benches_match = re.search(benches_pattern, pdf_text, re.DOTALL)
    if benches_match:
        benches_text = benches_match.group(1).strip()

        #Split benches text by "BENCH:" and process each bench separately
        #benches = benches_text.split("BENCH:")
        #benches = [bench.strip().split("\n") for bench in benches if bench.strip()]

        metadata['bench'] = benches_text

    # Extracting citations
    citation_match = re.search(citation_pattern, pdf_text, re.DOTALL)
    
    if citation_match:
        citation_text = citation_match.group(1).strip()
        
        #citations = [line.strip() for line in citation_text.splitlines() if line.strip()]

        metadata['citation'] = citation_text

    return metadata



def read_pdf(file_path):
    """ Extract texts from pdf file
        :param file_path [string]: path of pdf file
        :return [str]: extracted text from the pdf file
    """

    pdf_file = PyPDFLoader(file_path)
    pdf_data = pdf_file.load()

    text_data = ""
    for page in pdf_data:
      text_data += preprocess_text(page.page_content)

    metadata =  extract_metadata(pdf_data[0].page_content, file_path)

    return text_data, metadata


def create_docs(data, metadata,  chunk_size = 2000, chunk_overlap = 200, length_function = len):
    """Split the text into chunks.

      :param data [str]:The text data to be split into chunks.
      :param chunk_size [int], optional: The maximum size of each chunk of text after splitting. Default is 2000.
      :param chunk_overlap [int], optional: The number of characters from the end of one chunk to be included at the beginning of the next chunk. Default is 200.
      :param length_function [function], optional: The function used to measure the length of the text. Default is len.

      :return [list]: A list of text chunks.
      """

    text_splitter = NLTKTextSplitter(
      chunk_size = chunk_size,       # legalBERT have token limit 512, 512 * 5 char = 2560 ~ 2000 (average 5 char /word)
      chunk_overlap = chunk_overlap,
      length_function = length_function,
    )

    chunks = text_splitter.split_text(data)
    documents = [
        Document(page_content=chunk, metadata={**metadata, 'id': str(uuid.uuid4())})
        for chunk in chunks
    ]
    

    return documents