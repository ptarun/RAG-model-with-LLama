
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
def create_vector_db(documents):

    """  Create a new vector db from documents using Chroma and legalBERT
        :param documents [list]: list of documents
        :return [vector db]: created vector db using documents
    """


    embeddings = HuggingFaceEmbeddings(model_name = "nlpaueb/legal-bert-base-uncased")

    vectorstore = Chroma.from_documents(documents, embeddings)

    return vectorstore


def update_vector_db(documents, vectorstore):
    """ Update vector db
        :param documents [list]: list of documents
        :param vectorstore [vector db]: vector db 
        :return : 
    """
  
    vectorstore.add_documents(documents)
    return


def similarty_search_by_vector(vectorstore, query, k = 5):
    """ search db using vector
        :param vectorstore [vector db]: vector db 
        :param query [str]:
        :param k [int]: number of results to be returned
        :return [list]: list of similar docs
    """

    embeddings = HuggingFaceEmbeddings(model_name = "nlpaueb/legal-bert-base-uncased")
    query_embedding = embeddings.embed_query(query)
    similar_docs = vectorstore.similarity_search_by_vector(query_embedding,  k = k)
    return similar_docs

