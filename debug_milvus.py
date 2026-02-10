
from pymilvus import connections, Collection, utility
import os
from dotenv import load_dotenv

def check_milvus():
    load_dotenv()
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    collection_name = os.getenv("MILVUS_MEETING_COLLECTION_NAME", "meeting_embeddings")
    
    connections.connect(host=milvus_host, port=milvus_port)
    
    if not utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' does not exist.")
        return
        
    collection = Collection(collection_name)
    collection.load()
    
    num_entities = collection.num_entities
    print(f"Total entities in '{collection_name}': {num_entities}")
    
    # Check for the specific ID found in MongoDB
    target_id = "6dc86bab-758f-4738-88ff-f9ad4cc4317c"
    res = collection.query(expr=f"mongo_id == '{target_id}'", output_fields=["id", "mongo_id"])
    if res:
        print(f"FOUND target ID {target_id} in Milvus: {res}")
    else:
        print(f"NOT found target ID {target_id} in Milvus.")

    results = collection.query(expr="id >= 0", output_fields=["id", "mongo_id"], limit=20)
    
    connections.disconnect("default")

if __name__ == "__main__":
    check_milvus()
