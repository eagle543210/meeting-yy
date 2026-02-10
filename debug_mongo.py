
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

async def check_mongo():
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "meeting_db")
    collection_name = os.getenv("MONGO_TRANSCRIPT_COLLECTION_NAME", "meeting_transcripts")
    
    client = AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    
    with open("m:/meeting/debug_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Using DB: {db_name}\n")
        f.write(f"Checking collection: {collection_name}\n")
        
        coll = db[collection_name]
        count = await coll.count_documents({})
        f.write(f"Total documents in '{collection_name}': {count}\n")
        
        collections = await db.list_collection_names()
        f.write(f"Available collections: {collections}\n")

        if count > 0:
            docs = await coll.find().sort("timestamp", -1).limit(10).to_list(length=None)
            for doc in docs:
                f.write(f"ID: {doc['_id']} (Type: {type(doc['_id'])})\n")
        
        # Search specifically for one of the missing IDs
        missing_id = "f2403862-4137-4480-ba83-4c9a0986d392"
        f.write(f"\nSearching for {missing_id} in all collections...\n")
        for cname in collections:
            doc = await db[cname].find_one({"_id": missing_id})
            if doc:
                f.write(f"FOUND in {cname}: {doc}\n")
            else:
                f.write(f"NOT found in {cname}\n")

    client.close()

if __name__ == "__main__":
    asyncio.run(check_mongo())
