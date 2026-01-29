# Utility functions and classes for database interactions.
# Handles MongoDB connection and data retrieval.

"""MongoDB client utilities to export collections into Pandas DataFrames."""

import os
import logging
import pandas as pd
import pymongo
from dotenv import load_dotenv

from constant import DATABASE_NAME, MONGODB_URL_KEY

load_dotenv()

logger = logging.getLogger(__name__)

class MongoDBClient:
    def __init__(self, database_name: str = DATABASE_NAME):
        self.mongo_url = os.getenv(MONGODB_URL_KEY, "mongodb://localhost:27017")
        self.client = pymongo.MongoClient(self.mongo_url)
        self.database_name = os.getenv("MONGO_DB_NAME", database_name)
        self.database = self.client[self.database_name]
        logger.info(f"Connected to MongoDB database: {self.database_name}")

    def export_collection_as_dataframe(self, collection_name: str) -> pd.DataFrame:
        if collection_name not in self.database.list_collection_names():
            logger.warning(f"Collection '{collection_name}' not found in database '{self.database_name}'. Available collections: {self.database.list_collection_names()}")
        
        collection = self.database[collection_name]
        df = pd.DataFrame(list(collection.find()))
        
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        
        if df.empty:
            logger.warning(f"Dataframe created from collection '{collection_name}' is empty.")
            
        return df

def get_data_as_dataframe(collection_name: str, database_name: str = DATABASE_NAME) -> pd.DataFrame:
    """
    Convenience function to get data from MongoDB as a DataFrame without manually handling the client.
    
    Args:
        collection_name: Name of the collection to export.
        database_name: Name of the database to connect to.
        
    Returns:
        pd.DataFrame: DataFrame containing the collection data.
    """
    try:
        mongo_client = MongoDBClient(database_name=database_name)
        return mongo_client.export_collection_as_dataframe(collection_name=collection_name)
    except Exception as e:
        logger.error(f"Error checking data via helper function: {e}")
        raise e

