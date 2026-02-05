"""
MongoDB utility module.

Provides a reusable MongoDB client and helper functions
to fetch collections as Pandas DataFrames with optional
query-based filtering (e.g. batch_tag).
"""

import os
import logging
from typing import Optional, Dict, Any
import pandas as pd
import pymongo
from dotenv import load_dotenv

from constant import DATABASE_NAME, MONGODB_URL_KEY
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MongoDBClient:
    """
    MongoDB client wrapper to manage connections and collections.
    """

    def __init__(self, database_name: str = DATABASE_NAME):
        self.mongo_url = os.getenv(MONGODB_URL_KEY, "mongodb://localhost:27017")
        self.client = pymongo.MongoClient(self.mongo_url)

        self.database_name = os.getenv("MONGO_DB_NAME", database_name)
        self.database = self.client[self.database_name]

        logger.info(f"Connected to MongoDB database: {self.database_name}")

    def get_collection(self, collection_name: str):
        """
        Retrieve a MongoDB collection handle.
        """
        if collection_name not in self.database.list_collection_names():
            logger.warning(
                f"Collection '{collection_name}' not found in database "
                f"'{self.database_name}'. Available collections: "
                f"{self.database.list_collection_names()}"
            )

        return self.database[collection_name]

    # Fetch documents from a MongoDB collection and return as a DataFrame.
    
    def fetch_as_dataframe(
        self,
        collection_name: str,
        query: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        

        if query is None:
            query = {}

        collection = self.get_collection(collection_name)

        logger.info(
            f"Fetching data from collection='{collection_name}' "
            f"with query={query}"
        )

        cursor = collection.find(query, {"_id": 0})
        data = list(cursor)

        df = pd.DataFrame(data)

        if df.empty:
            logger.warning(
                f"No documents found in collection='{collection_name}' "
                f"for query={query}"
            )

        return df


def get_data_as_dataframe(
    collection_name: str,
    query: Optional[Dict[str, Any]] = None,
    database_name: str = DATABASE_NAME,
) -> pd.DataFrame:
    """
    Convenience helper to fetch MongoDB data as a Pandas DataFrame.

    This function abstracts away client creation and should be used
    by ZenML ingestion steps and other pipelines.

    """
    try:
        mongo_client = MongoDBClient(database_name=database_name)
        return mongo_client.fetch_as_dataframe(
            collection_name=collection_name,
            query=query,
        )
    except Exception as exc:
        logger.error(
            f"Failed to load data from MongoDB collection='{collection_name}' "
            f"with query={query} | Error: {exc}"
        )
        raise
