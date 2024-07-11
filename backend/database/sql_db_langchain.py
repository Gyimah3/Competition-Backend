from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy.orm import sessionmaker
from functools import lru_cache

@lru_cache(maxsize=1)
def get_engine():
    return create_engine("sqlite:///retail_data.db")

def init_db():
    df = pd.read_csv("database/retail_data.csv")
    engine = get_engine()
    df.to_sql("retail_data", engine, index=False, if_exists="replace")

db = SQLDatabase(engine=get_engine())

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
init_db()
