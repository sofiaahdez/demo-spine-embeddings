import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
import numpy as np

# Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embedding(row):
    """Generate embedding from template string"""
    template = f"El usuario con nombre {row['NAME']} tiene como CURP {row['CURP']}, su RFC es {row['RFC']} y su número de teléfono es {row['TEL']}"
    return model.encode(template)

def main():
    # AlloyDB connection 
    db_params = {
        'host': 'your-alloydb-host',
        'database': 'your-database',
        'user': 'your-username',
        'password': 'your-password'
    }
    
    # Read data from CSV into DataFrame
    df = pd.read_csv('your_data.csv')
    
    # Generate embeddings for each row
    df['embedding'] = df.apply(create_embedding, axis=1)
    
    # AlloyDB connection
    engine = create_engine(f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_params['database']}")
    
    # Create table
    with engine.connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS user_embeddings (
            id SERIAL PRIMARY KEY,
            name TEXT,
            curp TEXT,
            rfc TEXT,
            tel TEXT,
            embedding vector(384)  -- Dimension size for all-MiniLM-L6-v2
        );
        """)
    
    # Insert data and embeddings into AlloyDB
    with engine.connect() as conn:
        for _, row in df.iterrows():
            embedding_list = row['embedding'].tolist()
            conn.execute("""
            INSERT INTO user_embeddings (name, curp, rfc, tel, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """, (row['NAME'], row['CURP'], row['RFC'], row['TEL'], embedding_list))

if __name__ == "__main__":
    main()