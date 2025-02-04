import pandas as pd
from sqlalchemy import create_engine
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

host = os.getenv('DB_HOST')
database = os.getenv('DB_NAME')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
    
def load_fifa_data():
    return pd.read_csv('players_20.csv')

def create_player_description(row):
    return (f"El jugador {row['short_name']} con edad {row['age']} "
            f"juega en la posición {row['team_position']} "
            f"para el equipo {row['club']}. "
            f"Sus principales atributos son: "
            f"velocidad {row['pace']}, "
            f"tiro {row['shooting']}, "
            f"pase {row['passing']}, "
            f"regate {row['dribbling']}, "
            f"defensa {row['defending']}, "
            f"físico {row['physic']}.")

def main():
    # Initialize HuggingFace embeddings
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Database connection
    db_params = {
        'host': host,
        'database': database,
        'user': user,
        'password': password
    }
    
    # Load and prepare data
    df = load_fifa_data()
    required_columns = ['short_name', 'age', 'team_position', 'club',
                       'pace', 'shooting', 'passing', 'dribbling', 
                       'defending', 'physic']
    
    df = df[required_columns].fillna({
        'team_position': 'Unknown', 'club': 'Unknown',
        'pace': 0, 'shooting': 0, 'passing': 0,
        'dribbling': 0, 'defending': 0, 'physic': 0
    })
    
    # Generate descriptions and embeddings
    descriptions = [create_player_description(row) for _, row in df.iterrows()]
    
    batch_size = 50
    all_embeddings = []
    for i in tqdm(range(0, len(descriptions), batch_size)):
        batch = descriptions[i:i + batch_size]
        embeddings = embeddings_model.embed_documents(batch)
        all_embeddings.extend(embeddings)
    
    # Database setup
    engine = create_engine(f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_params['database']}")
    embedding_dim = len(all_embeddings[0])
    
    with engine.connect() as conn:
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS fifa_player_embeddings (
            id SERIAL PRIMARY KEY,
            player_name VARCHAR(255),
            embedding vector({embedding_dim})
        );
        """)
        
        # Insert data
        for (_, row), embedding in zip(df.iterrows(), all_embeddings):
            conn.execute("""
            INSERT INTO fifa_player_embeddings 
            (player_name, embedding)
            VALUES (%s, %s)
            """, (
                row['short_name'], embedding
            ))

if __name__ == "__main__":
    main()