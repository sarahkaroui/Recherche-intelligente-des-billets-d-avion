from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd, os, re

# --- helpers ---
def to_minutes(s):
    if not isinstance(s, str): return None
    s = s.replace('\xa0', ' ')
    h = re.search(r'(\d+)\s*h', s)
    m = re.search(r'(\d+)\s*min', s)
    return (int(h.group(1)) if h else 0) * 60 + (int(m.group(1)) if m else 0)

def to_price_tnd(s):
    if not isinstance(s, str): return None
    s = s.replace('\xa0', ' ')
    digits = re.sub(r'[^\d]', '', s)
    return int(digits) if digits else None

def stops_to_int(s):
    if not isinstance(s, str): return None
    s = s.lower().replace('\xa0', ' ')
    if 'sans escale' in s: return 0
    m = re.search(r'(\d+)\s*escale', s)
    return int(m.group(1)) if m else None

def arrival_next_day(s): 
    return isinstance(s, str) and ('+1' in s)

def strip_plus1(s):
    return s.replace('+1', '') if isinstance(s, str) else s

# --- charge le BON CSV de vols ---
df = pd.read_csv("Scraped_Data.csv", encoding="utf-8-sig")
df.columns = [c.replace('\ufeff', '').replace('\xa0', ' ').strip() for c in df.columns]

required = ['Departure Time','Arrival Time','Airline Company','Flight Duration','Stops','Price']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes: {missing} | Colonnes trouvées: {list(df.columns)}")

# --- colonnes dérivées (AVANT la boucle) ---
df['Price_num']     = df['Price'].apply(to_price_tnd)
df['Duration_min']  = df['Flight Duration'].apply(to_minutes)
df['Stops_num']     = df['Stops'].apply(stops_to_int)
df['Arrivee_plus1'] = df['Arrival Time'].apply(arrival_next_day)
df['Arrival_clean'] = df['Arrival Time'].apply(strip_plus1)

# --- embeddings & vectordb ---
embeddings  = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chroma_flights_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    docs, ids = [], []
    for i, row in df.iterrows():
        page_content = (
            f"Vol {row['Airline Company']} | Départ {row['Departure Time']} "
            f"| Arrivée {row['Arrival_clean']}{' (arrivée le lendemain)' if row['Arrivee_plus1'] else ''} "
            f"| Durée {row['Flight Duration']} | {row['Stops']} | Prix {row['Price']}."
        )
        metadata = {
            "airline": str(row["Airline Company"]),
            "departure_time": str(row["Departure Time"]),
            "arrival_time": str(row["Arrival_clean"]),
            "arrival_next_day": bool(row["Arrivee_plus1"]),
            "duration_min": int(row["Duration_min"]) if pd.notna(row["Duration_min"]) else None,
            "stops": int(row["Stops_num"]) if pd.notna(row["Stops_num"]) else None,
            "price_tnd": int(row["Price_num"]) if pd.notna(row["Price_num"]) else None,
        }
        docs.append(Document(page_content=page_content, metadata=metadata, id=str(i)))
        ids.append(str(i))

vector_store = Chroma(
    collection_name="billets_avion",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=docs, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 100})
