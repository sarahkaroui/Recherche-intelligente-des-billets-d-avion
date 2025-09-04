from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from langgraph.prebuilt import create_react_agent
model = OllamaLLM(model="llama3.2")

TEMPLATE = """
Tu es un assistant intelligent spécialisé dans le filtrage des billets d'avion selon les critères.

Voici les variables et ce que tu dois faire avec chacune :
- Escale = {escale}
  * Si 'oui' → propose seulement les vols sans escale.
  * Sinon → propose les vols avec au moins une escale.
- Budget max = {prix}
  * Ne propose pas de vol au-dessus de ce budget.
- Notes / préférences = {description}
  * Analyse ce texte libre (ex: vol à partir de 15h tu dois prosposer des vols dont la date de depart est à partir de 15h)
  ex: vol avec une certaine compagnie aerienne, tu dois proposer des vols avec cette compagnie
--- Liste des billets disponibles ---
{reviews}
----La sortie-----
✅ Vols trouvés selon vos critères :

1. [Compagnie] | Départ : [Aéroport Heure] | Arrivée : [Aéroport Heure] | Escales : [n] | Prix : [xxx TND]
   Ne sors **rien d’autre** que ce format, et surtout n ajoute aucun commentaires 

"""


prompt = ChatPromptTemplate.from_template(TEMPLATE)
chain = prompt | model


def format_criteres_for_prompt(data: dict) -> str:
    labels = {
        "de": "Départ (IATA)",
        "arivee": "Arrivée (IATA)",
        "date": "Date",
        "Escala": "Escales max",
        "PrixMax": "Budget max (TND)",
        "Description": "Notes / préférences",
    }
    lines = []
    for k, v in data.items():
        if v in (None, "", []):
            continue
        lines.append(f"- {labels.get(k, k)}: {v}")
    return "\n".join(lines) if lines else "(aucun critère)"


def run_flight_agent(data: dict, flights: list) -> dict:
    """
    Prend les critères (data) + liste complète des vols scrappés (flights)
    et laisse le LLM filtrer sans retriever.
    """
    # ✅ Debug : vérifier le contenu de data
    print("DEBUG data reçu par agent:", data)
    # 1) Convertir tous les vols en texte
    reviews_text = "\n".join(
        f"- {f.get('Airline Company', '?')} | "
        f"Durée: {f.get('Flight Duration', '?')} | "
        f"Escales: {f.get('Stops', '?')} | "
        f"Prix: {f.get('Price', '?')} | "
        f"Départ: {f.get('Departure Time', '?')} | "
        f"Arrivée: {f.get('Arrival Time', '?')}"
        for f in flights
    )

    # 2) Préparer les critères
    criteres_text = format_criteres_for_prompt(data)
    escale_text = data.get("Escale") or data.get("Escala") or "non précisé"
    print(escale_text)
    prix = data.get("Prix_Max") or data.get("PrixMax") or "non précisé"
    print(prix)
    description = data.get("Notes") or data.get("Description") or "aucune"
    print(description)
    # 3) Appel du LLM
    out = chain.invoke({"reviews": reviews_text, "escale":escale_text, "prix": prix,"description": description,})
    print(out)
    return {
        "total_count": len(flights),
        "preview_flights": flights[:5],  
        "answer": str(out),
    }
