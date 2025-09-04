"""
googleflight_scraping.py
------------------------
Ce module s’occupe de :
1. Scraper les résultats de vols sur Google Flights via Playwright
2. Nettoyer les données (suppression doublons, structuration)
3. Exposer une API FastAPI pour récupérer les vols en fonction
   des critères utilisateur (départ, arrivée, date).
"""
import asyncio
import uvicorn
import csv
import base64
from playwright.async_api import async_playwright
from typing import List, Dict, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from agent import run_flight_agent
from word import save_llm_to_word
# === Création API FastAPI ===
app = FastAPI(title="Flight Scraper API")
# === Génération URL Google Flights ===
class FlightURLBuilder:
    """Class to handle flight URL creation with base64 encoding."""
    """Construit l’URL encodée Google Flights à partir du départ, arrivée et date."""
    
    @staticmethod
    def _create_one_way_bytes(departure: str, destination: str, date: str) -> bytes:
        """Crée la base binaire pour un vol aller simple."""
        return (
            b'\x08\x1c\x10\x02\x1a\x1e\x12\n' + date.encode() +
            b'j\x07\x08\x01\x12\x03' + departure.encode() +
            b'r\x07\x08\x01\x12\x03' + destination.encode() +
            b'@\x01H\x01p\x01\x82\x01\x0b\x08\xfc\x06`\x04\x08'
        )
    
    @staticmethod
    def _modify_base64(encoded_str: str) -> str:
        """Ajoute des underscores dans la chaîne base64 (format Google Flights)."""
        insert_index = len(encoded_str) - 6
        return encoded_str[:insert_index] + '_' * 7 + encoded_str[insert_index:]

    @classmethod
    def build_url(
        cls,
        departure: str,
        destination: str,
        departure_date: str
    ) -> str:
        """Retourne l’URL finale de recherche Google Flights."""
        
        flight_bytes = cls._create_one_way_bytes(departure, destination, departure_date)
        base64_str = base64.b64encode(flight_bytes).decode('utf-8')
        modified_str = cls._modify_base64(base64_str)
        return f'https://www.google.com/travel/flights/search?tfs={modified_str}'

# === Gestion du navigateur (Playwright) ===
async def setup_browser():
    """Ouvre un navigateur Chromium avec Playwright."""
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=False) # False = pour voir le navigateur
    page = await browser.new_page()
    return p, browser, page


async def extract_flight_element_text(flight, selector: str, aria_label: Optional[str] = None) -> str:
    """Cherche un élément dans un vol et retourne son texte."""
    if aria_label:
        element = await flight.query_selector(f'{selector}[aria-label*="{aria_label}"]')
    else:
        element = await flight.query_selector(selector)
    return await element.inner_text() if element else "N/A"


async def scrape_flight_info(flight) -> Dict[str, str]:
    """Récupère toutes les infos principales d’un vol (compagnie, prix, horaires, etc)."""
    departure_time = await extract_flight_element_text(flight, 'div.wtdjmc')
    arrival_time = await extract_flight_element_text(flight, 'div.XWcVob')
    airline = await extract_flight_element_text(flight, ".sSHqwe")
    duration = await extract_flight_element_text(flight, "div.gvkrdb")
    stops =  await extract_flight_element_text(flight, "div.EfT7Ae span.ogfYpf")
    price =  await extract_flight_element_text(flight, "div.FpEdX span")
    co2_emissions =  await extract_flight_element_text(flight, "div.O7CXue")
    emissions_variation =  await extract_flight_element_text(flight, "div.N6PNV")
    return {
        "Departure Time": departure_time,
        "Arrival Time": arrival_time,
        "Airline Company": airline,
        "Flight Duration": duration,
        "Stops": stops,
        "Price": price
    }
# === Nettoyage et sauvegarde CSV ===
def clean_csv(filename: str):
    """Nettoie les caractères spéciaux dans le CSV final."""
    data = pd.read_csv(filename, encoding="utf-8")
    
    def clean_text(value):
        if isinstance(value, str):
            return value.replace('Â', '').replace(' ', ' ').replace('Ã', '').replace('¶', '').strip()
        return value

    cleaned_data = data.applymap(clean_text)
    cleaned_file_path = f"{filename}"
    cleaned_data.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned CSV saved to: {cleaned_file_path}")

def save_to_csv(data: List[Dict[str, str]], filename: str = "Scraped_Data.csv") -> None:
    """Sauvegarde les résultats en CSV sans doublons."""
    if not data:
        return
    
    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)  #  enlève les doublons exacts
    df.to_csv(filename, index=False, encoding="utf-8")
    
    # Nettoyage du CSV
    clean_csv(filename)
# === Scraping global (une recherche complète) ===
async def scrape_flight_data(one_way_url):
    """Scrape tous les vols d’une URL Google Flights."""
    flight_data = []

    playwright, browser, page = await setup_browser()
    
    try:
        await page.goto(one_way_url)
        
        # Wait for flight data to load
        await page.wait_for_selector(".pIav2d")  # attendre chargement des résultats
        
        # Get all flights and extract their information
        flights = await page.query_selector_all(".pIav2d")
        for flight in flights:
            flight_info = await scrape_flight_info(flight)
            flight_data.append(flight_info)
        
        # Save the extracted data in CSV format
        save_to_csv(flight_data)
            
    finally:
        await browser.close()
        await playwright.stop()
    return flight_data
# === Endpoint FastAPI ===
@app.post("/recherche-billets")
async def search_flights_iata(request: Request):
    """Endpoint principal: reçoit départ, arrivée, date → renvoie résultats vols."""
    data = await request.json()
    departure = (data.get("DE") or "").upper()
    destination = (data.get("AR") or "").upper()
    date = data.get("date")
    if not (departure and destination and date):
        raise HTTPException(status_code=400, detail="Champs requis: DE, AR, date (YYYY-MM-DD)")
    one_way_url = FlightURLBuilder.build_url(departure, destination, date)
    results = await scrape_flight_data(one_way_url)
     #save_to_csv(results, "Scraped_Data.csv")
    agent_out = run_flight_agent(data, results)
    print(agent_out)
    docx_path = save_llm_to_word(agent_out.get("answer", ""), path="LesVols.docx")

    return {"url": one_way_url, "csv": "Scraped_Data.csv", "flights": results}

# Démarrage local: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005) 