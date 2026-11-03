from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / "data" / "listings").glob("*_listings.csv"))
if not files:
    raise SystemExit("Place *_listings.csv files in data/listings first.")
COUNTRY = {
 "Amsterdam":"Netherlands","Antwerp":"Belgium","Athens":"Greece","Barcelona":"Spain", "Bergamo":"Italy","Berlin":"Germany", "Bologna":"Italy", "Bordeaux":"France", "Bristol":"United Kingdom", "Brussels":"Belgium", "Copenhagen":"Denmark", "Crete":"Greece", "Dublin":"Ireland", "Edinburgh":"United Kingdom", "Euskadi":"Spain", "Florence":"Italy", "Geneva":"Switzerland", "Ghent":"Belgium", "Girona":"Spain", "Greater Manchester":"United Kingdom", "Lisbon":"Portugal", "London":"United Kingdom", "Lyon":"France", "Madrid":"Spain", "Mallorca":"Spain", "Menorca":"Spain", "Milan":"Italy", "Munich":"Germany", "Naples":"Italy", "Oslo":"Norway", "Paris":"France", "Porto":"Portugal", "Prague":"Czech Republic", "Puglia":"Italy", "Riga":"Latvia", "Rome":"Italy", "Rotterdam":"Netherlands", "Sevilla":"Spain", "Sicily":"Italy", "South Aegean":"Greece", "The Hague":"Netherlands", "Thessaloniki":"Greece", "Valencia":"Spain", "Vaud":"Switzerland", "Venice":"Italy", "Vienna":"Austria"}
frames=[]
for file in files:
    df=pd.read_csv(file,low_memory=False)
    loc=file.stem.removesuffix('_listings').replace('_',' ')
    df['location']=loc;df['country']=COUNTRY.get(loc,'Unknown');df['source_file']=file.name
    frames.append(df)
    print(file.name,len(df))
out=ROOT/'data'/'processed'/'listings_master.parquet'
out.parent.mkdir(parents=True,exist_ok=True)
pd.concat(frames,ignore_index=True,sort=False).to_parquet(out,index=False)
print('Saved',out)
