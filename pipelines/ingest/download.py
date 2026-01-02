import os
import csv
import random
import faker
from pathlib import Path

# Use faker for realistic text
fake = faker.Faker()

DATA_DIR = Path("data/raw")

def generate_synthetic_data(num_news=500, num_users=100, num_behaviors=1000):
    print(f"Generating synthetic MIND data: {num_news} items, {num_users} users...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate News (items)
    # columns: item_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
    print("Generating news.tsv...")
    with open(DATA_DIR / "news.tsv", "w", newline='') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        
        for i in range(num_news):
            nid = f"N{i}"
            cat = random.choice(['sports', 'finance', 'music', 'tech', 'weather'])
            subcat = random.choice(['general', 'top', 'breaking'])
            title = fake.sentence(nb_words=10).replace('\t', ' ')
            abstract = fake.paragraph(nb_sentences=2).replace('\t', ' ')
            url = f"http://example.com/{nid}"
            
            writer.writerow([nid, cat, subcat, title, abstract, url, "[]", "[]"])
            
    # 2. Generate Behaviors (interactions)
    # columns: impression_id, user_id, time, history, impressions
    # history: "N1 N5 ..."
    # impressions: "N0-0 N1-1 ..."
    print("Generating behaviors.tsv...")
    with open(DATA_DIR / "behaviors.tsv", "w", newline='') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        
        for i in range(num_behaviors):
            impid = i
            uid = f"U{random.randint(0, num_users)}"
            ts = "11/11/2019 9:00:00 AM" # fixed format
            
            # History
            hist_len = random.randint(0, 10)
            history = " ".join([f"N{random.randint(0, num_news-1)}" for _ in range(hist_len)])
            
            # Impressions
            imp_len = random.randint(2, 20)
            # Ensure at least one positive click sometimes
            imps = []
            for _ in range(imp_len):
                # 10% click rate
                clicked = 1 if random.random() < 0.1 else 0
                item = f"N{random.randint(0, num_news-1)}"
                imps.append(f"{item}-{clicked}")
                
            impressions = " ".join(imps)
            
            writer.writerow([impid, uid, ts, history, impressions])
            
    print("Synthetic data generation complete.")

def main():
    generate_synthetic_data()

if __name__ == "__main__":
    main()
