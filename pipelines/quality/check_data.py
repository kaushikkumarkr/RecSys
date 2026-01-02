import great_expectations as ge
import pandas as pd
from pathlib import Path
import sys

DATA_DIR = Path("data/raw")

def validate_raw_data():
    print("Running Great Expectations on Raw Data...")
    
    # Validate News
    news_path = DATA_DIR / "news.tsv"
    if not news_path.exists():
        print(f"Error: {news_path} not found.")
        sys.exit(1)
        
    df_news = ge.read_csv(
        news_path, 
        sep="\t",
        header=None,
        names=["item_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"],
        quoting=3
    )
    
    print("\n[News Validation]")
    # Check 1: item_id should be unique and not null
    res1 = df_news.expect_column_values_to_be_unique("item_id")
    res2 = df_news.expect_column_values_to_be_null("item_id", mostly=0.0) # Should be 0% null
    
    # Check 2: Title should not be null
    res3 = df_news.expect_column_values_to_not_be_null("title")
    
    results = [res1, res2, res3]
    success = all(r.success for r in results)
    
    if success:
        print("✅ News Table: Passed Basic Checks")
    else:
        print("❌ News Table: Failed Checks")
        for r in results:
            if not r.success:
                print(r)
                
    # Validate Behaviors
    behaviors_path = DATA_DIR / "behaviors.tsv"
    df_behaviors = ge.read_csv(
        behaviors_path,
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"],
        quoting=3
    )
    
    print("\n[Behaviors Validation]")
    # Check 1: impression_id unique
    res_b1 = df_behaviors.expect_column_values_to_be_unique("impression_id")
    # Check 2: user_id not null
    res_b2 = df_behaviors.expect_column_values_to_not_be_null("user_id")
    
    results_b = [res_b1, res_b2]
    success_b = all(r.success for r in results_b)
    
    if success_b:
        print("✅ Behaviors Table: Passed Basic Checks")
    else:
        print("❌ Behaviors Table: Failed Checks")

if __name__ == "__main__":
    validate_raw_data()
