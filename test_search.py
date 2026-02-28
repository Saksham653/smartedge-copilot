from backend.knowledge_hub import search_knowledge_hub

DB_PATH = "data/metrics.db"  # ⚠️ Put your actual DB file name

def run_test():
    query = "AI"   # change this to something that exists in your notes
    results = search_knowledge_hub(DB_PATH, query)

    print(f"\nFound {len(results)} results:\n")

    for r in results:
        print("-" * 50)
        print("Type:", r["type"])
        print("ID:", r["id"])
        print("Title:", r["title"])
        print("Preview:", r["preview"])

if __name__ == "__main__":
    run_test()