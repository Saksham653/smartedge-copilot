from backend.export_service import export_note_markdown, NoteNotFoundError

DB_PATH = "data/metrics.db"

def run_test():
    try:
        # Change type and id to test different notes
        markdown = export_note_markdown(DB_PATH, "research", 2)
        print("\n--- EXPORTED MARKDOWN ---\n")
        print(markdown)

    except NoteNotFoundError as e:
        print("Error:", e)

    except Exception as e:
        print("Unexpected error:", e)


if __name__ == "__main__":
    run_test()