from backend.meeting_db import save_meeting_note
from backend.database import get_connection


DB_PATH = "data/metrics.db"


def count_tasks():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tasks")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def run_test():
    before = count_tasks()
    print("Tasks before:", before)

    meeting_id = save_meeting_note(
        title="Automation Test Meeting",
        transcript="Discussion transcript",
        summary="Discussed backend improvements",
        key_topics="Automation, Tasks",
        action_items="1. Prepare documentation\n2. Review PR\n3. Deploy to staging",
        deadlines="Tomorrow",
        decisions="Proceed with automation",
        total_tokens=100,
        latency_ms=50.0,
        cost=0.01,
        model="test-model"
    )

    print("Meeting saved with ID:", meeting_id)

    after = count_tasks()
    print("Tasks after:", after)

    print("Tasks created:", after - before)


if __name__ == "__main__":
    run_test()