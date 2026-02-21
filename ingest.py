"""
Ingest one or more webpages into the ChromaDB vector database.
Run this script SEPARATELY before starting the chatbot.

Usage:
    python ingest.py
"""

import time
from tqdm import tqdm
from rag_core import ingest_urls, CHROMA_DB_DIR
from logger_setup import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# ============================================================================
# ADD YOUR URLS HERE as ("Title", "url", "description") tuples
# ============================================================================
URLS = [
    ("Branch 2 - Council",
     "https://www.ul.edu.lb/en/colleges-faculties-branches-council/343/346/Faculty%20of%20Engineering",
     "Council members and structure of Branch 2"),
    ("Dean",
     "https://ul.edu.lb/en/colleges-faculties--dean/343/Faculty%20of%20Engineering",
     "Current dean and list of past deans of the Faculty of Engineering"),
    ("Deanship",
     "https://ul.edu.lb/en/colleges-faculties--deanship/343/Faculty%20of%20Engineering",
     "Unit council of the Faculty of Engineering"),
    ("Majors",
     "https://ul.edu.lb/en/colleges-faculties-majors/343/Faculty%20of%20Engineering",
     "Diplomas, research masters, engineering majors, and overview of each program offered by the faculty"),
    ("Guide & Internal System",
     "https://ul.edu.lb/en/colleges-faculties--guide--internalsystem/343/Faculty%20of%20Engineering",
     "Internal system rules, regulations, and student guide"),
    ("Branch 1 - Overview",
     "https://ul.edu.lb/en/colleges-faculties-branches-overview/343/345/Faculty%20of%20Engineering",
     "History and contact information of Branch 1 (Tripoli)"),
    ("Branch 1 - Admin Office",
     "https://ul.edu.lb/en/colleges-faculties-branches-admin-office/343/345/Faculty%20of%20Engineering",
     "Current director and list of successive directors of Branch 1"),
    ("Branch 1 - Council",
     "https://ul.edu.lb/en/colleges-faculties-branches-council/343/345/Faculty%20of%20Engineering",
     "Branch council members of Branch 1"),
    ("Branch 1 - Academic Departments",
     "https://ul.edu.lb/en/colleges-faculties-branches-academic-departments/343/345/Faculty%20of%20Engineering",
     "Academic departments of Branch 1"),
    ("Branch 1 - Administrative Departments",
     "https://ul.edu.lb/en/colleges-faculties-branches-administrative-departments/343/345/Faculty%20of%20Engineering",
     "Administrative departments and units of Branch 1"),
    ("Branch 2 - Overview",
     "https://ul.edu.lb/en/colleges-faculties-branches-overview/343/346/Faculty%20of%20Engineering",
     "General overview of Branch 2 (Roumieh)"),
    ("Branch 2 - Admin Office",
     "https://ul.edu.lb/en/colleges-faculties-branches-admin-office/343/346/Faculty%20of%20Engineering",
        "Current director and list of successive directors of Branch 2"),
    ("Branch 2 - Council",
     "https://ul.edu.lb/en/colleges-faculties-branches-council/343/346/Faculty%20of%20Engineering",
        "Branch council members of Branch 2"),
    ("Branch 2 - Academic Departments",
     "https://ul.edu.lb/en/colleges-faculties-branches-academic-departments/343/346/Faculty%20of%20Engineering",
        "Academic departments of Branch 2"),
    ("Branch 2 - Administrative Departments",
     "https://ul.edu.lb/en/colleges-faculties-branches-administrative-departments/343/346/Faculty%20of%20Engineering",
        "Administrative departments and units of Branch 2"),
    ("Branch 3 - Overview",
     "https://ul.edu.lb/en/colleges-faculties-branches-overview/343/347/Faculty%20of%20Engineering",
     "General overview of Branch 3 (Hadath)"),
    ("Branch 3 - Admin Office",
     "https://ul.edu.lb/en/colleges-faculties-branches-admin-office/343/347/Faculty%20of%20Engineering",
        "Current director and list of successive directors of Branch 3"),
    ("Branch 3 - Council",
     "https://ul.edu.lb/en/colleges-faculties-branches-council/343/347/Faculty%20of%20Engineering",
        "Branch council members of Branch 3"),
    ("Branch 3 - Academic Departments",
     "https://ul.edu.lb/en/colleges-faculties-branches-academic-departments/343/347/Faculty%20of%20Engineering",
        "Academic departments of Branch 3"),
    ("Branch 3 - Administrative Departments",
     "https://ul.edu.lb/en/colleges-faculties-branches-administrative-departments/343/347/Faculty%20of%20Engineering",
        "Administrative departments and units of Branch 3"),
]
# ============================================================================


def main():
    start_time = time.time()

    logger.info("%s", "=" * 60)
    logger.info("  Webpage(s) --> ChromaDB Ingestion")
    logger.info("%s", "=" * 60)
    logger.info("\n  URLs : %d", len(URLS))
    for i, (title, url, desc) in enumerate(URLS, 1):
        logger.info("    %2d. [%s] %s", i, title, desc)
    logger.info("  DB   : %s\n", CHROMA_DB_DIR)

    # Progress bar
    pbar = tqdm(total=len(URLS), desc="Ingesting", unit="page",
                bar_format="{l_bar}{bar:30}{r_bar}")

    def on_progress(i, total, title, status):
        status_label = "OK" if status == "ok" else "SKIP" if status == "empty" else "ERR"
        pbar.set_postfix_str(f"{status_label} {title}")
        pbar.update(1)

    vectorstore, stats = ingest_urls(URLS, progress_callback=on_progress)
    pbar.close()

    # Summary table
    elapsed = time.time() - start_time
    ok_count = sum(1 for s in stats if s["status"] == "ok")
    err_count = sum(1 for s in stats if s["status"].startswith("error"))
    empty_count = sum(1 for s in stats if s["status"] == "empty")
    total_chars = sum(s["chars"] for s in stats)
    total_chunks = sum(s["chunks"] for s in stats)

    logger.info("\n%s", "=" * 60)
    logger.info("  INGESTION SUMMARY")
    logger.info("%s", "=" * 60)
    logger.info("  Time elapsed : %.1fs", elapsed)
    logger.info("  URLs total   : %d", len(URLS))
    logger.info("  Succeeded    : %d", ok_count)
    if empty_count:
        logger.info("  Empty (skip) : %d", empty_count)
    if err_count:
        logger.warning("  Failed       : %d", err_count)
    logger.info("  Characters   : %s", f"{total_chars:,}")
    logger.info("  Chunks       : %d", total_chunks)
    logger.info("  DB location  : %s", CHROMA_DB_DIR)

    # Show per-URL details
    if err_count or empty_count:
        logger.info("\n  Details:")
        for s in stats:
            if s["status"] != "ok":
                logger.info("    [%5s] %s", s["status"].upper(), s["title"])

    stored = vectorstore._collection.count()
    logger.info("\n  %d chunks stored in ChromaDB.", stored)
    logger.info("  You can now run the chatbot:  streamlit run app.py")
    logger.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
