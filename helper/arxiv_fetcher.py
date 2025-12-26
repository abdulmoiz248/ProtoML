import arxiv
from typing import List, Dict
from datetime import datetime, timedelta, timezone
import config


class ArxivFetcher:
    def __init__(self):
        self.categories = config.ARXIV_CATEGORIES
        self.max_papers_per_category = config.MAX_PAPERS_PER_CATEGORY
        self.total_papers = config.TOTAL_PAPERS_TO_FETCH
        self.days_limit = 7

    def fetch_papers(self) -> List[Dict]:
        print("📚 Fetching papers from arXiv (last 7 days only)...")
        all_papers = []
        # Make cutoff_date timezone-aware to match arxiv result.published
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.days_limit)

        for category in self.categories:
            print(f"  → Fetching from category: {category}")

            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=self.max_papers_per_category,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            for result in search.results():
                if result.published < cutoff_date:
                    break

                paper = {
                    "title": result.title,
                    "abstract": result.summary.replace("\n", " "),
                    "authors": [author.name for author in result.authors],
                    "pdf_url": result.pdf_url,
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "published": result.published.strftime("%Y-%m-%d"),
                    "category": category,
                    "primary_category": result.primary_category
                }

                all_papers.append(paper)

                if len(all_papers) >= self.total_papers:
                    break

            if len(all_papers) >= self.total_papers:
                break

        papers = all_papers[:self.total_papers]
        print(f"✅ Fetched {len(papers)} papers from last week")
        return papers

    def split_papers(self, papers: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        mid = len(papers) // 2
        groq_papers = papers[:mid]
        gemini_papers = papers[mid:]
        print(f"📊 Split papers: {len(groq_papers)} for Groq, {len(gemini_papers)} for Gemini")
        return groq_papers, gemini_papers


