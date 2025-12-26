import sys
import json
import os
from datetime import datetime

# Import helper modules
from helper.arxiv_fetcher import ArxivFetcher
from helper.paper_scorer import PaperScorer
from helper.agent_debate import AgentDebate
from helper.pdf_processor import PDFProcessor
from helper.report_generator import ReportGenerator
from helper.discord_notifier import DiscordNotifier

import config


class ProtoML:
    """Main orchestrator for the ProtoML pipeline"""
    
    def __init__(self):
        print("\n" + "="*80)
        print("🚀 ProtoML - Automated ML Research Analysis")
        print("="*80 + "\n")
        
        # Validate API keys
        self._validate_config()
        
        # Initialize components
        self.fetcher = ArxivFetcher()
        self.scorer = PaperScorer()
        self.debate = AgentDebate()
        self.pdf_processor = PDFProcessor()
        self.report_gen = ReportGenerator()
        self.notifier = DiscordNotifier()
    
    def _validate_config(self):
        """Validate that required API keys are set"""
        missing = []
        
        if not config.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not config.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        
        if missing:
            print("❌ ERROR: Missing required API keys:")
            for key in missing:
                print(f"   - {key}")
            print("\n💡 Please set these in your .env file (see .env.example)")
            sys.exit(1)
        
        print("✅ Configuration validated")
    
    def run(self):
        """Execute the complete ProtoML pipeline"""
        try:
            # Step 1: Fetch papers from arXiv
            print("\n" + "="*80)
            print("STEP 1: FETCH PAPERS FROM arXiv")
            print("="*80)
            
            papers = self.fetcher.fetch_papers()
            
            if not papers:
                print("❌ No papers fetched. Exiting.")
                return
            
            # Step 2: Split papers between agents
            print("\n" + "="*80)
            print("STEP 2: SPLIT PAPERS BETWEEN AGENTS")
            print("="*80)
            
            groq_papers, gemini_papers = self.fetcher.split_papers(papers)
            
            # Step 3: Score papers
            print("\n" + "="*80)
            print("STEP 3: AGENTS SCORE PAPERS")
            print("="*80)
            
            groq_scored = self.scorer.score_with_groq(groq_papers)
            gemini_scored = self.scorer.score_with_gemini(gemini_papers)
            
            # Display top choices
            print(f"\n🏆 Top Scores:")
            print(f"   Groq:   {groq_scored[0]['title'][:60]}... ({groq_scored[0]['groq_total_score']:.2f}/10)")
            print(f"   Gemini: {gemini_scored[0]['title'][:60]}... ({gemini_scored[0]['gemini_total_score']:.2f}/10)")
            
            # Step 4: Agent debate and selection
            print("\n" + "="*80)
            print("STEP 4: AGENT DEBATE & PAPER SELECTION")
            print("="*80)
            
            selected_paper = self.debate.select_final_paper(groq_scored, gemini_scored)
            
            print(f"\n✅ Final Paper Selected:")
            print(f"   {selected_paper['title']}")
            print(f"   arXiv ID: {selected_paper['arxiv_id']}")
            
            # Step 5: Process PDF and generate embeddings
            print("\n" + "="*80)
            print("STEP 5: PDF PROCESSING & EMBEDDINGS")
            print("="*80)
            
            chunks, embeddings = self.pdf_processor.process_paper(selected_paper)
            
            if len(chunks) == 0:
                print("⚠️  PDF processing failed, continuing with abstract only...")
                chunks = [{"text": selected_paper['abstract'], "page": 1, "chunk_id": 0}]
                embeddings = None
            
            # Step 6: Groq generates key findings
            print("\n" + "="*80)
            print("STEP 6: GROQ ANALYZES KEY FINDINGS")
            print("="*80)
            
            key_findings = self.report_gen.generate_key_findings(
                selected_paper, chunks, embeddings
            )
            
            # Step 7: Gemini generates structured report
            print("\n" + "="*80)
            print("STEP 7: GEMINI GENERATES STRUCTURED REPORT")
            print("="*80)
            
            report = self.report_gen.generate_structured_report(
                selected_paper, key_findings
            )
            
            # Display formatted report
            print("\n" + "="*80)
            print("FINAL REPORT")
            print("="*80)
            
            formatted_report = self.report_gen.format_report_for_display(report)
            print(formatted_report)
            
            # Step 8: Send to Discord
            print("\n" + "="*80)
            print("STEP 8: SEND TO DISCORD")
            print("="*80)
            
            if config.DISCORD_ENABLED and config.DISCORD_WEBHOOK_URL:
                success = self.notifier.send_report(report)
                if success:
                    print("✅ Report sent to Discord!")
                else:
                    print("⚠️  Failed to send to Discord (see errors above)")
            else:
                print("ℹ️  Discord notifications disabled")
            
            # Pipeline complete
            print("\n" + "="*80)
            print("✅ PROTOML PIPELINE COMPLETE!")
            print("="*80)
            print(f"\n📊 Summary:")
            print(f"   Papers analyzed: {len(papers)}")
            print(f"   Selected paper: {selected_paper['title']}")
            print(f"   Report sent to Discord: {'Yes' if config.DISCORD_ENABLED else 'No'}")
            
            return report
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"\n\n❌ Pipeline error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point"""
    protoml = ProtoML()
    protoml.run()


if __name__ == "__main__":
    main()
