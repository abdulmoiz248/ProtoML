"""
Report Generator
Gemini generates structured report from paper analysis
"""

import json
from typing import Dict, List
from google import genai
from groq import Groq
import config


class ReportGenerator:
    """Generate structured reports using Gemini"""
    
    def __init__(self):
        # Initialize Gemini with new API
        self.gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.gemini_model = config.GEMINI_MODEL
        
        # Initialize Groq for key findings
        self.groq_client = Groq(api_key=config.GROQ_API_KEY)
        self.groq_model = config.GROQ_MODEL
    
    def generate_key_findings(
        self,
        paper: Dict,
        chunks: List[Dict],
        embeddings
    ) -> str:
        """
        Groq analyzes embeddings to generate key findings summary
        
        Args:
            paper: Paper dictionary
            chunks: Text chunks from PDF
            embeddings: Embeddings array
            
        Returns:
            Key findings summary string
        """
        print("\n🤖 Groq analyzing paper for key findings...")
        
        # Get most important sections (introduction, methods, results, conclusion)
        important_sections = []
        
        # Sample chunks evenly distributed through the paper
        if len(chunks) > 0:
            step = max(1, len(chunks) // 10)  # Get ~10 representative chunks
            for i in range(0, len(chunks), step):
                important_sections.append(chunks[i]['text'])
        
        # Combine sections
        combined_text = "\n\n".join(important_sections[:10])  # Limit to avoid token overflow
        
        prompt = f"""You are Agent A analyzing a research paper. Extract key findings for another agent (Gemini).

Paper: {paper['title']}
Authors: {', '.join(paper['authors'][:5])}
Abstract: {paper['abstract']}

Selected Paper Sections:
{combined_text[:8000]}

Provide a comprehensive summary focusing on:
1. Core Problem & Motivation
2. Key Methodology/Approach
3. Main Results & Findings
4. Datasets Used
5. Model Architecture Details
6. Practical Applications

Be detailed and technical. This will help Gemini create a structured report."""
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "You are an expert research analyst. Provide detailed technical summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            key_findings = response.choices[0].message.content.strip()
            print(f"✅ Key findings generated ({len(key_findings)} chars)")
            return key_findings
            
        except Exception as e:
            print(f"❌ Error generating key findings: {str(e)}")
            # Fallback to abstract
            return f"Based on abstract: {paper['abstract']}"
    
    def generate_structured_report(
        self,
        paper: Dict,
        key_findings: str
    ) -> Dict:
        """
        Gemini generates structured report from key findings
        
        Args:
            paper: Paper dictionary
            key_findings: Summary from Groq
            
        Returns:
            Structured report dictionary
        """
        print("\n🔮 Gemini generating structured report...")
        
        prompt = f"""You are Agent B. Create a comprehensive structured report for this research paper.

Paper Details:
Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Published: {paper['published']}
Category: {paper['primary_category']}
arXiv ID: {paper['arxiv_id']}

Abstract:
{paper['abstract']}

Key Findings from Groq Analysis:
{key_findings}

Generate a detailed structured report in JSON format:
{{
    "Problem": "Clear description of the problem being addressed",
    "Dataset": "Detailed description of dataset(s) used, including size, characteristics, and source",
    "Model": "Comprehensive description of model architecture, methodology, and key innovations",
    "WhyItMatters": "Real-world impact, significance, and potential applications",
    "MiniImplementationIdea": "Concrete mini-project idea for portfolio (be specific and actionable)",
    "KeyTechniques": "List of 3-5 key techniques or methods used",
    "Results": "Main results and performance metrics",
    "Limitations": "Acknowledged limitations or areas for improvement",
    "AdditionalNotes": "Any other relevant insights, related work, or interesting observations"
}}

Be comprehensive, technical, and specific. This is for a portfolio project."""
        
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config={
                    "temperature": 0.3,
                    "max_output_tokens": 8000,
                    "response_mime_type": "application/json"
                }
            )
            
            content = response.text
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            report = json.loads(content)
            
            # Add metadata
            report['paper_title'] = paper['title']
            report['paper_authors'] = paper['authors']
            report['arxiv_id'] = paper['arxiv_id']
            report['pdf_url'] = paper['pdf_url']
            report['published'] = paper['published']
            report['category'] = paper['primary_category']
            
            # Add selection info if available
            if 'selection_method' in paper:
                report['selection_method'] = paper['selection_method']
            if 'final_decision' in paper:
                report['final_decision_reasoning'] = paper['final_decision']['reasoning']
            
            print("✅ Structured report generated")
            return report
            
        except Exception as e:
            print(f"❌ Error generating report: {str(e)}")
            # Fallback basic report
            return {
                "Problem": "Error generating detailed report",
                "Dataset": paper.get('abstract', 'N/A')[:200],
                "Model": "See paper for details",
                "WhyItMatters": "Requires manual review",
                "MiniImplementationIdea": "Review paper and determine suitable implementation",
                "KeyTechniques": ["See paper"],
                "Results": "See paper",
                "Limitations": "Report generation error",
                "AdditionalNotes": f"Error: {str(e)}",
                "paper_title": paper['title'],
                "paper_authors": paper['authors'],
                "arxiv_id": paper['arxiv_id'],
                "pdf_url": paper['pdf_url'],
                "published": paper['published'],
                "category": paper['primary_category']
            }
    
    def format_report_for_display(self, report: Dict) -> str:
        """
        Format structured report for human-readable display
        
        Args:
            report: Report dictionary
            
        Returns:
            Formatted string
        """
        formatted = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       📄 PROTOML DAILY PAPER SUMMARY                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

📌 PAPER INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title:      {report['paper_title']}
Authors:    {', '.join(report['paper_authors'][:5])}{'...' if len(report['paper_authors']) > 5 else ''}
Published:  {report['published']}
Category:   {report['category']}
arXiv ID:   {report['arxiv_id']}
PDF:        {report['pdf_url']}

🎯 PROBLEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{report['Problem']}

📊 DATASET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{report['Dataset']}

🤖 MODEL & METHODOLOGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{report['Model']}

🔑 KEY TECHNIQUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        if isinstance(report['KeyTechniques'], list):
            for i, technique in enumerate(report['KeyTechniques'], 1):
                formatted += f"{i}. {technique}\n"
        else:
            formatted += f"{report['KeyTechniques']}\n"
        
        formatted += f"""
📈 RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{report['Results']}

💡 WHY IT MATTERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{report['WhyItMatters']}

🚀 MINI-PROJECT IDEA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{report['MiniImplementationIdea']}

⚠️  LIMITATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{report['Limitations']}

📝 ADDITIONAL NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{report['AdditionalNotes']}
"""
        
        if 'selection_method' in report:
            formatted += f"\n🏆 Selection Method: {report['selection_method']}\n"
        if 'final_decision_reasoning' in report:
            formatted += f"   Decision Reasoning: {report['final_decision_reasoning']}\n"
        
        formatted += "\n" + "="*80 + "\n"
        
        return formatted
