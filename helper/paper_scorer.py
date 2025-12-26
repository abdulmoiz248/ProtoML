import json
from typing import List, Dict
from groq import Groq
from google import genai
import config


class PaperScorer:
    
    def __init__(self):
        # Initialize Groq
        self.groq_client = Groq(api_key=config.GROQ_API_KEY)
        self.groq_model = config.GROQ_MODEL
        
        # Initialize Gemini with new API
        self.gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.gemini_model = config.GEMINI_MODEL
        
        self.scoring_weights = config.SCORING_WEIGHTS
    
    def _create_scoring_prompt(self, paper: Dict, agent_name: str) -> str:
        prompt = f"""You are {agent_name}, an expert AI research analyst. Score this research paper on the following criteria (scale 1-10):

1. Problem Relevance: How important is the problem being addressed?
2. Dataset Quality: Quality and scale of datasets used
3. Model Novelty: Innovation in the approach/architecture
4. Real-world Impact: Practical applicability and potential impact

Paper Details:
Title: {paper['title']}
Category: {paper['primary_category']}
Abstract:
{paper['abstract']}

Provide your analysis in JSON format:
{{
    "problem_relevance": <score 1-10>,
    "dataset_quality": <score 1-10>,
    "model_novelty": <score 1-10>,
    "real_world_impact": <score 1-10>,
    "reasoning": "<brief explanation of your scores>",
    "overall_impression": "<1-2 sentence summary>"
}}

Be critical and discerning in your evaluation."""
        
        return prompt
    
    def score_with_groq(self, papers: List[Dict]) -> List[Dict]:
       
        print(f"\n🤖 Groq scoring {len(papers)} papers...")
        scored_papers = []
        
        for i, paper in enumerate(papers, 1):
            print(f"  → Scoring paper {i}/{len(papers)}: {paper['title'][:60]}...")
            
            prompt = self._create_scoring_prompt(paper, "Agent A")
            
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": "You are an expert AI research analyst. Provide scores in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                # Parse response
                content = response.choices[0].message.content
                
                # Extract JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                scores = json.loads(content)
                
                # Calculate weighted score
                weighted_score = sum(
                    scores[criterion] * weight
                    for criterion, weight in self.scoring_weights.items()
                )
                
                paper_with_score = paper.copy()
                paper_with_score['groq_scores'] = scores
                paper_with_score['groq_total_score'] = weighted_score
                
                scored_papers.append(paper_with_score)
                
            except Exception as e:
                print(f"    ⚠️  Error scoring paper: {str(e)}")
                # Add default scores on error
                paper_with_score = paper.copy()
                paper_with_score['groq_scores'] = {
                    "problem_relevance": 5,
                    "dataset_quality": 5,
                    "model_novelty": 5,
                    "real_world_impact": 5,
                    "reasoning": f"Error during scoring: {str(e)}",
                    "overall_impression": "Unable to score"
                }
                paper_with_score['groq_total_score'] = 5.0
                scored_papers.append(paper_with_score)
        
        # Sort by score
        scored_papers.sort(key=lambda x: x['groq_total_score'], reverse=True)
        
        print(f"✅ Groq scoring complete")
        return scored_papers
    
    def score_with_gemini(self, papers: List[Dict]) -> List[Dict]:
        print(f"\n🔮 Gemini scoring {len(papers)} papers...")
        scored_papers = []
        
        
        response_schema = {
            "type": "object",
            "properties": {
                "problem_relevance": {"type": "integer"},
                "dataset_quality": {"type": "integer"},
                "model_novelty": {"type": "integer"},
                "real_world_impact": {"type": "integer"},
                "reasoning": {"type": "string"},
                "overall_impression": {"type": "string"}
            },
            "required": ["problem_relevance", "dataset_quality", "model_novelty", "real_world_impact", "reasoning", "overall_impression"]
        }
        
        for i, paper in enumerate(papers, 1):
            print(f"  → Scoring paper {i}/{len(papers)}: {paper['title'][:60]}...")
            
            prompt = self._create_scoring_prompt(paper, "Agent B")
            
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config={
                        "temperature": 0.3,
                        "max_output_tokens": 2048,
                        "response_mime_type": "application/json",
                        "response_schema": response_schema
                    }
                )
                

                content = response.text.strip()
               

                try:
                    scores = json.loads(content)
                except json.JSONDecodeError as json_err:
                    # If still failing, log the content for debugging
                    print(f"    ⚠️  JSON parse error. Content: {content[:200]}")
                    raise
                
               
                weighted_score = sum(
                    scores[criterion] * weight
                    for criterion, weight in self.scoring_weights.items()
                )
                
                paper_with_score = paper.copy()
                paper_with_score['gemini_scores'] = scores
                paper_with_score['gemini_total_score'] = weighted_score
                
                scored_papers.append(paper_with_score)
                
            except Exception as e:
                print(f"    ⚠️  Error scoring paper: {str(e)}")
             
                paper_with_score = paper.copy()
                paper_with_score['gemini_scores'] = {
                    "problem_relevance": 5,
                    "dataset_quality": 5,
                    "model_novelty": 5,
                    "real_world_impact": 5,
                    "reasoning": f"Error during scoring: {str(e)}",
                    "overall_impression": "Unable to score"
                }
                paper_with_score['gemini_total_score'] = 5.0
                scored_papers.append(paper_with_score)
        
        scored_papers.sort(key=lambda x: x['gemini_total_score'], reverse=True)
        
        print(f"✅ Gemini scoring complete")
        return scored_papers
