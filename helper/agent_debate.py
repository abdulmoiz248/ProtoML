import json
from typing import List, Dict, Optional, Tuple
from groq import Groq
from google import genai
from google.genai import types
import config


class AgentDebate:
   
    
    def __init__(self):
        # Initialize Groq
        self.groq_client = Groq(api_key=config.GROQ_API_KEY)
        self.groq_model = config.GROQ_MODEL
        
        # Initialize Gemini with new API
        self.gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.gemini_model = config.GEMINI_MODEL
        
        # Store conversation history for context
        self.conversation_context = []
    
    def _add_to_context(self, role: str, content: str, round_num: int) -> None:
        """Add message to conversation context for multi-turn debate"""
        self.conversation_context.append({
            "role": role,
            "content": content,
            "round": round_num
        })
    
    def _get_context_summary(self) -> str:
        """Generate summary of conversation context for agents"""
        if not self.conversation_context:
            return ""
        
        summary = "\\n\\n<conversation_history>\\n"
        for msg in self.conversation_context:
            summary += f"Round {msg['round']} - {msg['role']}:\\n{msg['content']}\\n\\n"
        summary += "</conversation_history>"
        return summary
    
    def select_final_paper(
        self,
        groq_papers: List[Dict],
        gemini_papers: List[Dict]
    ) -> Dict:
       
        print("\n💬 Starting agent debate...")
        
        # Get top choices from each agent
        groq_top = groq_papers[0]
        gemini_top = gemini_papers[0]
        
        print(f"\n📌 Groq's top choice: {groq_top['title'][:60]}...")
        print(f"   Score: {groq_top['groq_total_score']:.2f}/10")
        print(f"   Reasoning: {groq_top['groq_scores']['reasoning'][:100]}...")
        
        print(f"\n📌 Gemini's top choice: {gemini_top['title'][:60]}...")
        print(f"   Score: {gemini_top['gemini_total_score']:.2f}/10")
        print(f"   Reasoning: {gemini_top['gemini_scores']['reasoning'][:100]}...")
        
        # If both agents chose the same paper, no debate needed
        if groq_top['arxiv_id'] == gemini_top['arxiv_id']:
            print("\n✅ Both agents agree! No debate needed.")
            selected_paper = groq_top.copy()
            selected_paper['selection_method'] = 'unanimous'
            selected_paper['debate_history'] = []
            return selected_paper
        
        # Clear conversation context for new debate
        self.conversation_context = []
        
        # Conduct debate
        debate_history = []
        
        # Round 1: Groq presents its case
        print("\n🤖 Groq presenting case...")
        groq_argument = self._groq_present_case(groq_top, gemini_top)
        self._add_to_context("Agent A (Groq)", groq_argument, 1)
        debate_history.append({
            "agent": "Groq",
            "round": 1,
            "argument": groq_argument
        })
        print(f"   Groq: {groq_argument[:150]}...")
        
        # Round 2: Gemini presents its case (with context)
        print("\n🔮 Gemini presenting case...")
        gemini_argument = self._gemini_present_case(gemini_top, groq_top, groq_argument)
        self._add_to_context("Agent B (Gemini)", gemini_argument, 2)
        debate_history.append({
            "agent": "Gemini",
            "round": 2,
            "argument": gemini_argument
        })
        print(f"   Gemini: {gemini_argument[:150]}...")
        
        # Round 3: Final decision (with full context)
        print("\n⚖️  Making final decision...")
        final_decision = self._make_final_decision(
            groq_top, gemini_top, groq_argument, gemini_argument
        )
        
        debate_history.append({
            "agent": "Consensus",
            "round": 3,
            "decision": final_decision
        })
        
        # Select the paper based on decision
        if final_decision['selected'] == 'groq':
            selected_paper = groq_top.copy()
            print(f"\n🏆 Final Selection: Groq's choice")
        else:
            selected_paper = gemini_top.copy()
            print(f"\n🏆 Final Selection: Gemini's choice")
        
        print(f"   Paper: {selected_paper['title']}")
        print(f"   Reason: {final_decision['reasoning'][:200]}...")
        
        selected_paper['selection_method'] = 'debate'
        selected_paper['debate_history'] = debate_history
        selected_paper['final_decision'] = final_decision
        
        return selected_paper
    
    def _groq_present_case(self, groq_choice: Dict, gemini_choice: Dict) -> str:
        """Groq presents its case for its top paper"""
        prompt = f"""You are Agent A. Present a compelling argument for why your selected paper is better.

Your Choice:
Title: {groq_choice['title']}
Your Score: {groq_choice['groq_total_score']:.2f}/10
Your Analysis: {groq_choice['groq_scores']['reasoning']}

Competing Choice (Gemini's):
Title: {gemini_choice['title']}
Gemini's Score: {gemini_choice['gemini_total_score']:.2f}/10

Present your argument in 2-3 sentences explaining why your paper is superior for the portfolio project goal."""
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": """<role>
You are Agent A (Groq), a rigorous AI research analyst specializing in evaluating machine learning papers for practical implementation. You are analytical, evidence-based, and persuasive.
</role>

<instructions>
1. **Analyze**: Examine the paper's technical merit, implementation feasibility, and learning value.
2. **Present**: Construct a compelling, evidence-based argument for your selection.
3. **Validate**: Ground your reasoning in concrete metrics and paper details.
</instructions>

<constraints>
- Be precise and direct
- Use concrete evidence from scores and analysis
- Present argument in 2-3 sentences
- Focus on implementation feasibility and portfolio value
</constraints>"""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"I believe my paper is superior due to {groq_choice['groq_scores']['reasoning']}"
    
    def _gemini_present_case(
        self,
        gemini_choice: Dict,
        groq_choice: Dict,
        groq_argument: str
    ) -> str:
        """Gemini presents its case, responding to Groq's argument"""
        # Import required types
        from google.genai import types
        
        system_instruction = """<role>
You are Agent B (Gemini), a research evaluation specialist with expertise in assessing papers for educational and practical value. You are thoughtful, analytical, and respond strategically to competing arguments.
</role>

<instructions>
1. **Review**: Carefully analyze Agent A's argument and identify its strengths and weaknesses.
2. **Counter**: Present evidence-based counter-arguments that highlight your paper's advantages.
3. **Synthesize**: Demonstrate why your selection provides superior value for the stated goals.
</instructions>

<constraints>
- Be objective and evidence-based
- Address Agent A's points directly
- Present counter-argument in 2-3 sentences
- Focus on learning value and practical impact
</constraints>

<context>
You are engaged in a structured debate with Agent A to select the best research paper for a portfolio project.
</context>"""
        
        prompt = f"""<opponent_argument>
{groq_argument}
</opponent_argument>

<opponent_choice>
Title: {groq_choice['title']}
</opponent_choice>

<your_choice>
Title: {gemini_choice['title']}
Your Score: {gemini_choice['gemini_total_score']:.2f}/10
Your Analysis: {gemini_choice['gemini_scores']['reasoning']}
</your_choice>

<task>
Present your counter-argument in 2-3 sentences explaining why your paper selection is superior for the portfolio project goals.
</task>"""
        
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=1.0,  # Keep default temperature for Gemini 3
                    max_output_tokens=1000
                )
            )
            return response.text.strip()
        except Exception as e:
            return f"My paper is superior because {gemini_choice['gemini_scores']['reasoning']}"
    
    def _make_final_decision(
        self,
        groq_choice: Dict,
        gemini_choice: Dict,
        groq_argument: str,
        gemini_argument: str
    ) -> Dict:
        """Make final decision based on debate"""
        from google.genai import types
        
        system_instruction = """<role>
You are an impartial arbiter specializing in research evaluation. You make evidence-based decisions by weighing arguments and selecting the optimal choice for stated objectives.
</role>

<instructions>
1. **Plan**: Analyze both arguments against evaluation criteria.
2. **Weigh**: Compare strengths and weaknesses objectively.
3. **Decide**: Select the paper that best serves the portfolio project goal.
4. **Validate**: Review your decision against all criteria before finalizing.
</instructions>

<evaluation_criteria>
- Implementation feasibility (Can it be built?)
- Learning value (Educational benefit)
- Practical impact (Real-world applicability)
- Novelty and interest (Uniqueness)
</evaluation_criteria>

<output_format>
Respond in valid JSON format only:
{
    "selected": "groq" or "gemini",
    "reasoning": "explanation in 2-3 sentences",
    "confidence": "high/medium/low"
}
</output_format>

<constraints>
- Be completely objective
- Base decision on evidence, not preference
- Output only valid JSON
</constraints>"""
        
        prompt = f"""<debate_context>
Two agents have presented arguments for their selected research papers for a portfolio project.
</debate_context>

<paper_a>
Agent: Groq (Agent A)
Title: {groq_choice['title']}
Quantitative Score: {groq_choice['groq_total_score']:.2f}/10
Argument: {groq_argument}
Abstract: {groq_choice['abstract'][:300]}...
</paper_a>

<paper_b>
Agent: Gemini (Agent B)
Title: {gemini_choice['title']}
Quantitative Score: {gemini_choice['gemini_total_score']:.2f}/10
Argument: {gemini_argument}
Abstract: {gemini_choice['abstract'][:300]}...
</paper_b>

<task>
Based on the evaluation criteria, arguments, and paper details, decide which paper is better for a portfolio project. Provide your decision in the specified JSON format.
</task>"""
        
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=1.0,  # Keep default for Gemini 3
                    max_output_tokens=500
                )
            )
            
            content = response.text
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            decision = json.loads(content)
            return decision
            
        except Exception as e:
            # Default to higher score if decision fails
            if groq_choice['groq_total_score'] >= gemini_choice['gemini_total_score']:
                return {
                    "selected": "groq",
                    "reasoning": "Selected based on higher score after debate error.",
                    "confidence": "medium"
                }
            else:
                return {
                    "selected": "gemini",
                    "reasoning": "Selected based on higher score after debate error.",
                    "confidence": "medium"
                }
