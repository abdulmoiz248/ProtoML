import requests
import json
from typing import Dict
from datetime import datetime
import config


class DiscordNotifier:
    """Send notifications to Discord"""
    
    def __init__(self):
        self.webhook_url = config.DISCORD_WEBHOOK_URL
        self.username = config.DISCORD_USERNAME
        self.enabled = config.DISCORD_ENABLED and bool(self.webhook_url)
    
    def send_report(self, report: Dict) -> bool:
     
        if not self.enabled:
            print("⚠️  Discord notifications disabled or webhook URL not set")
            return False
        
        print("\n📤 Sending report to Discord...")
        
        try:
            # Send header embed with paper info
            self._send_header_embed(report)
            
            # Send each section as a separate message
            self._send_section_embed(report, "🎯 Problem", report['Problem'])
            self._send_section_embed(report, "📊 Dataset", report['Dataset'])
            self._send_section_embed(report, "🤖 Model & Methodology", report['Model'])
            self._send_section_embed(report, "🔑 Key Techniques", self._format_techniques(report['KeyTechniques']))
            self._send_section_embed(report, "💡 Why It Matters", report['WhyItMatters'])
            self._send_section_embed(report, "🚀 Mini-Project Idea", report['MiniImplementationIdea'])
            
            print("✅ Report sent to Discord successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error sending to Discord: {str(e)}")
            return False
    
    def _send_header_embed(self, report: Dict) -> None:
        """Send the header embed with paper information"""
        # Color based on category
        color_map = {
            "cs.CV": 0x3498db,  # Blue for Computer Vision
            "cs.CL": 0x2ecc71,  # Green for NLP
            "cs.LG": 0x9b59b6,  # Purple for ML
            "q-bio.QM": 0xe74c3c,  # Red for Healthcare
        }
        color = color_map.get(report.get('category', ''), 0x95a5a6)
        
        embed = {
            "title": f"📄 {report['paper_title'][:200]}",
            "url": report['pdf_url'],
            "description": f"**Authors:** {', '.join(report['paper_authors'][:3])}{'...' if len(report['paper_authors']) > 3 else ''}\n"
                          f"**Published:** {report['published']} | **Category:** {report['category']}\n"
                          f"**arXiv ID:** [{report['arxiv_id']}](https://arxiv.org/abs/{report['arxiv_id']})",
            "color": color,
            "footer": {
                "text": f"ProtoML • {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            }
        }
        
        payload = {
            "username": self.username,
            "embeds": [embed]
        }
        
        response = requests.post(
            self.webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
    
    def _send_section_embed(self, report: Dict, title: str, content: str) -> None:
        """Send a section as a separate embed"""
        # Truncate if too long
        if len(content) > 4000:
            content = content[:3997] + "..."
        
        # Color based on category
        color_map = {
            "cs.CV": 0x3498db,
            "cs.CL": 0x2ecc71,
            "cs.LG": 0x9b59b6,
            "q-bio.QM": 0xe74c3c,
        }
        color = color_map.get(report.get('category', ''), 0x95a5a6)
        
        embed = {
            "title": title,
            "description": content,
            "color": color
        }
        
        payload = {
            "username": self.username,
            "embeds": [embed]
        }
        
        response = requests.post(
            self.webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
    
    def _format_techniques(self, techniques) -> str:
        """Format key techniques for Discord"""
        if isinstance(techniques, list):
            formatted = "\n".join([f"• {t}" for t in techniques[:5]])
            if len(formatted) > 1024:
                formatted = formatted[:1020] + "..."
            return formatted
        else:
            return str(techniques)[:1024]
    
    def send_simple_message(self, message: str) -> bool:
       
        if not self.enabled:
            return False
        
        try:
            payload = {
                "username": self.username,
                "content": message
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            print(f"❌ Error sending message: {str(e)}")
            return False
