import re
import logging

logger = logging.getLogger(__name__)

class SafetyChecker:
    RED_FLAGS = {
        r"\b(suicide|kill myself|end it all)\b": "suicide_risk",
        r"\b(abuse|molest|rape)\b": "abuse_disclosure",
        r"\b(harm|hurt) (myself|others)\b": "violence_risk"
    }
    
    PROTOCOLS = {
        "suicide_risk": {
            "response": "I'm really sorry you're feeling this way. "
                        "Please contact the National Suicide Prevention Lifeline at 1-800-273-8255.",
            "action": "CRITICAL"
        },
        "abuse_disclosure": {
            "response": "Thank you for sharing this. Let's connect you with a specialist who can help.",
            "action": "URGENT"
        }
    }
    
    def check_input(self, text: str):
        text = text.lower()
        for pattern, flag_type in self.RED_FLAGS.items():
            if re.search(pattern, text):
                logger.info("Safety check triggered: %s", flag_type)
                return self.PROTOCOLS.get(flag_type)
        return None
