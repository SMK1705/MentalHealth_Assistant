import re
import logging

logger = logging.getLogger(__name__)

class SafetyChecker:
    RED_FLAGS = {
        r"\b(suicide|suicidal|kill myself|killing myself|end it all|end my life|"
        r"ending my life|take my life|take my own life|want to die|wanting to die|"
        r"better off dead|don'?t want to live|don'?t want to be alive|"
        r"no reason to live|no point in living)\b": "suicide_risk",
        r"\b(abuse|abused|abusing|abuser|molest|molested|molesting|molester|"
        r"rape|raped|raping|rapist|sexual assault|sexually assaulted)\b": "abuse_disclosure",
        r"\b((harm|hurt|kill|cut|cutting) "
        r"(myself|himself|herself|themselves|others|someone|him|her|them)|"
        r"self[- ]?harm)\b": "violence_risk",
    }
    
    PROTOCOLS = {
        "suicide_risk": {
            "response": "I'm really sorry you're feeling this way. "
                        "Please contact the 988 Suicide & Crisis Lifeline by calling or texting 988.",
            "action": "CRITICAL"
        },
        "abuse_disclosure": {
            "response": "Thank you for sharing this. Let's connect you with a specialist who can help.",
            "action": "URGENT"
        },
        "violence_risk": {
            "response": "Safety is the immediate priority. If anyone is in danger right now, "
                        "contact emergency services (911). You can also reach the 988 Suicide & "
                        "Crisis Lifeline by calling or texting 988.",
            "action": "CRITICAL"
        }
    }
    
    def check_input(self, text: str):
        text = text.lower()
        for pattern, flag_type in self.RED_FLAGS.items():
            if re.search(pattern, text):
                logger.info("Safety check triggered: %s", flag_type)
                protocol = self.PROTOCOLS.get(flag_type)
                if protocol:
                    return {**protocol, "flag_type": flag_type}
                return None
        return None
