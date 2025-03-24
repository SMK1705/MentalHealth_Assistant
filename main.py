from logging_config import setup_logging
setup_logging()

import logging
from unified_guidance import generate_counselor_guidance

logger = logging.getLogger(__name__)

def unified_chat_mode():
    logger.info("Starting Unified Guidance Chat Mode (multi-turn conversation)")
    print("Unified Guidance Chat Mode (multi-turn conversation):")
    print("Type 'exit' to end the session.\n")
    conversation_history = ""
    while True:
        user_input = input("Counselor: ").strip()
        if user_input.lower() == 'exit':
            logger.info("Exiting chat mode")
            break
        
        guidance = generate_counselor_guidance(user_input, conversation_history)
        
        print("\n--- Generated Advice ---")
        print(guidance.get("generated_advice", "No advice generated."))
        
        print("\n--- Predicted Topic ---")
        topic = guidance.get("predicted_topic", "N/A")
        score = guidance.get("topic_confidence", "N/A")
        print(f"{topic} (Confidence: {score})")
        
        print("\n--- Historical Examples ---")
        examples = guidance.get("historical_examples", [])
        if examples:
            for idx, ex in enumerate(examples, 1):
                print(f"\nExample {idx}:")
                print("Question:", ex.get("questionText", ""))
                print("Answer:", ex.get("answerText", ""))
                print("Topic:", ex.get("topic", ""))
                print("Upvotes:", ex.get("upvotes", ""))
                print("-----")
        else:
            print("No historical examples found.")
        
        conversation_history += "\nCounselor: " + user_input + "\nAdvice: " + guidance.get("generated_advice", "") + "\n"
        print("\n-------------------------------------------\n")
        
def main():
    logger.info("Mental Health Counselor Guidance System started")
    print("Mental Health Counselor Guidance System\n")
    unified_chat_mode()

if __name__ == "__main__":
    main()
