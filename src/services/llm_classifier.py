import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class ContractClassifier:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.categories = [
            "Non-Disclosure Agreements",
            "Service-Level Agreements",
            "Employment Contracts",
            "Vendor Agreements",
            "Partnership Agreements"
        ]

    def predict_contract_category(self, contract_text):
        """
        Predict the category of a contract text using OpenAI's chat model.

        Args:
            contract_text (str): The text content of the contract

        Returns:
            dict: {"category": ..., "reason": ...}
        """
        truncated_text = contract_text[:4000]

        prompt = f"""
        Analyze the following contract text and classify it into exactly one of these categories:
        - Non-Disclosure Agreements
        - Service-Level Agreements
        - Employment Contracts
        - Vendor Agreements
        - Partnership Agreements

        Return ONLY a valid JSON object with the following structure:
        {{
          "category": "exact category name from the list above",
          "reason": "brief explanation about 20 words for your classification choice"
        }}

        Contract Text:
        {truncated_text}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal contract classification expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1
            )

            raw_response = response.choices[0].message.content.strip()

            # Try to extract JSON (in case the model wraps it in text)
            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError:
                # Attempt to extract JSON substring if there's extra text
                import re
                match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                else:
                    return {"category": "Uncertain", "reason": f"Invalid JSON: {raw_response}"}

            # Validate category
            if parsed["category"] not in self.categories:
                return {
                    "category": "Uncertain",
                    "reason": f"Model returned unknown category: {parsed['category']}"
                }

            return parsed

        except Exception as e:
            return {"category": "Error", "reason": str(e)}


# Example usage
if __name__ == "__main__":
    classifier = ContractClassifier()

    sample_contract = """
    CONFIDENTIALITY AGREEMENT
    This Agreement is made between Company ABC and Contractor XYZ.
    The parties agree to maintain the confidentiality of all proprietary information...
    Neither party shall disclose any confidential information to third parties...
    """

    result = classifier.predict_contract_category(sample_contract)
    print(result)
