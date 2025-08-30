import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from typing import Dict, List


class FewShotContractClassifier:
    """Few-shot + keyword hybrid classifier for contract types."""

    def __init__(self):
        self.examples: Dict[str, List[str]] = {
            "Non-Disclosure Agreements": [
                "confidentiality agreement between parties to protect proprietary information",
                "non-disclosure agreement prohibiting disclosure of trade secrets",
                "NDA protecting confidential business information during discussions",
            ],
            "Service-Level Agreements": [
                "service level agreement defining uptime guarantees and performance metrics",
                "SLA outlining response times and availability commitments",
                "service agreement with performance standards and remedies",
            ],
            "Employment Contracts": [
                "employment agreement specifying salary benefits and job responsibilities",
                "contract of employment with termination conditions and compensation",
                "employee agreement outlining terms of service and benefits",
            ],
            "Vendor Agreements": [
                "vendor contract for supply of goods and services with payment terms",
                "supplier agreement outlining delivery schedules and quality standards",
                "purchasing agreement with vendor performance requirements",
            ],
            "Partnership Agreements": [
                "partnership agreement establishing business collaboration terms",
                "joint venture agreement with profit sharing and management structure",
                "partnership contract outlining roles responsibilities and contributions",
            ],
        }

        self.keywords: Dict[str, List[str]] = {
            "Non-Disclosure Agreements": [
                "confidential", "nda", "non-disclosure", "proprietary", "trade secret", "disclosure",
            ],
            "Service-Level Agreements": [
                "sla", "service level", "uptime", "response time", "performance", "availability", "guarantee",
            ],
            "Employment Contracts": [
                "employment", "salary", "benefits", "termination", "job", "wages", "employee", "employer",
            ],
            "Vendor Agreements": [
                "vendor", "supplier", "purchase", "delivery", "goods", "services", "procurement", "supply",
            ],
            "Partnership Agreements": [
                "partnership", "joint venture", "profit sharing", "collaboration", "partner", "venture", "joint",
            ],
        }

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        self._train_vectorizer()

    def _train_vectorizer(self) -> None:
        """Fit vectorizer on all examples."""
        all_texts = [txt for ex_list in self.examples.values() for txt in ex_list]
        self.vectorizer.fit(all_texts)

    def _get_similarity_score(self, text: str, contract_type: str) -> float:
        """Cosine similarity between text and few-shot examples."""
        text_vec = self.vectorizer.transform([text])
        example_vecs = self.vectorizer.transform(self.examples[contract_type])
        similarities = cosine_similarity(text_vec, example_vecs)
        return float(np.mean(similarities))

    def _keyword_score(self, text: str, contract_type: str) -> float:
        """Keyword coverage score."""
        text_lower = text.lower()
        keywords = self.keywords[contract_type]
        matches = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", text_lower))
        return matches / len(keywords) if keywords else 0.0

    def classify_contract(self, text: str, confidence_threshold: float = 0.3) -> Dict:
        """Classify text into contract type."""
        if len(text.strip()) < 10:
            return {
                "predicted_class": None,
                "confidence": 0.0,
                "is_confident": False,
                "all_scores": {},
                "method": "invalid-input",
                "error": "Text too short",
            }

        try:
            scores: Dict[str, float] = {}
            for ctype in self.examples:
                sim = self._get_similarity_score(text, ctype)
                kw = self._keyword_score(text, ctype)
                scores[ctype] = (0.7 * sim) + (0.3 * kw)

            best_type, best_score = max(scores.items(), key=lambda x: x[1])
            return {
                "predicted_class": best_type,
                "confidence": round(best_score, 3),
                "is_confident": best_score >= confidence_threshold,
                "all_scores": {k: round(v, 3) for k, v in scores.items()},
                "method": "few-shot" if best_score >= 0.2 else "keyword-fallback",
            }

        except Exception:
            return self._keyword_fallback(text)

    def _keyword_fallback(self, text: str) -> Dict:
        """Pure keyword-based fallback classification."""
        text_lower = text.lower()
        scores = {
            ctype: sum(1 for kw in kws if kw in text_lower) / len(kws)
            for ctype, kws in self.keywords.items()
        }
        best_type, best_score = max(scores.items(), key=lambda x: x[1])
        return {
            "predicted_class": best_type,
            "confidence": round(best_score, 3),
            "is_confident": best_score >= 0.3,
            "all_scores": {k: round(v, 3) for k, v in scores.items()},
            "method": "keyword-fallback",
        }

    def add_example(self, contract_type: str, example_text: str) -> None:
        """Add new training example and retrain vectorizer."""
        if contract_type in self.examples:
            self.examples[contract_type].append(example_text)
            self._train_vectorizer()


# Global instance
classifier = FewShotContractClassifier()

def classify_text(text: str, confidence_threshold: float = 0.3) -> Dict:
    return classifier.classify_contract(text, confidence_threshold)

def add_training_example(contract_type: str, example_text: str) -> None:
    classifier.add_example(contract_type, example_text)


# Example usage and testing
if __name__ == "__main__":
    print("Few-Shot Contract Classifier")
    print("=" * 50)

    test_cases = [
        '''vendor agreement 1 vendor agreement vendor agreement for products and or services for willow tree pruning this agreement is made and enter ed into as of the  day of , 20 by and between the city of foster city hereinafter called city and  hereinafter called vendor. recit als this agreement is enter ed into with reference to the following facts and circumstances: a. that city desir es to engage vendor to provide a product andor services to the city ; b. that vendor is qualified to provide the product andor services to the city and; c. that the city has elected to engage vendor upon the terms and conditions as hereinafter set forth. 1. a. services. the services to be performed by vendor under this agreement are set forth in exhibit a, which is, by this reference, incorporated herein and made a part hereof as though it were fully set forth herein. performance of the work specified in said exhibit a is hereby made an obligation of vendor under this agreement, subject to any changes that may be made subsequently hereto upon the mutual written agreement of the said parties. where in conflict, the terms of this agreement supersede and prevail over any terms set forth in exhibit a. b. product. the product to be supplied by vendor under this agreement is set forth in exhibit a which is, by this reference, incorporated herein and made a part hereof as though it were fully set forth herein. timely delivery of the product specified in said exhibit a is hereby made an obligation of vendor under this agreement, subject to any changes that may be made subsequently hereto upon the mutual written agreement of the said parties. 2. term; termination. (a) the term of this agreement shall commence upon the date hereinabove written and shall expir e upon the date enumerated in 2 updated 1292020 exhibit a, delivery of the product or completion of performance of services hereunder by vendor, whichever date shall first occur . (b) notwithstanding the provisions of (a) above, either party may terminate this agreement without cause by giving written notice not less than thirty (30) days prior to the effective date of termination, which date shall be included in said notice. city shall compensate vendor for any product deliver ed andor for services render ed, and reimburse vendor for costs and expenses incurr ed, to the date of termination, calculated in accordance with the provisions of paragraph 3. in ascertaining the services actually render ed to the date of termination, consideration shall be given both to completed work and work in process of completion. nothing herein contained shall be deemed a limitation upon the right of city to terminate this agreement for cause, or otherwise to exercise such rights or pursue such remedies as may accrue to city hereunder . 3. compensation; expenses; payment. city shall compensate vendor for all products supplied or services performed by vendor hereunder as shown in exhibit b attached hereto and by this reference incorporated herein. notwithstanding the foregoing, the combined total of compensation and reimbursement of costs payable hereunder shall not exceed the sum  (). invoices for amounts in excess of  () shall not be paid unless additional amounts have been appr oved in advance of supplying the product, performing the services or incurring the costs and expenses by city s city manager (for contracts less than 50,000) or city council (for contracts 50,000 or more) evidenced by motion duly made and carried and a written contract amendment having been executed. compensation and reimbursement of costs and expenses hereunder shall be payable upon vendor meeting contract milestones as defined in exhibit b. billing shall include an itemized statement, briefly describing by task and labor category or costexpense items billed. 4. additional services. in the event city desir es the delivery of additional products or performance of additional services not otherwise included within exhibit a, such products or services shall be authorized in advance by city s city manager (for contracts less than 50,000) or city council
        '''
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: {text}")
        result = classify_text(text)
        print(f"Predicted: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Method: {result['method']}")
        print(f"Confident: {result['is_confident']}")
        print("-" * 40)

    # Example of adding new training data
    print("\nAdding new training example...")
    add_training_example(
        "Non-Disclosure Agreements",
        "confidentiality clause protecting sensitive business information"
    )
    print("New example added successfully!")
