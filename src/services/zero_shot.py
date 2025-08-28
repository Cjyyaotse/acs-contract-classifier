# contract_classifier_service.py
from transformers import pipeline
from typing import Dict, List, Optional
import re

class ContractClassifier:
    """
    A zero-shot classifier service for contract document types.
    Classifies documents into: Non-Disclosure Agreements, Service-Level Agreements, 
    Employment Contracts, Vendor Agreements, Partnership Agreements.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize the zero-shot classifier.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name
        )
        
        # Define the target categories with potential aliases/synonyms
        self.candidate_labels = [
            "Non-Disclosure Agreement",
            "Service-Level Agreement", 
            "Employment Contract",
            "Vendor Agreement",
            "Partnership Agreement"
        ]
        
        # Alternative labels to improve recognition
        self.alternative_labels = {
            "Non-Disclosure Agreement": ["NDA", "confidentiality agreement", "non-disclosure"],
            "Service-Level Agreement": ["SLA", "service agreement", "service level"],
            "Employment Contract": ["employment agreement", "job contract", "work agreement"],
            "Vendor Agreement": ["supplier agreement", "vendor contract", "supplier contract"],
            "Partnership Agreement": ["partnership contract", "business partnership", "joint venture"]
        }
    
    def preprocess_text(self, text: str, max_length: int = 1500) -> str:
        """
        Preprocess the contract text for classification.
        
        Args:
            text: Raw contract text
            max_length: Maximum characters to keep (due to model limits)
            
        Returns:
            Preprocessed text
        """
        # Clean and truncate text
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate to model's effective context window
        if len(text) > max_length:
            # Keep the beginning (often contains type indicators)
            text = text[:max_length] + "..."
        
        return text
    
    def classify_contract(self, contract_text: str, 
                         confidence_threshold: float = 0.3) -> Dict:
        """
        Classify a contract document into one of the predefined categories.
        
        Args:
            contract_text: The text content of the contract document
            confidence_threshold: Minimum confidence score to accept prediction
            
        Returns:
            Dictionary containing classification results
        """
        # Preprocess the text
        processed_text = self.preprocess_text(contract_text)
        
        if not processed_text or len(processed_text) < 50:
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "error": "Text too short or empty for classification"
            }
        
        try:
            # Perform zero-shot classification
            result = self.classifier(
                processed_text, 
                self.candidate_labels,
                multi_label=False
            )
            
            # Get top prediction
            top_label = result['labels'][0]
            top_confidence = result['scores'][0]
            
            # Check if confidence meets threshold
            if top_confidence < confidence_threshold:
                prediction = "Unknown"
                confidence = top_confidence
            else:
                prediction = top_label
                confidence = top_confidence
            
            # Get scores for all categories
            all_scores = {
                label: score for label, score in zip(result['labels'], result['scores'])
            }
            
            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "all_scores": all_scores,
                "text_preview": processed_text[:200] + "..." if len(processed_text) > 200 else processed_text,
                "error": None
            }
            
        except Exception as e:
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "error": f"Classification error: {str(e)}"
            }
    
    def batch_classify(self, contracts: List[str], 
                      confidence_threshold: float = 0.3) -> List[Dict]:
        """
        Classify multiple contracts in batch.
        
        Args:
            contracts: List of contract texts
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of classification results
        """
        results = []
        for i, contract_text in enumerate(contracts):
            result = self.classify_contract(contract_text, confidence_threshold)
            result['contract_index'] = i
            results.append(result)
        
        return results
    
    def get_classification_stats(self, results: List[Dict]) -> Dict:
        """
        Generate statistics from batch classification results.
        
        Args:
            results: List of classification results from batch_classify
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_contracts": len(results),
            "successful_classifications": 0,
            "failed_classifications": 0,
            "confidence_distribution": {label: 0 for label in self.candidate_labels},
            "confidence_distribution": {label: [] for label in self.candidate_labels},
            "average_confidence": 0.0
        }
        
        confidences = []
        
        for result in results:
            if result['error'] is None and result['prediction'] != "Unknown":
                stats['successful_classifications'] += 1
                pred = result['prediction']
                if pred in stats['confidence_distribution']:
                    stats['confidence_distribution'][pred].append(result['confidence'])
                confidences.append(result['confidence'])
            else:
                stats['failed_classifications'] += 1
        
        if confidences:
            stats['average_confidence'] = sum(confidences) / len(confidences)
        
        # Calculate distribution percentages
        stats['label_distribution'] = {}
        for label, conf_list in stats['confidence_distribution'].items():
            stats['label_distribution'][label] = len(conf_list)
        
        return stats

# Example usage and test function
def test_classifier():
    """Test the contract classifier with example texts"""
    
    # Initialize classifier
    classifier = ContractClassifier()
    
    # Example contract texts (in real usage, you'd load these from your JSON)
    example_contracts = [
        # NDA example
        "NON-DISCLOSURE AGREEMENT This Agreement is made between Company A and Company B...",
        
        # SLA example  
        "SERVICE LEVEL AGREEMENT This Agreement defines the service levels for IT support...",
        
        # Employment Contract example
        "EMPLOYMENT AGREEMENT This Contract is between Employer and Employee...",
        
        # Vendor Agreement example
        "VENDOR SERVICES AGREEMENT This Agreement governs the provision of services by Vendor...",
        
        # Partnership Agreement example
        "PARTNERSHIP AGREEMENT This Agreement establishes a partnership between Party A and Party B..."
    ]
    
    print("Testing Contract Classifier Service")
    print("=" * 50)
    
    # Test single classification
    test_text = "This NON-DISCLOSURE AGREEMENT contains confidential information..."
    result = classifier.classify_contract(test_text)
    print(f"Single classification result: {result}")
    
    print("\n" + "=" * 50)
    print("Batch Classification Results:")
    print("=" * 50)
    
    # Test batch classification
    batch_results = classifier.batch_classify(example_contracts)
    
    for i, result in enumerate(batch_results):
        print(f"Contract {i+1}:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        if result['error']:
            print(f"  Error: {result['error']}")
        print(f"  Preview: {result['text_preview']}")
        print()
    
    # Get statistics
    stats = classifier.get_classification_stats(batch_results)
    print("Classification Statistics:")
    print(f"Total contracts: {stats['total_contracts']}")
    print(f"Successful: {stats['successful_classifications']}")
    print(f"Failed/Unknown: {stats['failed_classifications']}")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    print("Label distribution:", stats['label_distribution'])

if __name__ == "__main__":
    test_classifier()