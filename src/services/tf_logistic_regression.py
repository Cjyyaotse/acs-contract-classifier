# contract_classifier_service.py
import pickle
import re
import numpy as np
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContractClassifierService:
    """
    Service for classifying contract documents using a pre-trained TF-IDF model.
    """

    def __init__(self, model_path: str = 'models/TF_IDF_Logitic_Regression.pkl'):
        """
        Initialize the contract classifier service.

        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.tfidf = None
        self.classifier = None
        self.classes = None
        self.is_loaded = False

        # Load model during initialization
        self.load_model()

    def load_model(self) -> bool:
        """
        Load the trained TF-IDF vectorizer and classifier from disk.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.tfidf = model_data['tfidf']
            self.classifier = model_data['classifier']
            self.classes = model_data['classes']
            self.is_loaded = True

            logger.info(f"✅ Model loaded successfully from {self.model_path}")
            logger.info(f"📊 Available classes: {list(self.classes)}")
            return True

        except FileNotFoundError:
            logger.error(f"❌ Model file not found: {self.model_path}")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            return False

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess contract text for classification.

        Args:
            text: Raw contract text

        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Replace common legal abbreviations
        legal_replacements = {
            'inc.': 'incorporated',
            'llc': 'limited liability company',
            'ltd.': 'limited',
            'corp.': 'corporation',
            '&': 'and',
            'w/': 'with',
            'w/o': 'without'
        }

        for abbrev, full_form in legal_replacements.items():
            text = text.replace(abbrev, full_form)

        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep important legal terms
        text = re.sub(r'[^\w\s-]', ' ', text)

        # Remove extra spaces
        text = text.strip()

        return text

    def classify_contract(self, contract_text: str,
                         confidence_threshold: float = 0.3,
                         top_n: int = 3) -> Dict[str, Any]:
        """
        Classify a contract document into one of the predefined categories.
        Always returns the category with the highest score, never "Unknown".

        Args:
            contract_text: The text content of the contract document
            confidence_threshold: Minimum confidence score for confidence flag
            top_n: Number of top predictions to return

        Returns:
            Dictionary containing classification results
        """
        if not self.is_loaded:
            return {
                "success": False,
                "prediction": "Error",
                "confidence": 0.0,
                "all_scores": {},
                "is_confident": False,
                "error": "Model not loaded. Please load the model first.",
                "top_predictions": [],
                "text_preview": ""
            }

        if not contract_text or len(contract_text.strip()) < 20:
            return {
                "success": False,
                "prediction": "Error",
                "confidence": 0.0,
                "all_scores": {},
                "is_confident": False,
                "error": "Text too short for classification",
                "top_predictions": [],
                "text_preview": contract_text[:200] + "..." if len(contract_text) > 200 else contract_text
            }

        try:
            # Preprocess the text
            processed_text = self.preprocess_text(contract_text)

            # Transform text to TF-IDF features
            text_vector = self.tfidf.transform([processed_text])

            # Get prediction probabilities
            probabilities = self.classifier.predict_proba(text_vector)[0]

            # Create dictionary of all scores
            all_scores = {
                str(self.classes[i]): float(prob)
                for i, prob in enumerate(probabilities)
            }

            # Get top prediction (always use the highest scoring category)
            top_class_idx = np.argmax(probabilities)
            top_confidence = probabilities[top_class_idx]
            top_prediction = str(self.classes[top_class_idx])

            # Get top N predictions
            top_n_indices = np.argsort(probabilities)[-top_n:][::-1]
            top_predictions = [
                {
                    "category": str(self.classes[i]),
                    "confidence": float(probabilities[i])
                }
                for i in top_n_indices
            ]

            # Always return the highest scoring category, never "Unknown"
            # Use confidence_threshold only to set is_confident flag
            is_confident = top_confidence >= confidence_threshold

            return {
                "success": True,
                "prediction": top_prediction,  # Always the highest scoring category
                "confidence": float(top_confidence),
                "all_scores": all_scores,
                "is_confident": is_confident,  # Flag indicating if confidence meets threshold
                "top_predictions": top_predictions,
                "text_preview": processed_text[:200] + "..." if len(processed_text) > 200 else processed_text,
                "error": None
            }

        except Exception as e:
            logger.error(f"❌ Classification error: {str(e)}")
            return {
                "success": False,
                "prediction": "Error",
                "confidence": 0.0,
                "all_scores": {},
                "is_confident": False,
                "top_predictions": [],
                "error": f"Classification error: {str(e)}",
                "text_preview": contract_text[:200] + "..." if len(contract_text) > 200 else contract_text
            }

    def classify_batch(self, contract_texts: List[str],
                      confidence_threshold: float = 0.3,
                      top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Classify multiple contract documents in batch.
        Always returns the category with the highest score for each document.

        Args:
            contract_texts: List of contract texts to classify
            confidence_threshold: Minimum confidence score for confidence flag
            top_n: Number of top predictions to return

        Returns:
            List of classification results
        """
        results = []

        for i, text in enumerate(contract_texts):
            result = self.classify_contract(text, confidence_threshold, top_n)
            result['document_index'] = i
            results.append(result)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {
                "loaded": False,
                "error": "Model not loaded"
            }

        return {
            "loaded": True,
            "model_path": self.model_path,
            "num_classes": len(self.classes) if self.classes is not None else 0,
            "classes": list(self.classes) if self.classes is not None else [],
            "classifier_type": type(self.classifier).__name__ if self.classifier else "Unknown",
            "feature_count": self.tfidf.get_feature_names_out().shape[0] if self.tfidf else 0
        }

# Example usage and test function
def test_classifier():
    """Test the contract classifier service"""

    # Initialize the service
    classifier_service = ContractClassifierService()

    # Check if model loaded successfully
    model_info = classifier_service.get_model_info()
    print("📋 Model Information:")
    print(f"   Loaded: {model_info['loaded']}")
    if model_info['loaded']:
        print(f"   Classes: {model_info['classes']}")
        print(f"   Classifier: {model_info['classifier_type']}")
        print(f"   Features: {model_info['feature_count']}")

    # Test contracts
    test_contracts = [
        "This employment agreement is between ABC Corp and John Doe for the position of software engineer with a salary of $100,000 per year.",
        "Confidential information disclosed under this Non-Disclosure Agreement must be kept secret by all parties for a period of 5 years.",
        "The partnership between Smith and Jones will share all profits equally and both parties agree to contribute capital to the business.",
        "Vendor shall deliver goods pursuant to the following terms and conditions outlined in this vendor agreement.",
        "Service Level Agreement: The provider guarantees 99.9% uptime and will provide 24/7 support for all critical issues.",
        "This is a very short text that should be rejected."
    ]

    print("\n🧪 Testing Single Classification:")
    print("=" * 50)

    # Test single classification
    single_result = classifier_service.classify_contract(test_contracts[0])
    print(f"Text: {test_contracts[0][:60]}...")
    print(f"Prediction: {single_result['prediction']}")
    print(f"Confidence: {single_result['confidence']:.3f}")
    print(f"Confident: {single_result['is_confident']}")
    print(f"Top predictions: {[f'{p['category']}: {p['confidence']:.3f}' for p in single_result['top_predictions']]}")

    print("\n🧪 Testing Batch Classification:")
    print("=" * 50)

    # Test batch classification
    batch_results = classifier_service.classify_batch(test_contracts)

    for i, result in enumerate(batch_results):
        status = "✅" if result['success'] else "❌"
        confidence_status = "✓" if result.get('is_confident', False) else "✗"
        print(f"{status} Document {i+1}: {result['prediction']} (Confidence: {result['confidence']:.3f}) {confidence_status}")
        if result['error']:
            print(f"   Error: {result['error']}")

    # Print detailed results for the first successful prediction
    print("\n📊 Detailed results for first successful prediction:")
    successful_results = [r for r in batch_results if r['success']]
    if successful_results:
        first_success = successful_results[0]
        print(f"Prediction: {first_success['prediction']}")
        print(f"Confidence: {first_success['confidence']:.3f}")
        print(f"Confident: {first_success['is_confident']}")
        print("All scores:")
        for category, score in first_success['all_scores'].items():
            print(f"  {category}: {score:.4f}")

if __name__ == "__main__":
    test_classifier()
