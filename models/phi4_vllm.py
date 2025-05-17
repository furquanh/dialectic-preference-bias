"""
Phi-4 model interface using vLLM direct API.
"""

import os
import logging
import json
import torch
from typing import Dict, List, Any, Optional
from vllm import LLM, SamplingParams

# Change relative import to absolute import
try:
    from .model_interface import  ModelInterface
except ImportError:
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from models.model_interface import  ModelInterface

logger = logging.getLogger(__name__)

class Phi4VllmInterface(ModelInterface):
    """Interface for Phi-4 using vLLM direct API."""
    
    def __init__(self, model_id: str = "microsoft/phi-4", dtype: str = "bfloat16"):
        """Initialize the Phi-4 interface with direct vLLM API.
        
        Args:
            model_id: HuggingFace model ID for Phi-4
            dtype: Data type for model weights (bfloat16, float16, etc.)
        """
        super().__init__("phi4_vllm")
        self.model_id = model_id
        self.dtype = dtype
        self.llm = None
        # Reduce max_tokens to 20 for sentiment analysis to get concise responses
        self.sampling_params = SamplingParams(temperature=0.5, top_p=1.0, max_tokens=10)
        logger.info(f"Initializing Phi-4 with vLLM direct API using model {self.model_id}")
        
        # Initialize the LLM (lazy loading - will load on first call)
        self._load_model()
    
    def _load_model(self):
        """Load the vLLM model if not already loaded."""
        if self.llm is None:
            try:
                logger.info(f"Loading model {self.model_id} with dtype {self.dtype}")
                self.llm = LLM(model=self.model_id, dtype=self.dtype)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
    
    def call_model(self, text: str) -> str:
        """Call Phi-4 with a text prompt.
        
        Args:
            text: Input text/prompt
            
        Returns:
            Model response as string
        """
        try:
            # Ensure model is loaded
            self._load_model()
            
            # Format as chat prompt
            prompt = f"<|user|>\n{text}<|assistant|>\n"
            
            # Generate response
            output = self.llm.generate([prompt], self.sampling_params)[0]
            
            # Extract and return response
            return output.outputs[0].text.strip()
                
        except Exception as e:
            logger.error(f"Error calling Phi-4 via vLLM: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def batch_call_model(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Efficient batch processing for Phi-4 using vLLM's native batching.
        
        Args:
            texts: List of input texts
            batch_size: Size of each batch (note: vLLM handles batching internally)
            
        Returns:
            List of model responses
        """
        results = []
        
        # Process in batches to prevent memory issues with very large inputs
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Format prompts as chat messages
                prompts = [f"<|user|>\n{text}<|assistant|>\n" for text in batch]
                
                # Generate responses for the batch
                outputs = self.llm.generate(prompts, self.sampling_params)
                
                # Extract responses
                batch_results = [output.outputs[0].text.strip() for output in outputs]
                
                results.extend(batch_results)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                failed_results = ["ERROR: Processing failed"] * len(batch)
                results.extend(failed_results)
                
                # Try to recover GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results
    
    def translate_aae_to_sae(self, text: str) -> str:
        """Translate text from AAE to SAE.
        
        Args:
            text: Input text in AAE
            
        Returns:
            Translated text in SAE
        """
        prompt = f"""
        Translate the following text from African American English (AAE) to Standard American English (SAE).
        Preserve the meaning, tone, and intent of the original text.
        Only change dialectical features while maintaining the original message.
        
        Original text (AAE): "{text}"
        
        Standard American English translation:
        """
        
        return self.call_model(prompt)
    
    def translate_sae_to_aae(self, text: str) -> str:
        """Translate text from SAE to AAE.
        
        Args:
            text: Input text in SAE
            
        Returns:
            Translated text in AAE
        """
        prompt = f"""
        Translate the following text from Standard American English (SAE) to African American English (AAE).
        Preserve the meaning, tone, and intent of the original text.
        Only change dialectical features while maintaining the original message.
        
        Original text (SAE): "{text}"
        
        African American English translation:
        """
        
        return self.call_model(prompt)
    
    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment classification using Phi-4.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with sentiment classification
        """
        prompt = f"""
        Analyze the sentiment of this text: "{text}"
        
        IMPORTANT: Respond with ONLY ONE WORD - either 'positive', 'negative', or 'neutral'.
        Do not explain your answer. Just output the single word.
        
        Sentiment:
        """
        
        try:
            response = self.call_model(prompt)
            response = response.strip().lower()
            #logger.info(f"Raw response: {response}")
            
            # Improved sentiment detection - checking for exact matches first
            if response == 'positive':
                sentiment = 'positive'
                score = 1
            elif response == 'negative':
                sentiment = 'negative'
                score = -1
            elif response == 'neutral':
                sentiment = 'neutral'
                score = 0
            # Fallback to contains if no exact match
            elif 'positive' in response and 'negative' not in response:
                sentiment = 'positive'
                score = 1
            elif 'negative' in response:
                sentiment = 'negative'
                score = -1
            elif 'neutral' in response:
                sentiment = 'neutral'
                score = 0
            else:
                logger.warning(f"Unexpected sentiment response: {response}")
                sentiment = 'ERROR'
                score = 0
            
            return {
                'sentiment': sentiment,
                'score': score,
                'raw_response': response
            }
        except Exception as e:
            logger.error(f"Error getting sentiment: {str(e)}")
            return {'sentiment': 'ERROR', 'score': 0, 'raw_response': str(e)}
    
    def batch_get_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Get sentiment classification for a batch of texts using Phi-4's batch processing.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of dictionaries with sentiment classifications
        """
        # Create sentiment analysis prompts with stricter instructions
        prompts = [
            f"""
            Analyze the sentiment of this text: "{text}"
            
            IMPORTANT: Respond with ONLY ONE WORD - either 'positive', 'negative', or 'neutral'.
            Do not explain your answer. Just output the single word.
            
            Sentiment:
            """
            for text in texts
        ]
        
        try:
            # Format prompts as chat messages
            chat_prompts = [f"<|user|>\n{prompt}<|assistant|>\n" for prompt in prompts]
            
            # Generate responses for the batch
            outputs = self.llm.generate(chat_prompts, self.sampling_params)
            
            # Process responses
            results = []
            for i, output in enumerate(outputs):
                response = output.outputs[0].text.strip().lower()
                logger.info(f"Raw response: {response}")
                
                # Improved sentiment detection - checking for exact matches first
                if response == 'positive':
                    sentiment = 'positive'
                    score = 1
                elif response == 'negative':
                    sentiment = 'negative'
                    score = -1
                elif response == 'neutral':
                    sentiment = 'neutral'
                    score = 0
                # Fallback to contains if no exact match
                elif 'positive' in response and 'negative' not in response:
                    sentiment = 'positive'
                    score = 1
                elif 'negative' in response:
                    sentiment = 'negative'
                    score = -1
                elif 'neutral' in response:
                    sentiment = 'neutral'
                    score = 0
                else:
                    logger.warning(f"Unexpected sentiment response: {response}")
                    sentiment = 'ERROR'
                    score = 0
                
                results.append({
                    'sentiment': sentiment,
                    'score': score,
                    'raw_response': response
                })
            
            return results
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {str(e)}")
            return [{'sentiment': 'ERROR', 'score': 0, 'raw_response': str(e)} for _ in texts]


def call_phi4_vllm(text: str) -> str:
    """Function to call Phi-4 model via vLLM.
    
    Args:
        text: Input text/prompt
        
    Returns:
        Model response
    """
    model = Phi4VllmInterface()
    return model.call_model(text)


def batch_call_phi4_vllm(texts: List[str], batch_size: int = 8) -> List[str]:
    """Function to call Phi-4 model via vLLM in batch mode.
    
    Args:
        texts: List of input texts/prompts
        batch_size: Size of each batch
        
    Returns:
        List of model responses
    """
    model = Phi4VllmInterface()
    return model.batch_call_model(texts, batch_size)

if __name__ == "__main__":
    # Example usage
    model = Phi4VllmInterface()
    text = f"""
            Analyze the sentiment of this text: "Oh, babe is in his feelings. F***, I need a blunt, a cigarette, something. LOL, I love you, babe. ;)"
            
            IMPORTANT: Respond with ONLY ONE WORD - either 'positive', 'negative', or 'neutral'.
            Do not explain your answer. Just output the single word.
            
            Sentiment:
            """
    response = model.call_model(text)
    print(f"Response: {response}\n\n")
    text = f"""
            Analyze the sentiment of this text: "Aww, Toby. You get on my nerves, but at the same time, I love you. Shaking my head."
            
            IMPORTANT: Respond with ONLY ONE WORD - either 'positive', 'negative', or 'neutral'.
            Do not explain your answer. Just output the single word.
            
            Sentiment:
            """
    response = model.call_model(text)
    print(f"Response: {response}")
    