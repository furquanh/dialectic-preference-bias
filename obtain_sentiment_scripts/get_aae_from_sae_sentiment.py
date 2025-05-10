"""
Script to obtain sentiment classifications for AAE texts translated from SAE using GPT-4.1 Batch API.
"""

import os
import sys
import csv
import pandas as pd
import logging
import argparse
import asyncio
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import GPT41BatchInterface, ClaudeHaikuInterface, Phi3MediumInterface

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("aae_from_sae_sentiment_analysis.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def load_dataset(filepath: str, num_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load the AAE-from-SAE translation dataset from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        num_samples: Number of samples to load, if None, load all
        
    Returns:
        DataFrame containing the dataset
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with {len(df)} records")
        
        if num_samples:
            df = df.sample(num_samples, random_state=42) if len(df) > num_samples else df
            logger.info(f"Sampled {len(df)} records")
            
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def get_model_interface(model_name: str, api_key: Optional[str] = None):
    """
    Get the appropriate model interface based on the model name.
    
    Args:
        model_name: Name of the model to use
        api_key: API key for the model (if applicable)
        
    Returns:
        Model interface object
    """
    if model_name == 'gpt4o_mini':
        from models import GPT4oMiniInterface
        return GPT4oMiniInterface(api_key=api_key)
    elif model_name == 'gpt41_batch':
        from models import GPT41BatchInterface
        return GPT41BatchInterface(api_key=api_key)
    elif model_name == 'claude_haiku':
        from models import ClaudeHaikuInterface
        return ClaudeHaikuInterface(api_key=api_key)
    elif model_name == 'phi3_medium':
        from models import Phi3MediumInterface
        return Phi3MediumInterface()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def analyze_sentiment_batch(texts: List[str], model_interface) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for a batch of texts using the GPT-4.1 Batch API.
    
    Args:
        texts: List of texts to analyze
        model_interface: Model interface for sentiment analysis
        
    Returns:
        List of sentiment dictionaries
    """
    # If using GPT-4.1 Batch, use the specialized batch method
    if isinstance(model_interface, GPT41BatchInterface):
        sentiments = model_interface.batch_get_sentiment(texts)
        
        # Add the text to each sentiment dictionary
        for i, sentiment in enumerate(sentiments):
            sentiment['text'] = texts[i]
        
        return sentiments
    else:
        # For other models, use their batch method
        sentiments = model_interface.batch_get_sentiment(texts)
        
        # Add the text to each sentiment dictionary
        for i, sentiment in enumerate(sentiments):
            sentiment['text'] = texts[i]
        
        return sentiments

def main():
    parser = argparse.ArgumentParser(description="Analyze sentiment for AAE translations from SAE using GPT-4.1 Batch API")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file with AAE-from-SAE translations")
    parser.add_argument("--output", "-o", required=True, help="Path to output CSV file for sentiment results")
    parser.add_argument("--model", "-m", required=True, 
                        choices=['gpt4o_mini', 'gpt41_batch', 'claude_haiku', 'phi3_medium'], 
                        help="Model to use for sentiment analysis")
    parser.add_argument("--samples", "-s", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Batch size for processing (default: 100)")
    parser.add_argument("--api-key", "-k", help="API key for the selected model (if applicable)")
    parser.add_argument("--text-column", "-t", default="aae_from_sae_text", help="Column name containing the AAE-from-SAE text")
    args = parser.parse_args()
    
    # Load the dataset
    df = load_dataset(args.input, args.samples)
    
    # Get the model interface
    try:
        model_interface = get_model_interface(args.model, args.api_key)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Extract texts to analyze
    if args.text_column in df.columns:
        texts = df[args.text_column].tolist()
    else:
        logger.error(f"Input CSV must have '{args.text_column}' column with text to analyze")
        sys.exit(1)
    
    # Analyze sentiment in batches
    all_sentiments = []
    
    # Process batches
    for i in tqdm(range(0, len(df), args.batch_size), desc="Analyzing sentiment"):
        batch_texts = texts[i:i+args.batch_size]
        
        try:
            sentiments = analyze_sentiment_batch(batch_texts, model_interface)
            all_sentiments.extend(sentiments)
            
            # Log progress
            if (i // args.batch_size) % 10 == 0:
                logger.info(f"Processed {i+len(sentiments)}/{len(df)} records")
            
        except Exception as e:
            logger.error(f"Error processing batch {i//args.batch_size}: {str(e)}")
            # Add placeholder for failed analyses
            failed_sentiments = [
                {
                    'text': text,
                    'sentiment': 'ERROR',
                    'score': 0,
                    'raw_response': f"Batch processing error: {str(e)}"
                } for text in batch_texts
            ]
            all_sentiments.extend(failed_sentiments)
        
        # Clean up GPU memory if using Phi-3
        if args.model == 'phi3_medium' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create the output DataFrame
    output_df = pd.DataFrame({
        'original_aae_text': df['original_aae_text'].tolist() if 'original_aae_text' in df.columns else [""] * len(df),
        'sae_text': df['sae_text'].tolist() if 'sae_text' in df.columns else [""] * len(df),
        'aae_from_sae_text': [s['text'] for s in all_sentiments],
        'sentiment': [s['sentiment'] for s in all_sentiments],
        'score': [s['score'] for s in all_sentiments],
        'raw_response': [s.get('raw_response', '') for s in all_sentiments]
    })
    
    # Save the output
    output_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(output_df)} sentiment analyses to {args.output}")
    
    # Print some statistics
    sentiment_counts = output_df['sentiment'].value_counts()
    logger.info(f"Sentiment distribution: {dict(sentiment_counts)}")
    
    error_count = sum(1 for s in output_df['sentiment'] if s == 'ERROR')
    if error_count > 0:
        logger.warning(f"{error_count} records had errors during processing")

if __name__ == "__main__":
    main()