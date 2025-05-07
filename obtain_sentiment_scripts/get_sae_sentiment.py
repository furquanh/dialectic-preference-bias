"""
Script to obtain sentiment classifications for SAE translations of tweets.
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

from models import GPT4oMiniInterface, ClaudeHaikuInterface, Phi3MediumInterface

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("sae_sentiment_analysis.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def load_dataset(filepath: str, num_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load the SAE translation dataset from a CSV file.
    
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
        model_name: Name of the model to use ('gpt4o_mini', 'claude_haiku', or 'phi3_medium')
        api_key: API key for API-based models
        
    Returns:
        Model interface instance
    """
    if model_name == 'gpt4o_mini':
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No API key provided for GPT-4o-mini")
        return GPT4oMiniInterface(api_key)
    elif model_name == 'claude_haiku':
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No API key provided for Claude 3 Haiku")
        return ClaudeHaikuInterface(api_key)
    elif model_name == 'phi3_medium':
        return Phi3MediumInterface()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

async def analyze_sentiment_async(texts: List[str], model_interface) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for the given texts using the specified model interface.
    
    Args:
        texts: List of texts to analyze
        model_interface: Model interface to use
        
    Returns:
        List of sentiment dictionaries
    """
    if hasattr(model_interface, 'async_batch_call_model'):
        prompts = [f"""
        Please analyze the sentiment of the following text and respond with exactly one word: 
        either 'positive', 'negative', or 'neutral'.

        Text: "{text}"
        
        Sentiment:
        """ for text in texts]
        
        responses = await model_interface.async_batch_call_model(prompts)
        
        # Process the responses
        sentiments = []
        for i, response in enumerate(responses):
            try:
                response = response.strip().lower()
                
                if 'positive' in response:
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
                    
                sentiments.append({
                    'text': texts[i],
                    'sentiment': sentiment,
                    'score': score,
                    'raw_response': response
                })
            except Exception as e:
                logger.error(f"Error processing response for text {i}: {str(e)}")
                sentiments.append({
                    'text': texts[i],
                    'sentiment': 'ERROR',
                    'score': 0,
                    'raw_response': str(e)
                })
        
        return sentiments
    else:
        # Fall back to synchronous processing
        sentiments = []
        for text in texts:
            try:
                result = model_interface.get_sentiment(text)
                result['text'] = text
                sentiments.append(result)
            except Exception as e:
                logger.error(f"Error getting sentiment for text: {str(e)}")
                sentiments.append({
                    'text': text,
                    'sentiment': 'ERROR',
                    'score': 0,
                    'raw_response': str(e)
                })
        
        return sentiments

def main():
    parser = argparse.ArgumentParser(description="Analyze sentiment for SAE translations")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file with SAE translations")
    parser.add_argument("--output", "-o", required=True, help="Path to output CSV file for sentiment results")
    parser.add_argument("--model", "-m", required=True, choices=['gpt4o_mini', 'claude_haiku', 'phi3_medium'], 
                        help="Model to use for sentiment analysis")
    parser.add_argument("--samples", "-s", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size for processing (default: 16)")
    parser.add_argument("--api-key", "-k", help="API key for the selected model (if applicable)")
    parser.add_argument("--text-column", "-t", default="sae_text", help="Column name containing the SAE text to analyze")
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
    
    async def process_batches():
        for i in tqdm(range(0, len(df), args.batch_size), desc="Analyzing sentiment"):
            batch_texts = texts[i:i+args.batch_size]
            
            try:
                sentiments = await analyze_sentiment_async(batch_texts, model_interface)
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
    
    # Run the async processing
    asyncio.run(process_batches())
    
    # Create the output DataFrame
    output_df = pd.DataFrame({
        'original_text': df['original_text'].tolist() if 'original_text' in df.columns else [""] * len(df),
        'sae_text': [s['text'] for s in all_sentiments],
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