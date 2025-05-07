"""
Script to translate SAE tweets back to AAE using GPT-4o-mini.
"""

import os
import sys
import csv
import pandas as pd
import logging
import argparse
import asyncio
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import GPT4oMiniInterface

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("sae_to_aae_translation.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def load_sae_dataset(filepath: str, num_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load the SAE dataset (previously translated from AAE) from a CSV file.
    
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

async def translate_sae_to_aae_async(texts: List[str], api_key: str) -> List[str]:
    """
    Translate SAE texts to AAE asynchronously.
    
    Args:
        texts: List of SAE texts to translate
        api_key: OpenAI API key
        
    Returns:
        List of translated AAE texts
    """
    model = GPT4oMiniInterface(api_key)
    translations = await model.async_batch_call_model(
        [f"""
        Translate the following tweet from Standard American English (SAE) to African American English (AAE).
        Preserve the meaning, tone, and intent of the original tweet.
        Only change dialectical features while maintaining the original message.
        Original tweet (SAE): "{text}"
        African American English translation:
        """ for text in texts]
    )
    
    # Post-process to handle potential errors or unexpected responses
    processed_translations = []
    for i, translation in enumerate(translations):
        if translation.startswith("ERROR:") or "African American English translation:" in translation:
            # Extract only the translation part if the model returns the prompt
            parts = translation.split("African American English translation:")
            if len(parts) > 1:
                processed_translations.append(parts[1].strip())
            else:
                logger.warning(f"Translation failed for text {i}: {translation}")
                processed_translations.append(f"TRANSLATION_FAILED: {texts[i]}")
        else:
            processed_translations.append(translation)
    
    return processed_translations

def main():
    parser = argparse.ArgumentParser(description="Translate SAE tweets back to AAE")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file with SAE tweets (and original AAE)")
    parser.add_argument("--output", "-o", required=True, help="Path to output CSV file for AAE translations")
    parser.add_argument("--samples", "-s", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="Batch size for API calls (default: 10)")
    parser.add_argument("--api-key", "-k", help="OpenAI API key (if not provided, will use environment variable)")
    args = parser.parse_args()
    
    # Load the dataset
    df = load_sae_dataset(args.input, args.samples)
    
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("No API key provided. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Extract SAE texts to translate
    if "sae_text" in df.columns:
        sae_texts = df["sae_text"].tolist()
    else:
        logger.error("Input CSV must have 'sae_text' column with SAE translations")
        sys.exit(1)
    
    # Translate in batches
    all_translations = []
    
    async def process_batches():
        for i in tqdm(range(0, len(df), args.batch_size), desc="Translating"):
            batch_texts = sae_texts[i:i+args.batch_size]
            
            try:
                translations = await translate_sae_to_aae_async(batch_texts, api_key)
                all_translations.extend(translations)
                
                # Log progress
                if (i // args.batch_size) % 10 == 0:
                    logger.info(f"Processed {i+len(translations)}/{len(df)} records")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//args.batch_size}: {str(e)}")
                # Add placeholder for failed translations
                all_translations.extend([f"TRANSLATION_FAILED: {text}" for text in batch_texts])
    
    # Run the async processing
    asyncio.run(process_batches())
    
    # Create the output DataFrame
    output_df = pd.DataFrame({
        "original_aae_text": df["original_text"].tolist() if "original_text" in df.columns else [""] * len(df),
        "sae_text": sae_texts,
        "aae_from_sae_text": all_translations
    })
    
    # Save the output
    output_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(output_df)} translations to {args.output}")

if __name__ == "__main__":
    main()