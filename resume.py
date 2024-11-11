import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
import json
import time
from typing import Optional, Dict, Any

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def clean_resume_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace('\x00', '')
    text = ' '.join(text.split())
    return text

# Debug log code taken from other sources
def process_resume(text: str, retry_count: int = 3) -> Optional[Dict[str, Any]]:
    
    text = clean_resume_text(text)
    if not text:
        print("Empty or invalid resume text")
        return None
        
    prompt = """Parse this resume and return ONLY a JSON object with this exact structure (no additional text or explanation):
    {
        "education": [
            {
                "degree": "",
                "field": "",
                "school": "",
                "graduation_date": ""
            }
        ],
        "experience": [
            {
                "title": "",
                "company": "",
                "duration": "",
                "responsibilities": []
            }
        ],
        "skills": [],
        "awards": []
    }"""
    
    # Code taken from someone else to avoid using too many tokens so I can stop it in time
    for attempt in range(retry_count):
        try:
            # Add delay between retries
            if attempt > 0:
                time.sleep(2 ** attempt)  
                
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a resume parser. Extract information in the requested JSON format. Return ONLY valid JSON."},
                    {"role": "user", "content": f"{prompt}\n\nResume text: {text[:4000]}"}  # Limit text length
                ],
                temperature=0,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Debug logging
            if attempt == retry_count - 1:  # On last attempt, log the response
                print(f"\nAPI Response:\n{response_text[:500]}...")
            
            # Try to parse JSON response
            try:
                parsed_data = json.loads(response_text)
                required_keys = {"education", "experience", "skills", "awards"}
                if all(key in parsed_data for key in required_keys):
                    return parsed_data
                else:
                    print(f"Missing required keys in response. Found keys: {parsed_data.keys()}")
                    continue
            except json.JSONDecodeError as je:
                print(f"JSON decode error on attempt {attempt + 1}: {str(je)}")
                continue
                
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt == retry_count - 1:
                return None
            continue
    
    return None

def main():
    # Load resumes
    try:
        df = pd.read_csv('Resume.csv')
        print(f"Loaded {len(df)} resumes")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Process resumes
    print("Processing resumes...")
    processed_resumes = []
    failed_resumes = []
    
    # Process resumes with progress bar
    for idx, row in tqdm(df.head(5).iterrows(), total=5):
        result = process_resume(row['Resume_str'])
        if result:
            result['original_category'] = row['Category']
            processed_resumes.append(result)
            print(f"\nSuccessfully processed resume {idx} in category: {row['Category']}")
        else:
            failed_resumes.append(idx)
            print(f"\nFailed to process resume {idx}")

    # Save results
    if processed_resumes:
        # Save successful results to JSON
        with open('processed_resumes.json', 'w') as f:
            json.dump(processed_resumes, f, indent=2)
        print(f"\nSuccessfully processed {len(processed_resumes)} resumes")
        print(f"Failed to process {len(failed_resumes)} resumes")
        
        # Show example
        print("\nExample of processed resume:")
        print(json.dumps(processed_resumes[0], indent=2))
    else:
        print("No resumes were successfully processed")
        
    # Save failed resume indices
    if failed_resumes:
        with open('failed_resumes.json', 'w') as f:
            json.dump(failed_resumes, f)
        print(f"\nSaved {len(failed_resumes)} failed resume indices to failed_resumes.json")

if __name__ == "__main__":
    main()