import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from typing import Optional, Dict, Any

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def clean_resume_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace('\x00', '')
    text = ' '.join(text.split())
    return text

def process_resume(text: str) -> Optional[Dict[str, Any]]:
    text = clean_resume_text(text)
    if not text:
        print("Empty or invalid resume text")
        return None
        
    prompt = """Parse this resume into a detailed JSON object. For each section, extract as much specific information as possible:

{
    "education": [
        {
            "degree": "", // Specific degree title
            "level": "", // PhD, Masters, Bachelors, Associate, Certificate, or Other
            "field": "", // Specific field of study
            "school": "", // Full school name
            "graduation_date": "", // Format as "YYYY-MM" if available
            "gpa": null, // Include if mentioned
            "honors": [] // Any academic honors mentioned
        }
    ],
    "experience": [
        {
            "title": "", // Exact job title
            "company": "", // Company name
            "duration": "", // Original date format
            "start_date": "", // Format as "YYYY-MM" if possible
            "end_date": "", // Format as "YYYY-MM" or "Current"
            "responsibilities": [], // List of specific responsibilities
            "achievements": [], // Measurable achievements if mentioned
            "location": "" // Location if mentioned
        }
    ],
    "skills": {
        "technical": [], // Technical/hard skills
        "soft": [], // Soft/interpersonal skills
        "certifications": [] // Professional certifications
    },
    "awards": [
        {
            "title": "",
            "issuer": "",
            "date": "",
            "description": ""
        }
    ],
    "languages": [] // Languages if mentioned
}

Extract specific details, dates, and measurable achievements. Ensure all dates are consistent and all lists contain unique items."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise resume parser. Extract detailed, specific information in the requested JSON format. Be thorough and accurate."},
                {"role": "user", "content": f"{prompt}\n\nResume text: {text[:6000]}"}
            ],
            temperature=0,
            max_tokens=2048
        )
        
        response_text = response.choices[0].message.content.strip()
        parsed_data = json.loads(response_text)
        
        required_keys = {"education", "experience", "skills", "awards", "languages"}
        if all(key in parsed_data for key in required_keys):
            return parsed_data
        else:
            print(f"Missing required keys in response. Found keys: {parsed_data.keys()}")
            return None
            
    except Exception as e:
        print(f"Error processing resume: {str(e)}")
        return None

def main():
    df = pd.read_csv('Resume.csv')
    print(f"Loaded {len(df)} resumes")

    print("Processing first 5 resumes...")
    processed_resumes = []
    failed_resumes = []
    
    # Only process first 5 resumes
    for idx, row in df.head(5).iterrows():
        print(f"\nProcessing resume {idx + 1} of 5...")
        result = process_resume(row['Resume_str'])
        if result:
            result['original_category'] = row['Category'] # To check original category
            processed_resumes.append(result)
            print(f"Successfully processed resume {idx} in category: {row['Category']}")
        else:
            failed_resumes.append(idx)
            print(f"Failed to process resume {idx}")

    if processed_resumes:
        with open('processed_resumes.json', 'w') as f:
            json.dump(processed_resumes, f, indent=2)
        print(f"\nSuccessfully processed {len(processed_resumes)} resumes")
        print(f"Failed to process {len(failed_resumes)} resumes")
        print("\nProcessed resume:")
        print(json.dumps(processed_resumes[0], indent=2))
    else:
        print("No resumes were successfully processed")
        
    if failed_resumes:
        with open('failed_resumes.json', 'w') as f:
            json.dump(failed_resumes, f)
        print(f"\nSaved {len(failed_resumes)} failed resume indices to failed_resumes.json")

if __name__ == "__main__":
    main()
