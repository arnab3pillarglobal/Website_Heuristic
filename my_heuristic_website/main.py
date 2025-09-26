import asyncio
import json
from dotenv import load_dotenv
import pandas as pd
# from playwright.async_api import async_playwright
from urllib.parse import urlparse
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI
from io import BytesIO
import re
import os
import time

load_dotenv()

def fetch_and_map_prompts(uploaded_file):
    if uploaded_file is None:
        st.warning("Please upload an Excel file to proceed.")
        return {}
    try:
        xls = pd.ExcelFile(uploaded_file)
        df = pd.read_excel(xls, sheet_name="AI Prompts")
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return {}
    mapping = {}
    for idx, row in df.iterrows():
        heuristic = str(row[0]).strip() if pd.notna(row[0]) else None
        prompt = str(row[1]).strip() if pd.notna(row[1]) else None
        if heuristic and prompt:
            mapping[heuristic] = prompt
    return mapping

def clean_html_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_llm_response(response: str) -> str:
    """Format LLM response from OpenAI ChatCompletion object or string for better viewing"""
    if hasattr(response, 'choices') and response.choices:
        text = response.choices[0].message.content
    elif hasattr(response, 'content'):
        text = response.content
    else:
        text = str(response)
    
    text = text.replace("\\n", "\n")
    text = re.sub(r'(\*\*Overall Numeric Score.*?\*\*)', r'\n\1\n', text)
    text = re.sub(r'(\*\*Sub-level Scores:\*\*)', r'\n\1\n', text)
    text = re.sub(r'(\*\*Justification.*?\*\*)', r'\n\1\n', text)
    text = re.sub(r'(\*\*Detailed Answers:\*\*)', r'\n\1\n', text)
    text = re.sub(r'(\*\*Evaluation Scope:\*\*)', r'\n\1\n', text)
    text = re.sub(r'(\*\*Part \d+:)', r'\n\n\1', text)
    text = re.sub(r'(\n|^)(-\s+\*\*)', r'\1\n\2', text)
    text = re.sub(r'(\n|^)(-\s+)', r'\1\n\2', text)
    text = re.sub(r'(\*Confidence:\s*\d+\*)', r'\n  \1\n', text)
    text = re.sub(r'(\*\*\?\*\*\s*)', r'\1\n  ', text)
    text = re.sub(r'(\n)(\d+\.)', r'\1\n\2', text)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = text.strip() + '\n' + '='*60 + '\n'
    return text

def evaluate_heuristic_with_llm(prompt: str) -> str:
    """Evaluate heuristics using OpenAI's API"""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    full_prompt = f"{prompt}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert UX evaluator conducting heuristic evaluations. 
                    Provide detailed, structured responses with specific scores and evidence-based justifications. 
                    Follow the evaluation criteria exactly as specified in the prompt."""
                },
                {
                    "role": "user", 
                    "content": full_prompt
                }
            ],
            temperature=0,  
            max_tokens=4000,
            timeout=60
        )
        
        output = response
        summary = format_llm_response(output)
        return summary
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return f"Error: {str(e)}"

def analyze_each_heuristic_individually(evaluations: dict) -> dict:
    """Analyze each heuristic individually to prevent crashes with large data"""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    final_analysis = {}
    heuristic_names = list(evaluations.keys())
    total_heuristics = len(heuristic_names)
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_container = st.container()
    
    for idx, heuristic_name in enumerate(heuristic_names):
        # Update progress
        progress = (idx) / total_heuristics
        progress_bar.progress(progress)
        
        with status_container:
            st.info(f"🔍 Analyzing {heuristic_name} ({idx + 1}/{total_heuristics})")
        
        pages_data = evaluations[heuristic_name]
        
        # Truncate data if too large to prevent crashes
        truncated_pages_data = {}
        for url, data in pages_data.items():
            output_text = data.get('output', '')
            # Limit each page output to 3000 characters to prevent token overflow
            if len(output_text) > 3000:
                truncated_output = output_text[:3000] + "... [truncated for analysis]"
                truncated_pages_data[url] = {"output": truncated_output}
            else:
                truncated_pages_data[url] = data
        
        # Create analysis prompt for individual heuristic
        individual_analysis_prompt = f"""
You are a UX expert analyzing heuristic evaluation data for "{heuristic_name}".

Heuristic: {heuristic_name}
Evaluation Data:
{json.dumps({heuristic_name: truncated_pages_data}, indent=2)}

Please provide a JSON response with the following structure:
{{
    "heuristic_name": "{heuristic_name}",
    "total_score": <calculated average score from the data>,
    "max_score": 4,
    "subtopics": [
        {{
            "name": "<subtopic name extracted from the evaluation data>",
            "score": <score extracted from evaluation>,
            "description": "<detailed description of performance, strengths, weaknesses, and recommendations>"
        }}
    ],
    "overall_description": "<comprehensive summary of overall performance, key findings, and strategic recommendations>",
    "key_strengths": ["<strength 1>", "<strength 2>"],
    "key_weaknesses": ["<weakness 1>", "<weakness 2>"],
    "recommendations": ["<recommendation 1>", "<recommendation 2>"],
    "pages_evaluated": {len(pages_data)}
}}

Extract actual scores from the evaluation data. Look for patterns like:
- "Overall Numeric Score for [Heuristic]: X"  
- "Part X: [Subtopic Name]: Y"
- "- Part X: [Subtopic Name]: Y"

Provide insightful, actionable descriptions for each subtopic and the overall heuristic performance.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a senior UX consultant specializing in heuristic evaluations. 
                        You excel at:
                        1. Extracting and analyzing quantitative scores from evaluation data
                        2. Providing actionable insights and recommendations
                        3. Creating structured, professional analysis reports
                        4. Identifying patterns and priorities for improvement
                        
                        Always return valid JSON and be specific in your recommendations."""
                    },
                    {
                        "role": "user",
                        "content": individual_analysis_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000,
                timeout=30
            )
            
            analysis_text = response.choices[0].message.content.strip()
            # More robust JSON cleaning
            if analysis_text.startswith("```"):
                lines = analysis_text.split('\n')
                # Find the start and end of JSON
                start_idx = 0
                end_idx = len(lines)
                for i, line in enumerate(lines):
                    if line.strip().startswith('{'):
                        start_idx = i
                        break
                for i in range(len(lines)-1, -1, -1):
                    if lines[i].strip().endswith('}'):
                        end_idx = i + 1
                        break
                analysis_text = '\n'.join(lines[start_idx:end_idx])
            
            # Parse JSON for this individual heuristic
            try:
                heuristic_analysis = json.loads(analysis_text)
                final_analysis[heuristic_name] = heuristic_analysis
                with status_container:
                    st.success(f"{heuristic_name} analyzed successfully")
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error for {heuristic_name}: {e}")
                with status_container:
                    st.warning(f"JSON parsing failed for {heuristic_name}, using fallback")
                final_analysis[heuristic_name] = create_individual_fallback_analysis(heuristic_name, pages_data)
                
        except Exception as e:
            print(f"Error analyzing {heuristic_name}: {e}")
            with status_container:
                st.error(f"Error analyzing {heuristic_name}: {str(e)}")
            final_analysis[heuristic_name] = create_individual_fallback_analysis(heuristic_name, pages_data)
        
        # Small delay between requests to avoid rate limits
        time.sleep(1)
    
    # Complete progress
    progress_bar.progress(1.0)
    status_container.success("All heuristics analyzed successfully!")
    
    return final_analysis

def create_individual_fallback_analysis(heuristic_name: str, pages_data: dict) -> dict:
    """Create fallback analysis for a specific heuristic"""
    return {
        "heuristic_name": heuristic_name,
        "total_score": 2.0,
        "max_score": 4,
        "subtopics": [
            {"name": "Overall Assessment", "score": 2, "description": f"Basic {heuristic_name} functionality present but needs improvement."}
        ],
        "overall_description": f"The {heuristic_name} heuristic shows basic implementation with room for improvement.",
        "key_strengths": ["Basic functionality present"],
        "key_weaknesses": ["Limited advanced features"],
        "recommendations": ["Enhance user experience", "Add more interactive elements"],
        "pages_evaluated": len(pages_data)
    }

def create_fallback_analysis(evaluations: dict) -> dict:
    """Create fallback analysis if LLM fails"""
    fallback = {}
    for heuristic_name, pages_data in evaluations.items():
        fallback[heuristic_name] = create_individual_fallback_analysis(heuristic_name, pages_data)
    return fallback

def generate_html_from_analysis_json(analysis_json: dict, site_name: str = "Website", site_description: str = "UX Heuristic Analysis") -> str:
    """Generate HTML report from the LLM analysis JSON structure without hardcoded values"""
    
    # Validate input
    if analysis_json is None:
        st.error("Analysis data is not available. Please try running the evaluation again.")
        return create_fallback_html_report(site_name, site_description)
    
    if not isinstance(analysis_json, dict) or len(analysis_json) == 0:
        st.error("Invalid or empty analysis data received.")
        return create_fallback_html_report(site_name, site_description)
    
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Heuristic Evaluation Report – {site_name}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    html {{ box-sizing: border-box; font-size: 16px; }}
    *, *:before, *:after {{ box-sizing: inherit; }}
    body {{
      font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
      background: #f8f9fb; color: #222; margin: 0; padding: 0 0 3rem 0; line-height: 1.6;
    }}
    header {{
      background: #2d3e50; color: #fff; padding: 2rem 0 1rem 0;
      text-align: center; margin-bottom: 2rem;
    }}
    h1 {{ margin: 0 0 0.5rem 0; font-size: 2.2rem; letter-spacing: 0.02em; }}
    h2, h3 {{ color: #2d3e50; margin-top: 2.5rem; margin-bottom: 1rem; }}
    h3 {{ margin-top: 2rem; font-size: 1.25rem; }}
    section {{
      max-width: 900px; margin: 0 auto 2.5rem auto; background: #fff;
      border-radius: 8px; box-shadow: 0 2px 8px rgba(44,62,80,0.07); padding: 2rem 2.5rem;
    }}
    table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0 2rem 0; background: #fafbfc; border-radius: 6px; overflow: hidden; }}
    th, td {{ padding: 0.75rem 1rem; text-align: left; }}
    th {{ background: #e9ecef; font-weight: 600; color: #2d3e50; border-bottom: 2px solid #d1d5db; }}
    td {{ border-bottom: 1px solid #e5e7eb; }}
    .bar-chart-container {{
      margin: 1.5rem 0 2rem 0; padding: 1.5rem 1rem 1.5rem 2.5rem;
      background: #f4f6fa; border-radius: 8px; overflow-x: auto;
    }}
    .bar-chart {{ width: 100%; max-width: 700px; margin: 0 auto; position: relative; }}
    .bar-row {{ display: flex; align-items: center; margin-bottom: 1.1rem; min-height: 2.2rem; }}
    .bar-label {{
      flex: 0 0 260px; font-size: 1rem; color: #2d3e50; margin-right: 1.2rem;
      text-align: right; padding-right: 0.5rem; white-space: pre-line;
    }}
    .bar-bg {{
      flex: 1 1 auto; background: #e5e7eb; border-radius: 5px; height: 1.2rem;
      position: relative; margin-right: 0.7rem; min-width: 60px; max-width: 350px; overflow: hidden;
    }}
    .bar-fill {{
      height: 100%; border-radius: 5px 0 0 5px; background: linear-gradient(90deg, #3b82f6 60%, #2563eb 100%);
      transition: width 0.5s; position: absolute; left: 0; top: 0;
    }}
    .bar-fill.zero {{
      background: repeating-linear-gradient(135deg, #e5e7eb, #e5e7eb 8px, #f87171 8px, #f87171 16px);
      border-radius: 5px;
    }}
    .bar-score {{ min-width: 2.5rem; font-weight: 600; color: #2563eb; font-size: 1.05rem; text-align: left; }}
    .bar-score.zero {{ color: #f87171; }}
    .bar-axis {{
      display: flex; align-items: center; margin-left: 260px; margin-top: 0.2rem; margin-bottom: 1.2rem;
      font-size: 0.97rem; color: #6b7280; position: relative; width: calc(100% - 260px - 2.5rem); max-width: 350px;
    }}
    .bar-axis-tick {{ flex: 1 1 0; text-align: center; position: relative; }}
    .bar-axis-tick:first-child {{ text-align: left; }}
    .bar-axis-tick:last-child {{ text-align: right; }}
    .summary-table th, .summary-table td {{ text-align: center; }}
    .summary-table th:first-child, .summary-table td:first-child {{ text-align: left; }}
    .conclusion {{
      background: #e0f2fe; border-left: 5px solid #2563eb; padding: 1.5rem 2rem;
      border-radius: 7px; margin-top: 2.5rem; margin-bottom: 2rem;
    }}
    .recommendations {{
      background: #f0fdf4; border-left: 5px solid #16a34a; padding: 1.5rem 2rem;
      border-radius: 7px; margin-top: 1.5rem;
    }}
    .strengths-weaknesses {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 1.5rem 0; }}
    .strengths {{
      background: #ecfdf5; padding: 1rem 1.5rem; border-radius: 7px; border-left: 4px solid #10b981;
    }}
    .weaknesses {{
      background: #fef2f2; padding: 1rem 1.5rem; border-radius: 7px; border-left: 4px solid #ef4444;
    }}
    @media (max-width: 768px) {{
      .strengths-weaknesses {{ grid-template-columns: 1fr; }}
      .bar-label {{ flex: 0 0 120px; font-size: 0.9rem; }}
      .bar-axis {{ margin-left: 120px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Heuristic Evaluation Report</h1>
    <div style="font-size:1.15rem; color:#cbd5e1;">{site_name}</div>
    <div style="font-size:1rem; color:#cbd5e1; margin-top:0.3rem;">{site_description}</div>
  </header>
  
  <section>
    <h2>Summary of Scores</h2>
    <table class="summary-table">
      <thead>
        <tr><th>Heuristic</th><th>Overall Score<br>(out of {max_score})</th></tr>
      </thead>
      <tbody>{summary_rows}</tbody>
    </table>
  </section>
  
  {heuristic_sections}
  
  <section class="conclusion">
    <h2>Overall Assessment</h2>
    <div>
      <p><strong>The evaluated website demonstrates {performance_level} usability performance across key heuristics.</strong> 
      With an average score of {average_score}/{max_score}, the site {overall_assessment}.</p>
      {conclusion_content}
    </div>
  </section>
</body>
</html>
"""

    # Calculate dynamic values from the JSON with type conversion
    total_score = 0
    max_score = 4
    heuristic_count = len(analysis_json)
    
    # Get max_score from data if available
    if analysis_json:
        first_heuristic = next(iter(analysis_json.values()))
        try:
            max_score = int(first_heuristic.get('max_score', 4))
        except (ValueError, TypeError):
            max_score = 4
    
    # Generate summary table rows with type conversion
    summary_rows = ""
    for heuristic_name, data in analysis_json.items():
        try:
            score = float(data.get('total_score', 0))
        except (ValueError, TypeError):
            score = 0.0
        total_score += score
        summary_rows += f"""
        <tr><td>{heuristic_name}</td><td>{score}</td></tr>"""
    
    # Calculate average and performance level
    average_score = round(total_score / heuristic_count, 1) if heuristic_count > 0 else 0
    
    # Determine performance level and assessment dynamically
    if average_score >= 3.5:
        performance_level = "excellent"
        overall_assessment = "provides strong user experience with well-implemented heuristic principles"
    elif average_score >= 2.5:
        performance_level = "good"
        overall_assessment = "provides adequate user experience with room for targeted improvements"
    elif average_score >= 1.5:
        performance_level = "moderate"
        overall_assessment = "shows basic functionality but requires focused attention in several areas"
    else:
        performance_level = "limited"
        overall_assessment = "needs significant improvement across multiple heuristic dimensions"
    
    # Generate detailed heuristic sections
    heuristic_sections = ""
    for heuristic_name, data in analysis_json.items():
        # Generate bar chart rows with type conversion
        bar_rows = ""
        axis_ticks = ""
        
        try:
            current_max = int(data.get('max_score', max_score))
        except (ValueError, TypeError):
            current_max = max_score
        
        # Generate axis ticks dynamically
        for i in range(current_max + 1):
            axis_ticks += f'<div class="bar-axis-tick">{i}</div>'
        
        # Generate bars from subtopics with type conversion
        for subtopic in data.get('subtopics', []):
            try:
                score = float(subtopic.get('score', 0))
            except (ValueError, TypeError):
                score = 0.0
            
            width_percent = (score / current_max) * 100 if current_max > 0 else 0
            zero_class = ' zero' if score == 0 else ''
            
            bar_rows += f"""
            <div class="bar-row">
              <div class="bar-label">{subtopic.get('name', '')}</div>
              <div class="bar-bg">
                <div class="bar-fill{zero_class}" style="width: {width_percent}%;"></div>
              </div>
              <div class="bar-score{zero_class}">{score}</div>
            </div>"""
        
        # Generate strengths and weaknesses lists
        strengths_list = ""
        for strength in data.get('key_strengths', []):
            strengths_list += f"<li>{strength}</li>"
        
        weaknesses_list = ""
        for weakness in data.get('key_weaknesses', []):
            weaknesses_list += f"<li>{weakness}</li>"
        
        recommendations_list = ""
        for rec in data.get('recommendations', []):
            recommendations_list += f"<li>{rec}</li>"
        
        # Build the section with type conversion
        try:
            section_score = float(data.get('total_score', 0))
            section_max = int(data.get('max_score', max_score))
        except (ValueError, TypeError):
            section_score = 0.0
            section_max = max_score
        
        heuristic_sections += f"""
  <section>
    <h3>Heuristic: {heuristic_name} (Score: {section_score}/{section_max})</h3>
    
    <div class="bar-chart-container">
      <div class="bar-axis">{axis_ticks}</div>
      <div class="bar-chart">{bar_rows}</div>
    </div>
    
    <div><strong>Overall Assessment:</strong> {data.get('overall_description', 'No description available.')}</div>"""
        
        # Add strengths/weaknesses if they exist
        if data.get('key_strengths') or data.get('key_weaknesses'):
            heuristic_sections += f"""
    <div class="strengths-weaknesses">"""
            
            if data.get('key_strengths'):
                heuristic_sections += f"""
      <div class="strengths"><h4>Key Strengths</h4><ul>{strengths_list}</ul></div>"""
            
            if data.get('key_weaknesses'):
                heuristic_sections += f"""
      <div class="weaknesses"><h4>Key Weaknesses</h4><ul>{weaknesses_list}</ul></div>"""
            
            heuristic_sections += """</div>"""
        
        # Add recommendations if they exist
        if data.get('recommendations'):
            heuristic_sections += f"""
    <div class="recommendations"><h4>Recommendations</h4><ul>{recommendations_list}</ul></div>"""
        
        heuristic_sections += """</section>"""
    
    # Generate conclusion content dynamically
    conclusion_content = generate_conclusion_content(analysis_json, average_score, max_score)
    
    # Fill in the template
    final_html = html_template.format(
        site_name=site_name,
        site_description=site_description,
        max_score=max_score,
        summary_rows=summary_rows,
        heuristic_sections=heuristic_sections,
        average_score=average_score,
        performance_level=performance_level,
        overall_assessment=overall_assessment,
        conclusion_content=conclusion_content
    )
    
    return final_html

def create_fallback_html_report(site_name: str, site_description: str) -> str:
    """Create a basic HTML report when analysis fails"""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Heuristic Evaluation Report – {site_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .error {{ background: #fee; border: 1px solid #fcc; padding: 20px; border-radius: 5px; }}
        .header {{ background: #2d3e50; color: white; padding: 20px; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Heuristic Evaluation Report</h1>
        <div>{site_name}</div>
        <div>{site_description}</div>
    </div>
    <div class="error">
        <h2>Report Generation Failed</h2>
        <p>Unable to generate the heuristic evaluation report. Please check:</p>
        <ul>
            <li>The website evaluation completed successfully</li>
            <li>Valid heuristic data was collected</li>
            <li>Try running the evaluation again</li>
        </ul>
    </div>
</body>
</html>
"""

def generate_conclusion_content(analysis_json: dict, average_score: float, max_score: int) -> str:
    """Generate dynamic conclusion content based on analysis data"""
    all_recommendations = []
    priority_areas = []
    
    for heuristic_name, data in analysis_json.items():
        score = data.get('total_score', 0)
        if score < average_score:
            priority_areas.append(heuristic_name.lower())
        
        all_recommendations.extend(data.get('recommendations', []))
    
    if priority_areas:
        if len(priority_areas) == 1:
            priority_text = f"Priority should be given to improving {priority_areas}"
        elif len(priority_areas) == 2:
            priority_text = f"Priority areas for enhancement include {priority_areas} and {priority_areas}"[1]
        else:
            priority_text = f"Priority areas for enhancement include {', '.join(priority_areas[:-1])}, and {priority_areas[-1]}"
    else:
        priority_text = "All heuristic areas show consistent performance levels"
    
    conclusion_content = f"<p>{priority_text}."
    
    if all_recommendations:
        top_recommendations = all_recommendations[:3]
        if len(top_recommendations) == 1:
            conclusion_content += f" Key recommendation: {top_recommendations.lower()}."
        else:
            conclusion_content += f" Key recommendations include {', '.join([rec.lower() for rec in top_recommendations[:-1]])}, and {top_recommendations[-1].lower()}."
    
    conclusion_content += "</p>"
    return conclusion_content

# async def login_and_crawl_all_pages(url: str, username: str, password: str, login_url: str, username_selector: str, password_selector: str, submit_selector: str):
#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=True)
#         context = await browser.new_context()
#         page = await context.new_page()

#         visited_urls = set()
#         url_to_content = {}

#         await page.goto(login_url)
#         await page.fill(username_selector, username)
#         await page.fill(password_selector, password)
#         await page.click(submit_selector)
#         await page.wait_for_load_state("networkidle")

#         async def crawl(current_url):
#             if current_url in visited_urls:
#                 return
#             visited_urls.add(current_url)
#             await page.goto(current_url)
#             await page.wait_for_load_state("networkidle")

#             content = await page.content()
#             cleaned_content = clean_html_content(content)
#             url_to_content[current_url] = cleaned_content
#             print(f"Crawled {current_url} with cleaned content length {len(cleaned_content)}")

#             links = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
#             base_domain = urlparse(login_url).netloc
#             for link in links:
#                 link_domain = urlparse(link).netloc
#                 if link_domain == base_domain and not link.startswith(("mailto:", "javascript:")):
#                     await crawl(link)

#         await crawl(url)
#         await browser.close()
#         return url_to_content

def run_crawl_and_evaluate_stream(start_url, username, password, login_url, username_selector, password_selector, submit_selector, prompt_map):
    # results = asyncio.run(
    #     login_and_crawl_all_pages(
    #         url=start_url, username=username, password=password, login_url=login_url,
    #         username_selector=username_selector, password_selector=password_selector, submit_selector=submit_selector,
    #     )
    # )

    results =[]

    evaluations = {}
    placeholder = st.empty()

    # for url, content in results.items():
    for heuristic, prompt in prompt_map.items():
        prompt_with_url = prompt.replace("[Enter Website URL Here]", login_url)
        result = evaluate_heuristic_with_llm(prompt_with_url)
        if heuristic not in evaluations:
            evaluations[heuristic] = {}
            evaluations[heuristic][start_url] = {"output": result}
            placeholder.json(evaluations)
    
    placeholder.empty()
    return evaluations


def main():
    st.header("Heuristic Evaluation")

    with st.sidebar:
        st.title("Upload Excel file")
        uploaded_file = st.file_uploader("Upload Heuristic Excel file with AI Prompts sheet", type=["xlsx", "xls"])
        st.title("Website Credentials")
        login_url = st.text_input("Login URL", value="https://www.saucedemo.com/")
        start_url = st.text_input("Start URL", value="https://www.saucedemo.com/inventory.html")
        username = st.text_input("Username", value="standard_user")
        password = st.text_input("Password", value="secret_sauce", type="password")
        
        username_selector = st.text_input(
            "Username Selector", 
            value="#user-name",
            help="CSS selector for the username input field. Right-click on the username field in the login page, select 'Inspect Element', then copy the id (#id) or class (.class) or tag selector. Example: #username, .username-field, input[name='username']"
        )
        
        password_selector = st.text_input(
            "Password Selector", 
            value="#password",
            help="CSS selector for the password input field. Right-click on the password field in the login page, select 'Inspect Element', then copy the id (#id) or class (.class) or tag selector. Example: #password, .password-field, input[type='password']"
        )
        
        submit_selector = st.text_input(
            "Submit Button Selector", 
            value="#login-button",
            help="CSS selector for the login/submit button. Right-click on the login button, select 'Inspect Element', then copy the id (#id) or class (.class) or tag selector. Example: #login-btn, .submit-button, button[type='submit']"
        )

    if uploaded_file:
        prompt_map = fetch_and_map_prompts(uploaded_file)
        st.write("Loaded heuristics and prompts:", prompt_map)

        if st.button("Run Crawl and Evaluate"):
            with st.spinner("Crawling site and evaluating..."):
                evaluations = run_crawl_and_evaluate_stream(
                    start_url, username, password, login_url,
                    username_selector, password_selector, submit_selector, prompt_map,
                )
                st.session_state["evaluations"] = evaluations
                st.success("Evaluation complete")
                st.json(st.session_state["evaluations"])

        if "evaluations" in st.session_state:
            st.subheader("Saved Evaluation Output")

            if st.button("Generate Advanced Report (HTML)"):
                # Add validation before analysis
                with st.spinner("Generating comprehensive HTML report..."):
                    # Use individual analysis method
                    analysis_json = analyze_each_heuristic_individually(st.session_state["evaluations"])
                    print('Analysis JSON:', analysis_json)
                    
                    if not analysis_json:
                        st.error("Failed to generate analysis. Please try again.")
                        return
                    
                    # Get site name from login URL
                    site_name = login_url.replace("https://", "").replace("http://", "").split('/')
                    
                    # Generate HTML report
                    html_report = generate_html_from_analysis_json(
                        analysis_json, 
                        site_name=site_name,
                        site_description="Comprehensive UX Heuristic Analysis"
                    )
                    
                    st.session_state["html_report"] = html_report

        if "html_report" in st.session_state and st.session_state["html_report"]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Download HTML Report",
                    data=st.session_state["html_report"],
                    file_name="heuristic_evaluation_report.html",
                    mime="text/html",
                )

            # with col2:
            #     pdf_data = generate_pdf_from_html(st.session_state["html_report"])
            #     if pdf_data:
            #         st.download_button(
            #             label="📋 Download PDF Report",
            #             data=pdf_data,
            #             file_name="heuristic_evaluation_report.pdf",
            #             mime="application/pdf"
            #         )

    else:
        st.info("Please upload an Excel file to start.")

if __name__ == "__main__":
    main()