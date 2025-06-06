import streamlit as st
import json
import os
from datetime import datetime
from collections import defaultdict
import re
import io

# PDF and Charting
import matplotlib
matplotlib.use('Agg') # Use Agg backend for scripts without a GUI
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup

# ReportLab for PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, ListFlowable, KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.pagesizes import A4

# LLM SDKs
import google.generativeai as genai
from groq import Groq

# --- Backend Functions (Copied from the previous robust script) ---
# Subject mapping
SUBJECT_ID_TO_NAME_MAPPING = {
    "68387e8c19acf6636e6acfab": "Physics", # From sample_submission_analysis_2.json
    "68387e8c19acf6636e6acfac": "Chemistry",
    "68387e8c19acf6636e6acfad": "Maths",
    # Add more from other sample files if needed
    "607018ee404ae53194e73d92": "Physics",
    "607018ee404ae53194e73d90": "Chemistry",
    "607018ee404ae53194e73d91": "Maths"
}
SUBJECT_TITLE_KEYWORDS = {
    "physics": "Physics", "chemistry": "Chemistry", "maths": "Maths",
    "mathematics": "Maths", "bio": "Biology", "biology": "Biology"
}

def load_json_data(uploaded_file_obj):
    if uploaded_file_obj is None:
        return None
    try:
        # To read an uploaded file, it's already in bytes or string mode.
        # If it's an UploadedFile object from Streamlit, use its read method.
        string_data = uploaded_file_obj.getvalue().decode("utf-8")
        return json.loads(string_data)
    except json.JSONDecodeError as e:
        st.error(f"Error: Could not decode JSON. Details: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading JSON data: {e}")
        return None

def parse_syllabus(html_syllabus):
    if not html_syllabus: return "Syllabus not provided."
    soup = BeautifulSoup(html_syllabus, 'html.parser')
    text_parts = []
    title_tag = soup.find('h1')
    if title_tag: text_parts.append(title_tag.get_text(strip=True))
    subject_headings = soup.find_all('h2')
    for heading in subject_headings:
        text_parts.append(f"\n{heading.get_text(strip=True)}:")
        ul = heading.find_next_sibling('ul')
        if ul:
            for li in ul.find_all('li'): text_parts.append(f"  • {li.get_text(strip=True)}")
    return "\n".join(text_parts)

def get_subject_name_from_id(subject_oid):
    return SUBJECT_ID_TO_NAME_MAPPING.get(subject_oid, f"Subject ID: {subject_oid[:6]}...")

def get_subject_name_from_section_title(section_title_str):
    title_lower = section_title_str.lower()
    for keyword, name in SUBJECT_TITLE_KEYWORDS.items():
        if keyword in title_lower: return name
    return "General Section"

def safe_division(numerator, denominator, default=0.0):
    return numerator / denominator if denominator != 0 else default

def format_time(seconds, short=False):
    if seconds is None: return "N/A"
    seconds = int(seconds)
    if short:
        m, s = divmod(seconds, 60)
        return f"{m:01d}m{s:02d}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m:02d}m {s:02d}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:01d}h {m:02d}m {s:02d}s"

def process_submission_data(data_list): # Expecting a list from JSON
    if not data_list or not isinstance(data_list, list) or not data_list[0]:
        st.error("Error: Invalid or empty data structure provided for processing.")
        return None
    
    data = data_list[0] # The actual report data is the first element
    test_details = data.get("test", {})

    processed = {
        "student_info": {"name": data.get("userId", {}).get("name", "Valued Student")},
        "test_info": {
            "name": test_details.get("syllabus", "N/A").splitlines()[0].replace("<h1>","").replace("</h1>","").strip() if test_details.get("syllabus") else "Test Analysis",
            "date": datetime.now().strftime("%B %d, %Y"),
            "syllabus_raw": test_details.get("syllabus", ""),
            "syllabus_parsed": parse_syllabus(test_details.get("syllabus", "")),
            "total_time_allowed_min": test_details.get("totalTime", 0),
            "total_questions": test_details.get("totalQuestions", 0),
            "total_marks": test_details.get("totalMarks", 0),
        },
        "overall_performance": {
            "time_taken_sec": data.get("totalTimeTaken", 0), "marks_scored": data.get("totalMarkScored", 0),
            "attempted": data.get("totalAttempted", 0), "correct": data.get("totalCorrect", 0),
            "accuracy_on_attempted": safe_division(data.get("totalCorrect", 0) * 100, data.get("totalAttempted", 1)),
            "accuracy_overall": safe_division(data.get("totalCorrect", 0) * 100, test_details.get("totalQuestions", 1)),
            "percentage_score": safe_division(data.get("totalMarkScored", 0) * 100, test_details.get("totalMarks", 1)),
        },
        "subjects_performance": [],
        "chapter_performance": defaultdict(lambda: defaultdict(lambda: {'attempted': 0, 'correct': 0, 'incorrect': 0, 'time_taken_sec': 0, 'questions': [] })),
        "difficulty_performance": defaultdict(lambda: {'attempted': 0, 'correct': 0, 'time_taken_sec': 0, 'marks_scored':0}),
        "time_analysis": {
            "time_per_correct_q_sec": 0, "time_per_incorrect_q_sec": 0, "time_per_attempted_q_sec": 0,
            "time_on_correct_sec": 0, "time_on_incorrect_sec": 0, "time_efficiency_percentage": 0,
            "questions_over_avg_time_incorrect": [],
        }
    }
    
    total_time_allowed_sec = processed["test_info"]["total_time_allowed_min"] * 60
    processed["time_analysis"]["time_efficiency_percentage"] = max(0, safe_division(
        (total_time_allowed_sec - processed["overall_performance"]["time_taken_sec"]) * 100,
        total_time_allowed_sec if total_time_allowed_sec > 0 else 1 ))

    marks_per_question = safe_division(processed["test_info"]["total_marks"], processed["test_info"]["total_questions"], 4)

    temp_subject_perf_from_array = {}
    for sub_data in data.get("subjects", []):
        sub_oid = sub_data.get("subjectId", {}).get("$oid")
        sub_name = get_subject_name_from_id(sub_oid)
        temp_subject_perf_from_array[sub_name] = {
            "marks_scored": sub_data.get("totalMarkScored", 0), "attempted": sub_data.get("totalAttempted", 0),
            "correct": sub_data.get("totalCorrect", 0), "accuracy_on_attempted": sub_data.get("accuracy", 0),
            "time_taken_sec": sub_data.get("totalTimeTaken", 0)
        }

    all_questions_data = []
    subject_question_counts = defaultdict(int)
    subject_total_marks_possible = defaultdict(float)

    for section in data.get("sections", []):
        section_title_str = section.get("sectionId", {}).get("title", "Unknown Section")
        current_section_subject = get_subject_name_from_section_title(section_title_str)

        for q_detail in section.get("questions", []):
            q_info = q_detail.get("questionId", {})
            status = q_detail.get("status", "notAttempted")
            time_taken = q_detail.get("timeTaken", 0)
            is_correct = False; answered_incorrectly = False

            if status == "answered":
                if q_detail.get("markedOptions") and q_detail["markedOptions"]: is_correct = q_detail["markedOptions"][0].get("isCorrect", False)
                elif q_detail.get("inputValue") and q_detail["inputValue"].get("value") is not None: is_correct = q_detail["inputValue"].get("isCorrect", False)
                if not is_correct: answered_incorrectly = True
            
            chapter_title = q_info.get("chapters", [{}])[0].get("title", "Unknown Chapter")
            
            all_questions_data.append({
                "subject": current_section_subject, "chapter": chapter_title,
                "topic": q_info.get("topics", [{}])[0].get("title", "Unknown Topic"),
                "concept": q_info.get("concepts", [{}])[0].get("title", "Unknown Concept"),
                "level": q_info.get("level", "medium"),
                "text": q_info.get("question", {}).get("text", "N/A")[:100] + "...",
                "status": status, "is_correct": is_correct, "answered_incorrectly": answered_incorrectly,
                "time_taken_sec": time_taken, "marks_per_q": marks_per_question
            })
            subject_question_counts[current_section_subject] += 1
            subject_total_marks_possible[current_section_subject] += marks_per_question

    for qd in all_questions_data:
        ctp = processed["chapter_performance"][qd["subject"]][qd["chapter"]]
        if qd["status"] == "answered":
            ctp['attempted'] += 1; ctp['time_taken_sec'] += qd["time_taken_sec"]
            if qd["is_correct"]: ctp['correct'] += 1
            else: ctp['incorrect'] +=1
        ctp['questions'].append({"text": qd["text"], "level": qd["level"], "correct": qd["is_correct"], "status": qd["status"], "time_taken": qd["time_taken_sec"]})

        dp = processed["difficulty_performance"][qd["level"]]
        if qd["status"] == "answered":
            dp['attempted'] += 1; dp['time_taken_sec'] += qd["time_taken_sec"]
            if qd["is_correct"]: dp['correct'] += 1; dp['marks_scored'] += qd["marks_per_q"]
        
        if qd["status"] == "answered":
            if qd["is_correct"]: processed["time_analysis"]["time_on_correct_sec"] += qd["time_taken_sec"]
            else: processed["time_analysis"]["time_on_incorrect_sec"] += qd["time_taken_sec"]

    for sub_name, sub_array_data in temp_subject_perf_from_array.items():
        total_q_in_sub = subject_question_counts.get(sub_name, sub_array_data["attempted"] if sub_array_data["attempted"] > 0 else 1)
        max_marks_sub = subject_total_marks_possible.get(sub_name, sub_array_data["attempted"] * marks_per_question if sub_array_data["attempted"] > 0 else marks_per_question * 25)
        processed["subjects_performance"].append({
            "name": sub_name, "attempted": sub_array_data["attempted"], "correct": sub_array_data["correct"],
            "marks_scored": sub_array_data["marks_scored"], "total_marks_possible": max_marks_sub,
            "total_questions_in_subject": total_q_in_sub,
            "accuracy_on_attempted": sub_array_data["accuracy_on_attempted"],
            "accuracy_overall_subject": safe_division(sub_array_data["correct"] * 100, total_q_in_sub),
            "time_taken_sec": sub_array_data["time_taken_sec"],
            "avg_time_per_attempted_q_sec": safe_division(sub_array_data["time_taken_sec"], sub_array_data["attempted"])
        })

    op_correct = processed["overall_performance"]["correct"]; op_attempted = processed["overall_performance"]["attempted"]
    op_incorrect = op_attempted - op_correct
    processed["time_analysis"]["time_per_correct_q_sec"] = safe_division(processed["time_analysis"]["time_on_correct_sec"], op_correct)
    processed["time_analysis"]["time_per_incorrect_q_sec"] = safe_division(processed["time_analysis"]["time_on_incorrect_sec"], op_incorrect)
    processed["time_analysis"]["time_per_attempted_q_sec"] = safe_division(processed["time_analysis"]["time_on_correct_sec"] + processed["time_analysis"]["time_on_incorrect_sec"], op_attempted)
    
    avg_time_by_level = {level: safe_division(data['time_taken_sec'], data['attempted']) for level, data in processed["difficulty_performance"].items() if data['attempted'] > 0}
    for qd in all_questions_data:
        if qd["answered_incorrectly"]:
            avg_lvl_t = avg_time_by_level.get(qd["level"], processed["time_analysis"]["time_per_attempted_q_sec"])
            if avg_lvl_t > 0 and qd["time_taken_sec"] > 1.5 * avg_lvl_t:
                 processed["time_analysis"]["questions_over_avg_time_incorrect"].append({
                     "text": qd["text"], "time_taken": format_time(qd["time_taken_sec"]),
                     "level": qd["level"], "subject": qd["subject"], "chapter": qd["chapter"],
                     "avg_time_for_level": format_time(avg_lvl_t)})
    return processed

# --- Chart Generation (Matplotlib for PDF) ---
def generate_charts(processed_data, output_dir="charts_rl_st"): # Changed output_dir name
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    chart_paths = {}
    plt.style.use('seaborn-v0_8-whitegrid')
    title_fs, label_fs, tick_fs, legend_fs, dpi = 10, 8, 7, 7, 150

    # 1. Overall Score Donut Chart
    try:
        op, tm = processed_data["overall_performance"], processed_data["test_info"]["total_marks"]
        ms, mu = op["marks_scored"], tm - op["marks_scored"]
        if tm > 0:
            fig, ax = plt.subplots(figsize=(3,3))
            ax.pie([ms, mu], labels=['Scored', 'Unscored'], autopct='%1.1f%%', startangle=90, colors=['#66BB6A', '#FFCDD2'], wedgeprops={'width':0.4, 'edgecolor':'w'}, textprops={'fontsize': tick_fs})
            ax.axis('equal'); ax.set_title(f'Overall Score: {ms:.0f}/{tm:.0f}', fontsize=title_fs)
            path = os.path.join(output_dir, "overall_score_donut.png"); plt.savefig(path, bbox_inches='tight', dpi=dpi); plt.close(fig)
            chart_paths["overall_score"] = path
    except Exception as e:
        if 'st' in globals(): st.warning(f"Chart Error (Overall Score): {e}")
        else: print(f"Chart Error (Overall Score): {e}")


    # 2. Subject-wise Marks Bar Chart
    try:
        sp_list = processed_data["subjects_performance"]
        names, scored, possible = [s["name"] for s in sp_list], [s["marks_scored"] for s in sp_list], [s["total_marks_possible"] for s in sp_list]
        if names:
            x, width = np.arange(len(names)), 0.4
            fig, ax = plt.subplots(figsize=(4.5, 3))
            rects1 = ax.bar(x, scored, width, label='Scored', color='#42A5F5')
            ax.bar(x, [max(0, p - s) for s, p in zip(scored, possible)], width, bottom=scored, label='Missed Potential', color='#FFCA28')
            ax.set_ylabel('Marks', fontsize=label_fs); ax.set_title('Subject-wise Marks', fontsize=title_fs)
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=tick_fs)
            ax.legend(fontsize=legend_fs, loc='upper right'); ax.grid(axis='y', linestyle='--', alpha=0.7)
            for r in rects1:
                h = r.get_height()
                offset_y = 0.05 * max(possible) if max(possible) > 0 else 1 
                text_y_pos = h - offset_y if h > 0.1 * max(possible) else h + (0.02 * max(possible) if max(possible) > 0 else 0.5)
                va_pos = 'bottom' if h > 0.1 * max(possible) else 'top'
                text_color = 'white' if h > 0.1 * max(possible) else 'black'

                ax.text(r.get_x() + r.get_width()/2., text_y_pos, f'{int(h)}',
                        ha='center', va=va_pos, color=text_color,
                        fontsize=tick_fs-1, fontweight='bold')
            path = os.path.join(output_dir, "subject_marks_bar.png"); plt.tight_layout(); plt.savefig(path, dpi=dpi); plt.close(fig)
            chart_paths["subject_marks"] = path
    except Exception as e:
        if 'st' in globals(): st.warning(f"Chart Error (Subject Marks): {e}")
        else: print(f"Chart Error (Subject Marks): {e}")


    # 3. Accuracy on Attempted (Subject-wise)
    try:
        sp_list = processed_data["subjects_performance"]
        labels, accuracies = [s["name"] for s in sp_list], [s["accuracy_on_attempted"] for s in sp_list]
        if labels: 
            fig, ax = plt.subplots(figsize=(4.5, 3))
            bars = ax.bar(labels, accuracies, color='#26C6DA', width=0.5)
            ax.set_ylabel('Accuracy on Attempted (%)', fontsize=label_fs); ax.set_title('Subject Accuracy (Attempted)', fontsize=title_fs)
            ax.set_ylim(0, 105); ax.grid(axis='y', linestyle='--', alpha=0.7); plt.xticks(rotation=30, ha="right", fontsize=tick_fs)
            for bar in bars: yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.0f}%', ha='center', va='bottom', fontsize=tick_fs-1)
            path = os.path.join(output_dir, "subject_accuracy_bar.png"); plt.tight_layout(); plt.savefig(path, dpi=dpi); plt.close(fig)
            chart_paths["subject_accuracy"] = path
    except Exception as e:
        if 'st' in globals(): st.warning(f"Chart Error (Subject Accuracy): {e}")
        else: print(f"Chart Error (Subject Accuracy): {e}")
    return chart_paths

def prepare_llm_context_summary(processed_data):
    summary_parts = [f"Student Performance Analysis for Test: {processed_data['test_info']['name']}\n"]
    op, ti = processed_data["overall_performance"], processed_data["test_info"]
    summary_parts.append(f"--- Overall ---\nScore: {op['marks_scored']:.0f}/{ti['total_marks']} ({op['percentage_score']:.1f}%)\nAttempted: {op['attempted']}/{ti['total_questions']}, Correct: {op['correct']}\nAccuracy (Attempted): {op['accuracy_on_attempted']:.1f}%, Accuracy (Overall): {op['accuracy_overall']:.1f}%\nTime Taken: {format_time(op['time_taken_sec'])} / {format_time(ti['total_time_allowed_min']*60)} (Efficiency: {processed_data['time_analysis']['time_efficiency_percentage']:.1f}%)")
    summary_parts.append("\n--- Subject Performance ---")
    for sp in processed_data["subjects_performance"]: summary_parts.append(f"  {sp['name']}: Score {sp['marks_scored']:.0f}/{sp['total_marks_possible']:.0f}, Acc(Att): {sp['accuracy_on_attempted']:.1f}%, Correct: {sp['correct']}/{sp['attempted']}, Time: {format_time(sp['time_taken_sec'], short=True)}")
    summary_parts.append("\n--- Chapter Highlights (Illustrative) ---")
    chap_hl_count = 0
    for subj, chaps in processed_data["chapter_performance"].items():
        if chap_hl_count >= 4: break
        strong, weak = [], []
        for ch_n, ch_d in chaps.items():
            if ch_d['attempted'] > 1:
                acc = safe_division(ch_d['correct']*100, ch_d['attempted']); avg_t = safe_division(ch_d['time_taken_sec'], ch_d['attempted'])
                if acc < 50: weak.append((ch_n, acc, ch_d['correct'], ch_d['attempted'], avg_t))
                elif acc > 75: strong.append((ch_n, acc, ch_d['correct'], ch_d['attempted'], avg_t))
        strong.sort(key=lambda x: x[1], reverse=True); weak.sort(key=lambda x: x[1])
        if weak and chap_hl_count < 4: ch_n, acc, co, att, avg_t = weak[0]; summary_parts.append(f"  Focus ({subj}): {ch_n} ({acc:.0f}% Acc, {co}/{att}, AvgT: {format_time(avg_t, short=True)})"); chap_hl_count+=1
        if strong and chap_hl_count < 4: ch_n, acc, co, att, avg_t = strong[0]; summary_parts.append(f"  Strength ({subj}): {ch_n} ({acc:.0f}% Acc, {co}/{att}, AvgT: {format_time(avg_t, short=True)})"); chap_hl_count+=1
    summary_parts.append("\n--- Difficulty Level Performance ---")
    for lvl, dp in processed_data["difficulty_performance"].items():
        if dp['attempted'] > 0: acc = safe_division(dp['correct']*100,dp['attempted']); avg_t=safe_division(dp['time_taken_sec'],dp['attempted']); summary_parts.append(f"  {lvl.capitalize()}: Att {dp['attempted']}, Corr {dp['correct']}, Acc {acc:.0f}%, AvgT: {format_time(avg_t, short=True)}")
    ta = processed_data["time_analysis"]
    summary_parts.append(f"\n--- Time Analysis ---\nAvgT/CorrectQ: {format_time(ta['time_per_correct_q_sec'], short=True)}, AvgT/IncorrectQ: {format_time(ta['time_per_incorrect_q_sec'], short=True)}")
    if ta["questions_over_avg_time_incorrect"]:
        summary_parts.append(f"  Found {len(ta['questions_over_avg_time_incorrect'])} incorrect Qs taking >1.5x avg time.")
        if ta["questions_over_avg_time_incorrect"]: q_ex = ta["questions_over_avg_time_incorrect"][0]; summary_parts.append(f"    e.g., '{q_ex['text'][:30]}...' ({q_ex['subject']}/{q_ex['level']}) took {q_ex['time_taken']}.")
    return "\n".join(summary_parts)

# ==============================================================================
# ===== REVISED & PROFESSIONAL LLM FEEDBACK FUNCTION ===========================
# ==============================================================================

def get_llm_feedback(context_summary, processed_data, student_name, gemini_key, groq_key, llm_provider_choice, model_name=None):
    """
    Generates professional, data-driven feedback with a formal coaching analyst persona.
    """
    # STEP 1: Define the prompt as a TEMPLATE with a new professional persona.
    prompt_template = """
You are an Expert Performance Analyst from a leading coaching institute for competitive exams (JEE/NEET). Your role is to provide a precise, data-driven analysis of a student's performance.
Your tone must be:
- **Professional & Authoritative:** You are an expert providing a formal analysis.
- **Data-Centric:** Every insight must be tied directly to the performance data provided.
- **Encouraging & Constructive:** The goal is to empower the student with clarity and a path for improvement, not to criticize.
- **Action-Oriented:** The feedback must conclude with clear, strategic recommendations.

**Core Analytical Principles:**
- **Conceptual Clarity:** The foundation of all high scores. Gaps here must be prioritized.
- **Application Skill:** The ability to apply concepts correctly under time pressure.
- **Strategic Time Management:** Differentiating between questions to attempt, questions to leave, and managing time per question is a critical skill.
- **Error Analysis:** Categorizing mistakes (e.g., conceptual errors vs. calculation errors vs. time pressure errors) is key to targeted improvement.

Here is the raw performance data summary:
---
{context_summary}
---
Now, generate a formal Performance Analysis Report for **{student_name}**. The report must follow this exact structure and tone, using Markdown for formatting.

## 🌟 Performance Analysis: {test_name}
(Begin with a formal opening. State the purpose of the report: to analyze the performance in the recent test and identify strategic areas for improvement. Acknowledge the student's score as a data point for analysis.)

## 📊 Overall Performance Metrics
(Present the key metrics clearly.
- **Score:** {score_string} ({percentage_score:.1f}%)
- **Accuracy (Attempted):** {accuracy_on_attempted:.1f}% ({correct_questions} out of {attempted_questions} correct)
- **Attempt Rate:** {attempted_questions} out of {total_questions} questions attempted.
Follow this with a concise analytical statement. E.g., "This score indicates a foundational understanding, but there are clear opportunities to improve conversion of knowledge into marks by focusing on accuracy and strategic attempts.")

## 📚 Subject-wise Performance Breakdown
(Provide a brief, analytical summary for each subject based on the data.
- **Example (Strong):** "In **Physics**, performance was strong with a high accuracy rate. The primary focus here should be on optimizing time to solve questions even faster."
- **Example (Weak):** "**Chemistry** presents a significant opportunity for improvement. The lower accuracy suggests a need to revisit core concepts and strengthen problem-solving application skills.")

## 🎯 Diagnostic Analysis: Conceptual Gaps & Strategic Errors
(This is the core diagnostic section. Use insights from the visualizations (Scatter Plot, Treemap).
- **Conceptual Gaps:** "The analysis identifies specific chapters, such as **'{weak_chapter_example}'**, as areas of conceptual weakness. This is evidenced by a high number of incorrect answers in this chapter."
- **Strategic Errors:** "The Time vs. Outcome analysis reveals two critical patterns:
    1.  **Time Traps:** A number of questions were attempted that consumed significant time but resulted in incorrect answers. This indicates difficulty in recognizing challenging problems early and moving on.
    2.  **Silly Mistakes:** Several questions were answered incorrectly in a very short amount of time, suggesting a potential lack of attention to detail or rushing through questions that should have been scoring opportunities.")

## ✨ Strategic Recommendations for Improvement
(Provide 3 clear, prioritized, and actionable directives. Frame them as professional recommendations.
1.  **Recommendation 1 (Addressing Conceptual Gaps):** "Prioritize the chapter '{weak_chapter_example}'. The recommended approach is to first review this chapter from your core study material or coaching modules, followed by solving a targeted set of 30-40 problems, focusing on variety and difficulty levels."
2.  **Recommendation 2 (Improving Test-Taking Strategy):** "Implement a 'Two-Pass' strategy in your next mock test. In the first pass (approx. 50-60% of the total time), solve only the questions you are confident you can answer correctly and quickly. Mark others for the second pass. This strategy is designed to maximize your score from your current knowledge base and build momentum."
3.  **Recommendation 3 (Reducing Errors):** "To reduce errors on easy-to-medium questions, dedicate 5-10 minutes at the end of each section to review your marked answers. This simple check can help catch calculation errors and misinterpretations of questions.")

## 🌱 Concluding Summary
(Conclude with a professional, forward-looking statement. Reiterate that this analysis is a tool for focused effort. "This report provides a clear, data-driven path forward. Consistent effort applied to these specific areas of improvement will lead to measurable progress in subsequent tests. We are confident in your ability to leverage these insights for a stronger performance.")
"""
    try:
        # STEP 2: Define the variables needed for the template before formatting.
        op = processed_data["overall_performance"]
        ti = processed_data["test_info"]

        # Find an example of a weak chapter to make the recommendation more concrete
        weak_chapter_example = "a key chapter" # Default
        if processed_data.get("chapter_performance"):
            all_chapters = []
            for subject, chapters in processed_data["chapter_performance"].items():
                for chapter, data in chapters.items():
                    if data['incorrect'] > 0:
                        all_chapters.append({'name': chapter, 'incorrect': data['incorrect']})
            if all_chapters:
                # Sort chapters by most incorrect answers to find the weakest one
                all_chapters.sort(key=lambda x: x['incorrect'], reverse=True)
                weak_chapter_example = all_chapters[0]['name']

        # STEP 3: Use the .format() method to safely create the final prompt.
        final_prompt = prompt_template.format(
            student_name=student_name,
            test_name=ti['name'],
            context_summary=context_summary,
            score_string=f"{op['marks_scored']:.0f} / {ti['total_marks']}",
            percentage_score=op['percentage_score'],
            accuracy_on_attempted=op['accuracy_on_attempted'],
            correct_questions=op['correct'],
            attempted_questions=op['attempted'],
            total_questions=ti['total_questions'],
            weak_chapter_example=weak_chapter_example
        )

        # The rest of the function remains the same.
        if llm_provider_choice == "gemini" and gemini_key:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(model_name or "gemini-1.5-flash-latest")
            response = model.generate_content(final_prompt)
            return response.text
        elif llm_provider_choice == "groq" and groq_key:
            client = Groq(api_key=groq_key)
            completion = client.chat.completions.create(messages=[{"role": "user", "content": final_prompt}], model=model_name or "llama3-8b-8192")
            return completion.choices[0].message.content
        st.warning(f"LLM provider '{llm_provider_choice}' not properly configured. Falling back to template.")
        return generate_template_feedback(processed_data) # Fallback
    except Exception as e:
        st.error(f"Error during LLM API call: {e}")
        return generate_template_feedback(processed_data) # Fallback

def generate_template_feedback(processed_data):
    op, ti, student_name = processed_data["overall_performance"], processed_data["test_info"], processed_data["student_info"]["name"]
    greeting = f"Dear {student_name},"
    intro = f"{greeting}\n\nWell done for taking the {ti['name']}! Every test is a learning opportunity. You scored {op['marks_scored']:.0f}/{ti['total_marks']}. Let's break this down."
    if op['percentage_score'] >= 75: intro = f"{greeting}\n\nFantastic work on {ti['name']}! Scoring {op['marks_scored']:.0f}/{ti['total_marks']} is excellent. Your dedication shows!"
    elif op['percentage_score'] >= 50: intro = f"{greeting}\n\nGood job on {ti['name']}, scoring {op['marks_scored']:.0f}/{ti['total_marks']}! Solid performance. This report helps refine and build strengths."
    overall_sum = f"Overall: {op['attempted']}/{ti['total_questions']} attempted, {op['correct']} correct. Accuracy on attempted: {op['accuracy_on_attempted']:.1f}%."
    s_strengths, s_focus = ["- Subject data limited."], ["- Review all subjects."]
    if processed_data["subjects_performance"]:
        sorted_s = sorted(processed_data["subjects_performance"], key=lambda x: x["accuracy_on_attempted"], reverse=True)
        if sorted_s:
            s_strengths = [f"- **{sorted_s[0]['name']}** ({sorted_s[0]['accuracy_on_attempted']:.1f}% acc.) is a strong point."]
            if len(sorted_s) > 1: s_focus = [f"- Focus on **{sorted_s[-1]['name']}** ({sorted_s[-1]['accuracy_on_attempted']:.1f}% acc.) for gains."]
    suggestions = f"### ✨ Actionable Suggestions\n1. **Master Concepts:** Revisit topics from {s_focus[0].split('**')[1] if '**' in s_focus[0] else 'weaker areas'}.\n2. **Strategic Practice:** For low accuracy chapters, solve easy/medium Qs first.\n3. **Time Allocation:** Practice timed mocks. Allocate time per Q by difficulty."
    conclusion = "Consistent effort & smart strategies are key. Use this feedback, stay positive, keep pushing. You've got this!"
    return f"## 🌟 Personalized Introduction\n{intro}\n\n## 📊 Overall Performance Snapshot\n{overall_sum}\n\n## 📚 Subject Highlights\n**Strengths:**\n{s_strengths[0]}\n\n**Areas for Focus:**\n{s_focus[0]}\n\n{suggestions}\n\n## 🌱 Concluding Motivational Message\n{conclusion}"

# --- PDF Generation Functions (Unchanged) ---
def create_pdf_styles():
    styles = getSampleStyleSheet()
    base_font, base_font_bold = 'Helvetica', 'Helvetica-Bold'
    styles['Heading1'].fontSize, styles['Heading1'].spaceBefore, styles['Heading1'].spaceAfter, styles['Heading1'].textColor, styles['Heading1'].fontName, styles['Heading1'].leading = 16, 0.6*cm, 0.3*cm, colors.HexColor("#E63946"), base_font_bold, 18
    styles['Heading2'].fontSize, styles['Heading2'].spaceBefore, styles['Heading2'].spaceAfter, styles['Heading2'].textColor, styles['Heading2'].fontName, styles['Heading2'].leading = 13, 0.4*cm, 0.2*cm, colors.HexColor("#1D3557"), base_font_bold, 15
    styles['BodyText'].fontSize, styles['BodyText'].alignment, styles['BodyText'].spaceAfter, styles['BodyText'].leading, styles['BodyText'].fontName = 10, TA_JUSTIFY, 0.2*cm, 14, base_font
    styles.add(ParagraphStyle(name='MainTitle', fontSize=20, alignment=TA_CENTER, spaceAfter=0.6*cm, textColor=colors.HexColor("#1D3557"), fontName=base_font_bold))
    styles.add(ParagraphStyle(name='SubTitle', fontSize=12, alignment=TA_CENTER, spaceAfter=0.2*cm, textColor=colors.HexColor("#457B9D"), fontName=base_font))
    styles.add(ParagraphStyle(name='CustomBullet', parent=styles['BodyText'], firstLineIndent=0, leftIndent=1*cm, bulletIndent=0.5*cm, spaceBefore=0.1*cm))
    styles.add(ParagraphStyle(name='TableHeader', fontSize=9, alignment=TA_CENTER, textColor=colors.whitesmoke, fontName=base_font_bold, leading=11))
    styles.add(ParagraphStyle(name='TableCell', fontSize=9, alignment=TA_CENTER, fontName=base_font, leading=11))
    styles.add(ParagraphStyle(name='TableCellLeft', parent=styles['TableCell'], alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='SmallText', fontSize=8, alignment=TA_LEFT, textColor=colors.darkgrey, fontName=base_font))
    styles.add(ParagraphStyle(name='SyllabusText', fontSize=9, fontName='Courier', alignment=TA_LEFT, leading=12, spaceAfter=0.1*cm))
    return styles

def build_pdf_story(processed_data, llm_feedback_md, chart_paths, styles):
    story = []
    story.append(Paragraph(f"Performance Analysis Report", styles['MainTitle']))
    story.append(Paragraph(f"Test: {processed_data['test_info']['name']}", styles['SubTitle']))
    story.append(Paragraph(f"Student: {processed_data['student_info']['name']} | Date: {processed_data['test_info']['date']}", styles['SubTitle']))
    story.append(Spacer(1, 0.5*cm))

    img_overall = Paragraph("(Overall Score Chart N/A)", styles['SmallText'])
    if "overall_score" in chart_paths and chart_paths["overall_score"] and os.path.exists(chart_paths["overall_score"]):
        try: img_overall = Image(chart_paths["overall_score"], width=2.5*inch, height=2.5*inch)
        except Exception as e:
            if 'st' in globals(): st.warning(f"PDF Chart Error (Overall Score Image Load): {e}")
            else: print(f"PDF Chart Error (Overall Score Image Load): {e}")
            img_overall = Paragraph("(Chart unavailable)", styles['SmallText'])
    
    op, ti, ta = processed_data["overall_performance"], processed_data["test_info"], processed_data["time_analysis"]
    key_metrics_data = [
        [Paragraph("<b>Metric</b>", styles['TableCellLeft']), Paragraph("<b>Value</b>", styles['TableCell'])],
        [Paragraph("Total Score:", styles['TableCellLeft']), Paragraph(f"{op['marks_scored']:.0f} / {ti['total_marks']}", styles['TableCell'])],
        [Paragraph("Percentage:", styles['TableCellLeft']), Paragraph(f"{op['percentage_score']:.1f}%", styles['TableCell'])],
        [Paragraph("Accuracy (Att.):", styles['TableCellLeft']), Paragraph(f"{op['accuracy_on_attempted']:.1f}%", styles['TableCell'])],
        [Paragraph("Time Taken:", styles['TableCellLeft']), Paragraph(format_time(op['time_taken_sec']), styles['TableCell'])],
        [Paragraph("Efficiency:", styles['TableCellLeft']), Paragraph(f"{ta['time_efficiency_percentage']:.1f}%", styles['TableCell'])]
    ]
    key_metrics_table = Table(key_metrics_data, colWidths=[1.7*inch, 1.3*inch])
    key_metrics_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#A8DADC")), ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#1D3557")), ('ALIGN', (0,0), (-1,-1), 'LEFT'), ('ALIGN', (1,0), (1,-1), 'RIGHT'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('LEFTPADDING', (0,0), (-1,-1), 3), ('RIGHTPADDING', (0,0), (-1,-1), 3)]))
    summary_layout = Table([[img_overall, key_metrics_table]], colWidths=[3*inch, 3.2*inch])
    summary_layout.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')])); story.append(summary_layout)
    story.append(Spacer(1, 0.5*cm))
    
    # --- PDF Markdown Parsing ---
    list_items = []
    # Use a custom bullet style that has less space before it
    styles.add(ParagraphStyle(name='PdfBullet', parent=styles['CustomBullet'], spaceBefore=1, leftIndent=20))
    def flush_list(story_list, items_list, style_obj):
        if items_list:
            para_items = [Paragraph(re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', item), style_obj['PdfBullet']) for item in items_list]
            list_flow = ListFlowable(para_items, bulletType='bullet', start='bulletchar', bulletFontSize=10, leftIndent=5)
            story_list.append(list_flow)
            items_list.clear()

    for line in llm_feedback_md.split('\n'):
        s_line = line.strip()
        if s_line.startswith('## '): flush_list(story, list_items, styles); story.append(Paragraph(s_line[3:], styles['Heading1']))
        elif s_line.startswith('### '): flush_list(story, list_items, styles); story.append(Paragraph(s_line[4:], styles['Heading2']))
        elif s_line.startswith(('* ', '- ')): list_items.append(s_line[2:])
        else:
            flush_list(story, list_items, styles)
            if s_line: story.append(Paragraph(re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', s_line), styles['BodyText']))
    flush_list(story, list_items, styles); story.append(Spacer(1, 0.5*cm))
    # --- End PDF Markdown Parsing ---
    
    story.append(PageBreak()); story.append(Paragraph("Subject-wise Performance Analysis", styles['Heading1']))
    if "subject_marks" in chart_paths and chart_paths["subject_marks"] and os.path.exists(chart_paths["subject_marks"]):
        try:
            img_sm = Image(chart_paths["subject_marks"], width=6.5*inch, height=3.5*inch); img_sm.hAlign='CENTER'
            story.append(Spacer(1,0.2*cm)); story.append(img_sm); story.append(Spacer(1,0.3*cm))
        except Exception as e: story.append(Paragraph(f"(Chart Error: {e})", styles['SmallText']))
    else: story.append(Paragraph("(Subject Marks Chart data missing or file not found)", styles['SmallText']))

    if "subject_accuracy" in chart_paths and chart_paths["subject_accuracy"] and os.path.exists(chart_paths["subject_accuracy"]):
        try:
            img_sa = Image(chart_paths["subject_accuracy"], width=6.5*inch, height=3.5*inch); img_sa.hAlign='CENTER'
            story.append(img_sa); story.append(Spacer(1,0.5*cm))
        except Exception as e: story.append(Paragraph(f"(Chart Error: {e})", styles['SmallText']))
    else: story.append(Paragraph("(Subject Accuracy Chart data missing or file not found)", styles['SmallText']))
        
    subj_tbl_data = [[Paragraph(h, styles['TableHeader']) for h in ["Subject", "Score (/Pot.)", "Att.", "Corr.", "Acc%(Att)", "Time"]]]
    for sp in processed_data["subjects_performance"]:
        subj_tbl_data.append([ Paragraph(sp['name'], styles['TableCellLeft']), Paragraph(f"{sp['marks_scored']:.0f}/{sp['total_marks_possible']:.0f}", styles['TableCell']), Paragraph(str(sp['attempted']), styles['TableCell']), Paragraph(str(sp['correct']), styles['TableCell']), Paragraph(f"{sp['accuracy_on_attempted']:.1f}", styles['TableCell']), Paragraph(format_time(sp['time_taken_sec'], short=True), styles['TableCell']) ])
    subj_table = Table(subj_tbl_data, colWidths=[1.8*inch,1.2*inch,0.6*inch,0.6*inch,1.1*inch,0.9*inch])
    subj_table.setStyle(TableStyle([ ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#A8DADC")),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor("#1D3557")), ('ALIGN',(0,0),(-1,-1),'CENTER'),('ALIGN',(0,1),(0,-1),'LEFT'), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('GRID',(0,0),(-1,-1),0.5,colors.darkgrey),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke,colors.HexColor("#F1FAEE")]) ]))
    story.append(subj_table); story.append(Spacer(1,0.5*cm))

    if processed_data["test_info"]["syllabus_parsed"].strip() != "Syllabus not provided.":
        story.append(PageBreak()); story.append(Paragraph("Test Syllabus Overview", styles['Heading1']))
        for line in processed_data["test_info"]["syllabus_parsed"].split('\n'):
            if line.strip():
                if line.startswith("  • "): story.append(Paragraph(f"• {line[4:]}", styles['SyllabusText'], bulletText='•'))
                elif ":" in line and not line.startswith(" "): story.append(Paragraph(line, styles['Heading2']))
                else: story.append(Paragraph(line, styles['SyllabusText']))
    return story

def generate_pdf_report_reportlab(processed_data, llm_feedback_md, chart_paths, output_buffer):
    doc = SimpleDocTemplate(output_buffer, pagesize=A4, rightMargin=0.75*inch, leftMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch, title=f"Report - {processed_data['test_info']['name']}", author="Performance Analytics System")
    styles = create_pdf_styles()
    def header_footer(canvas, doc_obj): canvas.saveState(); canvas.setFont('Helvetica', 9); canvas.drawString(doc_obj.leftMargin, 0.5*inch, f"Page {doc_obj.page} | {processed_data['student_info']['name']} - {processed_data['test_info']['name']}"); canvas.restoreState()
    story_elements = build_pdf_story(processed_data, llm_feedback_md, chart_paths, styles)
    story_in_frames = [KeepInFrame(doc.width, doc.height, [el], mode='shrink') if isinstance(el, (Image, Table, ListFlowable)) else el for el in story_elements]
    try:
        doc.build(story_in_frames, onFirstPage=header_footer, onLaterPages=header_footer)
        return True
    except Exception as e:
        st.error(f"Error generating PDF with ReportLab: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Streamlit App UI ---
def main():
    st.set_page_config(page_title="🎓 Student Performance Analyzer", layout="wide", page_icon="📊")

    st.markdown("""
    <style>
        .main-header {font-size: 28px; color: #1E88E5; text-align: center; margin-bottom: 20px; font-weight: bold;}
        .sub-header {font-size: 20px; color: #424242; margin-top: 20px; margin-bottom:10px; border-bottom: 2px solid #1E88E5; padding-bottom: 5px;}
        .stMetric {background-color: #FFFFFF; border-radius: 5px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);}
        .stButton>button {background-color: #1E88E5; color: white; font-weight:bold; border-radius: 5px;}
        .stFileUploader label {font-size: 1.1em !important;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<p class='main-header'>🎓 IIT-JEE & NEET Performance Analyzer</p>", unsafe_allow_html=True)

    if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
    if 'processed_data' not in st.session_state: st.session_state.processed_data = None
    if 'chart_paths' not in st.session_state: st.session_state.chart_paths = {}
    if 'llm_feedback' not in st.session_state: st.session_state.llm_feedback = ""
    if 'pdf_buffer' not in st.session_state: st.session_state.pdf_buffer = None

    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader("Upload Performance JSON", type="json")

        st.markdown("---")
        st.subheader("🤖 AI Feedback (Optional)")
        enable_ai = st.checkbox("Enable AI-Powered Feedback", value=True)
        
        selected_llm_provider = "None"
        gemini_api_key, groq_api_key = "", ""

        if enable_ai:
            llm_options = ["Gemini", "Groq"]
            default_index = llm_options.index("Groq") if os.environ.get("GROQ_API_KEY") else (llm_options.index("Gemini") if os.environ.get("GEMINI_API_KEY") else 0)
            selected_llm_provider = st.selectbox("Choose LLM Provider", llm_options, index=default_index, help="Groq (Llama3) is recommended for speed and quality.")

            if selected_llm_provider == "Gemini": gemini_api_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY",""), help="Get from Google AI Studio")
            elif selected_llm_provider == "Groq": groq_api_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY",""), help="Get from console.groq.com")
        
        use_sample_data = st.button("📄 Use Sample Data", help="Loads a predefined sample JSON for demo.")

        if st.button("🚀 Analyze Performance", type="primary", disabled=(not uploaded_file and not use_sample_data)):
            st.session_state.analysis_done = False
            st.session_state.pdf_buffer = None

            raw_data_list = None
            if uploaded_file: raw_data_list = load_json_data(uploaded_file)
            elif use_sample_data:
                try:
                    with open('sample_submission_analysis_2.json', 'r', encoding='utf-8') as f_sample:
                        raw_data_list = json.load(f_sample)
                    st.info("Using sample_submission_analysis_2.json")
                except FileNotFoundError:
                    st.error("sample_submission_analysis_2.json not found. Please add it to the script's directory.")
                    st.stop()
                except Exception as e:
                    st.error(f"Error loading sample data: {e}"); st.stop()

            if raw_data_list:
                with st.spinner("🔬 Analyzing data, generating charts, and crafting feedback from your AI 'Bhaiya'... This might take a moment."):
                    st.session_state.processed_data = process_submission_data(raw_data_list)
                    if st.session_state.processed_data:
                        charts_dir = f"charts_rl_st_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        st.session_state.chart_paths = generate_charts(st.session_state.processed_data, output_dir=charts_dir)
                        
                        context_summary = prepare_llm_context_summary(st.session_state.processed_data)
                        st.session_state.llm_feedback = get_llm_feedback(
                            context_summary,
                            st.session_state.processed_data,
                            st.session_state.processed_data["student_info"]["name"], # Pass student name
                            gemini_api_key if enable_ai and selected_llm_provider == "Gemini" else "",
                            groq_api_key if enable_ai and selected_llm_provider == "Groq" else "",
                            selected_llm_provider.lower() if enable_ai else "none"
                        )
                        st.session_state.analysis_done = True
                        st.success("✅ Analysis Complete!")
                    else: st.error("Data processing failed. Please check the JSON structure.")
            else: st.warning("Please upload a JSON file or use sample data to start the analysis.")

    if st.session_state.analysis_done and st.session_state.processed_data:
        pd_data = st.session_state.processed_data
        op, ti, ta = pd_data["overall_performance"], pd_data["test_info"], pd_data["time_analysis"]

        st.markdown("<p class='sub-header'>🚀 Overall Performance Summary</p>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Score", f"{op['marks_scored']:.0f} / {ti['total_marks']}", f"{op['percentage_score']:.1f}%")
        col2.metric("Accuracy (Attempted)", f"{op['accuracy_on_attempted']:.1f}%", f"{op['correct']}/{op['attempted']} Correct")
        col3.metric("Time Taken", format_time(op['time_taken_sec']), f"Limit: {format_time(ti['total_time_allowed_min']*60)}")
        col4.metric("Time Efficiency", f"{ta['time_efficiency_percentage']:.1f}%")

        if st.session_state.chart_paths.get("overall_score"): st.image(st.session_state.chart_paths["overall_score"], caption="Overall Score Distribution", width=350)
        
        st.markdown("<p class='sub-header'>🧠 AI Mentor Feedback</p>", unsafe_allow_html=True)
        st.markdown(st.session_state.llm_feedback, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<p class='sub-header'>📄 Download Full Report</p>", unsafe_allow_html=True)
        
        if st.session_state.pdf_buffer is None:
            with st.spinner("Generating PDF report..."):
                pdf_buffer_io = io.BytesIO() 
                success = generate_pdf_report_reportlab(st.session_state.processed_data, st.session_state.llm_feedback, st.session_state.chart_paths, pdf_buffer_io)
                if success and pdf_buffer_io.getvalue():
                    st.session_state.pdf_buffer = pdf_buffer_io.getvalue()
                    st.success("PDF report is ready for download.")
                else:
                    st.session_state.pdf_buffer = None
                    if success: st.error("PDF generation resulted in an empty file.")

        if st.session_state.pdf_buffer:
            st.download_button(label="📥 Download PDF Report", data=st.session_state.pdf_buffer, file_name=f"Performance_Report_{ti['name'].replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")
        elif st.session_state.analysis_done and not st.session_state.pdf_buffer:
            st.warning("PDF report could not be generated. Please check for error messages above.")
        
        with st.expander("📊 View Subject Performance Charts & Details", expanded=False):
            st.markdown("<p class='sub-header'>📚 Subject-wise Performance</p>", unsafe_allow_html=True)
            if st.session_state.chart_paths.get("subject_marks"): st.image(st.session_state.chart_paths["subject_marks"], caption="Subject-wise Marks")
            if st.session_state.chart_paths.get("subject_accuracy"): st.image(st.session_state.chart_paths["subject_accuracy"], caption="Subject Accuracy (Attempted)")
            for sp_item in pd_data["subjects_performance"]: st.markdown(f"**{sp_item['name']}**: Score {sp_item['marks_scored']:.0f}/{sp_item['total_marks_possible']:.0f} | Accuracy (Att.): {sp_item['accuracy_on_attempted']:.1f}% | Time: {format_time(sp_item['time_taken_sec'])}")
        
        with st.expander("📜 View Test Syllabus", expanded=False):
            st.markdown("<p class='sub-header'>📝 Test Syllabus</p>", unsafe_allow_html=True); st.text(pd_data["test_info"]["syllabus_parsed"])

    elif not st.session_state.analysis_done:
        st.info("👋 Welcome! Please upload your performance JSON file or use the sample data, then click 'Analyze Performance' in the sidebar.")

if __name__ == "__main__":
    main()