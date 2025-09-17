JUDGE_TBQA_PROMPT = """
You are a medical assistant. 
Your task is to determine whether a given medical exam question is related to tuberculosis (TB). 

Background: 
- Tuberculosis (TB) is an acute or chronic bacterial infection caused by Mycobacterium tuberculosis, mainly affecting the lungs. 

Instructions: 
- Read the following medical exam question: {message}
- If the question is related to tuberculosis, answer "Yes".
- If the question is not related to tuberculosis, answer "No".

Output format (strictly follow this JSON format without extra explanation):
{"Answer":"Yes"} or {"Answer":"No"}
"""


JUDGE_LEPROSYQA_PROMPT = """
You are a medical assistant. 
Your task is to determine whether a given medical exam question is related to leprosy. 

Background: 
- Leprosy, also known as Hansen’s disease, is a chronic infectious disease caused by the bacteria Mycobacterium leprae. 

Instructions: 
- Read the following medical exam question: {message}
- If the question is related to leprosy, answer "Yes".
- If the question is not related to leprosy, answer "No".

Output format (strictly follow this JSON format without extra explanation):
{"Answer":"Yes"} or {"Answer":"No"}
"""

JUDGE_MALARIAQA_PROMPT = """
You are a medical assistant. 
Your task is to determine whether a given medical exam question is related to Malaria. 

Background: 
- Malaria is a communicable parasitic disorder spread through the bite of the Anopheles mosquito. 

Instructions: 
- Read the following medical exam question: {message}
- If the question is related to Malaria, answer "Yes".
- If the question is not related to Malaria, answer "No".

Output format (strictly follow this JSON format without extra explanation):
{"Answer":"Yes"} or {"Answer":"No"}
"""

EXTRACT_BIOMED_RELATIONS_PROMPT = """
You are a medical information extraction assistant. Your task is to extract relationships between entities in a biomedical abstract.

Instructions:
1. Read the following abstract: {abstract}
2. For each entity in {entities}, identify relationships with other entities mentioned in the abstract.
3. Choose a relation from the following list if appropriate:
   (caused, interacts, symptoms, resembles, biomarker, diagnosis, associates, binds, treats, palliates, Age of onset, Onset site, examine)
4. If none of the above relations fit, create a new relation that accurately describes the connection between the two entities.
5. Output all extracted triples strictly in the following Python list format:
  [(Leprosy, synonyms, Hansen's Disease), (Tuberculosis, caused, Mycobacterium tuberculosis)]

6. Each entity should appear in the format exactly as in {entities} and maintain consistency in naming.
7. Do not include any explanations, text, or comments—only the Python list of triples.

"""

ANALYZE_MEDICAL_QUESTION_PROMPT = """
You are a helpful medical assistant. Your task is to help candidates answer the following medical exam question.

Question: {message}

Instructions:
1. Decompose the key medical knowledge points involved in the question.
2. Analyze each option using the latest clinical guidelines and authoritative textbooks.
3. For each option, provide a clear analysis of why it is correct or incorrect.
4. Output strictly in the following JSON format without extra explanations:
{
  "Analysis": {
    "A": "Analysis of option A",
    "B": "Analysis of option B",
    ...
  },
  "Answer": "Correct answer option"
}
"""

EXTRACT_MEDICAL_ENTITIES_FROM_TEXT_PROMPT = """
You are a medical information extraction assistant.

Task:
- Extract all medical entities from the following biomedical text.

Text: {answer_llM_analyse}

Instructions:
1. Identify all relevant medical entities mentioned in the text.
2. Output the entities strictly as a JSON array of strings, without any extra explanations.
3. Each entity should appear only once in the array, even if mentioned multiple times.
4. Format example:
   Input: "Gluconeogenesis refers to the process of converting non-sugar substances into glucose. This process involves glucose metabolism."
   Output: ["Gluconeogenesis", "non-sugar substances", "glucose", "glucose metabolism"]
"""

ANSWER_MEDICAL_EXAM_QUESTION_PROMPT = """
You are a medical exam question answering assistant.

Task:
- Determine whether the retrieved knowledge graph triplets, combined with your own medical expertise, are sufficient to answer the given medical exam question.
- If sufficient, provide the correct answer with option-by-option analysis.
- If insufficient, explain why.

Question: {question}
Knowledge Triplets: {cluster_chain_of_entities}

Instructions:
1. Identify and decompose the key medical knowledge points in the question.
2. Analyze the question by integrating your own medical expertise (based on authoritative clinical guidelines and standard textbooks) with the retrieved knowledge graph triplets.
3. If the knowledge is sufficient:
   - Provide the correct answer.
   - Give a detailed analysis of each option (A, B, C, D...).
   - Output strictly in the following JSON format:
     {{
       "Yes": {{
         "A": "Analysis of option A",
         "B": "Analysis of option B",
         "C": "Analysis of option C",
         "D": "Analysis of option D"
       }},
       "Answer": "Correct option"
     }}
4. If the knowledge is insufficient:
   - Output strictly in the following JSON format:
     {{
       "No": "reason"
     }}
"""

