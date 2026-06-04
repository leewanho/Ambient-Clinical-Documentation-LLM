import json
import os
import requests

# [확장] 15개 다빈도 임상 질환 파라미터 셋업
target_cases = [
    {"disease": "Acute Myocardial Infarction (급성 심근경색)", "patient": "David, 62-year-old male", "symptoms": "Crushing substernal chest pain radiating to left arm, diaphoresis, shortness of breath for 2 hours."},
    {"disease": "Cerebrovascular Accident (뇌졸중/뇌경색)", "patient": "Michael, 70-year-old male", "symptoms": "Sudden onset right-sided facial drooping, right arm weakness, slurred speech for 1 hour."},
    {"disease": "Bronchial Asthma (기관지 천식 발작)", "patient": "Chloe, 18-year-old female", "symptoms": "Severe wheezing, non-productive cough, chest tightness, dyspnea triggered by cold air for 6 hours."},
    {"disease": "Chronic Obstructive Pulmonary Disease (COPD 급성 악화)", "patient": "Gary, 67-year-old male", "symptoms": "Worsening chronic cough, increased purulent sputum production, severe shortness of breath at rest for 3 days."},
    {"disease": "Acute Pyelonephritis (급성 신우신염)", "patient": "Hannah, 29-year-old female", "symptoms": "High fever, chills, severe right-sided flank pain, dysuria, frequency for 36 hours."},
    {"disease": "Deep Vein Thrombosis (심부정맥 혈전증)", "patient": "Kevin, 45-year-old male", "symptoms": "Unilateral left calf swelling, warmth, redness, throbbing pain increased with walking for 5 days."},
    {"disease": "Anaphylaxis (아나필락시스 쇼크)", "patient": "Ryan, 15-year-old male", "symptoms": "Generalized urticaria, lip and tongue swelling, inspiratory stridor, hypotension after peanut ingestion 30 minutes ago."},
    {"disease": "Gouty Arthritis (급성 통풍성 관절염)", "patient": "Mark, 51-year-old male", "symptoms": "Sudden excruciating pain, swelling, erythema, warmth in the right first metatarsophalangeal joint for 12 hours."},
    {"disease": "Hyperthyroidism (갑상선 기능 항진증)", "patient": "Lisa, 34-year-old female", "symptoms": "Palpitations, heat intolerance, weight loss of 5kg despite increased appetite, hand tremors for 2 months."},
    {"disease": "Acute Cholecystitis (급성 담낭염)", "patient": "Brian, 42-year-old male", "symptoms": "Severe right upper quadrant abdominal pain radiating to right shoulder, nausea, postprandial episodes for 18 hours."},
    {"disease": "Panic Disorder (공황 장애)", "patient": "Jessica, 28-year-old female", "symptoms": "Sudden unprovoked episodes of racing heart, smothering sensation, intense fear of dying lasting 20 minutes."},
    {"disease": "Rheumatoid Arthritis (류마티스 관절염)", "patient": "Nancy, 47-year-old female", "symptoms": "Symmetrical bilateral hand and wrist joint pain, stiffness lasting over 1 hour in the morning for 6 months."},
    {"disease": "Peptic Ulcer Disease (위궤양)", "patient": "Jason, 39-year-old male", "symptoms": "Burning epigastric pain partially relieved by food or antacids, occasional bloating for 4 weeks."},
    {"disease": "Atrial Fibrillation (심방세동)", "patient": "Donald, 73-year-old male", "symptoms": "Irregularly irregular heart palpitations, mild fatigue, lightheadedness during normal daily activities for 1 week."},
    {"disease": "Migraine Headache (편두통 악화)", "patient": "Sarah, 33-year-old female", "symptoms": "Severe throbbing unilateral headache, photophobia, phonophobia, preceded by visual aura for 8 hours."}
]

jsonl_file = r"D:\ko-medscribe-llm\ko-medscribe-llm\notebooks\train_llm.jsonl"
system_prompt = "You are a medical writer. Generate a realistic doctor-patient conversation and a matching clinical SOAP note. Your output must be a single, strict JSON object with EXACTLY two keys: 'dialogue' and 'note'. Do not use any other key names."
ollama_url = "http://localhost:11434/api/chat"

print("="*60)
print(f"START: BIG DATA AUGMENTATION PROCESS (Total: {len(target_cases)} cases)")
print("="*60)

for i, case in enumerate(target_cases):
    print(f"[{i+1}/{len(target_cases)}] Generating: {case['disease']}...")
    user_instruction = f"Generate a medical case based on these constraints:\n- Patient: {case['patient']}\n- Condition: {case['disease']}\n- Key Symptoms: {case['symptoms']}\n\nCRITICAL: The JSON keys must be exactly 'dialogue' and 'note'."
    payload = {"model": "llama3", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_instruction}], "stream": False, "format": "json", "options": {"temperature": 0.8}}
    
    try:
        response = requests.post(ollama_url, json=payload, timeout=120)
        if response.status_code == 200:
            parsed_json = json.loads(response.json()['message']['content'])
            dialogue_text = parsed_json.get('dialogue' if 'dialogue' in parsed_json else 'Dialogue', '')
            note_key = next((k for k in ['note', 'Note', 'clinical_note', 'soap_note'] if k in parsed_json), None)
            note_text = parsed_json.get(note_key) if note_key else None
            
            if not dialogue_text or not note_text: raise KeyError("Required keys missing.")
            
            structured_data = {
                "messages": [
                    {"role": "system", "content": "You are an expert clinical AI assistant."},
                    {"role": "user", "content": f"Conversation:\n{dialogue_text}\n\nGenerate Clinical Note:"},
                    {"role": "assistant", "content": note_text}
                ]
            }
            with open(jsonl_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(structured_data, ensure_ascii=False) + "\n")
            print(f"  -> SUCCESS: {case['disease']} saved.")
        else:
            print(f"  -> OLLAMA ERROR: {response.status_code}")
    except Exception as e:
        print(f"  -> BYPASSED ERROR: {e}")

print("\n" + "="*60)
print("ALL BATCH PROCESS COMPLETE")
print("="*60)