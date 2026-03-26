class Personas:
    # --- INTERVIEWER PERSONAS ---
    FRIENDLY_HR = {
        "label": "Friendly HR Recruiter",
        "emoji": "🤝",
        "persona": "🤝 Friendly HR Recruiter (Focuses on soft skills & culture fit)",
        "meaning": "A standard, efficient first-round interview. Focus on high-level experience and culture fit.",
        "recommended_mode": "Standard Interview"
    }

    STRICT_TECH_LEAD = {
        "label": "Strict Technical Lead",
        "emoji": "💼",
        "persona": "💼 Strict Technical Lead (Focuses purely on accuracy & efficiency)",
        "meaning": "Common for technical assessments. Expect in-depth scrutiny and follow-ups.",
        "recommended_mode": "Technical / Complex"
    }

    STRESS_INTERVIEWER = {
        "label": "Stress Interviewer",
        "emoji": "🔥",
        "persona": "🔥 Stress Interviewer (Highly critical, looks for flaws & hesitations)",
        "meaning": "Advanced panel or final stage interview. High pressure.",
        "recommended_mode": "Technical / Complex"
    }

    # --- SYSTEM PROMPT MAPPINGS ---
    PERSONA_PROMPTS = {
        "Standard HR": "You are acting as a friendly HR Recruiter. Focus on soft skills, culture fit, and high-level behavioral examples. Be professional yet encouraging.",
        "Technical Lead": "You are acting as a strict Technical Lead. Focus on precision, technical architecture, and engineering trade-offs. Scrutinize the accuracy of technical claims.",
        "Stress Interviewer": "You are a ruthless, time-pressed panelist conducting a high-stakes interview. You offer ZERO pleasantries and zero encouragement. You heavily scrutinize logical gaps, cut the candidate off if they are too wordy, and aggressively challenge their confidence and technical claims. If they falter, press them harder. You are adversarial, highly critical, and impatient."
    }

    # --- COACHING PERSONAS ---
    AI_COACH = {
        "system_prompt": "You are an expert, direct, and highly constructive career coach.",
        "overall_impression_header": "### Overall Impression",
        "key_strengths_header": "### Key Strengths",
        "areas_for_improvement_header": "### Areas for Improvement",
        "star_framework_header": "### STAR Framework Analysis"
    }

    @staticmethod
    def get_interviewer_by_type(round_type, seniority):
        if "HR" in round_type or "Screen" in round_type or "First" in round_type:
            return Personas.FRIENDLY_HR
        elif "Technical" in round_type or "System" in round_type or "Code" in round_type:
            return Personas.STRICT_TECH_LEAD
        else:
            if seniority == "Executive":
                # Special case for executive presentation rounds
                p = Personas.STRESS_INTERVIEWER.copy()
                p["recommended_mode"] = "Presentation"
                return p
            return Personas.STRESS_INTERVIEWER

    @staticmethod
    def get_interview_sys_prompt(persona_name, round_name, seniority, job_title, industry):
        return (
            f"You are acting as a {persona_name} conducting a {round_name} interview "
            f"for a {seniority} {job_title} role in the {industry} industry. "
            f"Your goal is to assess the candidate's skills based on the persona. "
            f"Ask ONE question at a time. Keep your questions concise (1-2 sentences). "
            f"Base your follow-ups strictly on the candidate's previous answer. "
            f"Do not break character. Do not provide feedback yet."
        )

    @staticmethod
    def get_final_feedback_prompt(seniority, job_title, industry, full_transcript):
        return f"""
        You are a senior hiring manager. Review this interview transcript for a {seniority} {job_title} role in the {industry} industry.
        
        TRANSCRIPT:
        {full_transcript}
        
        Provide a comprehensive evaluation. Format your response strictly using these Markdown headers (Do NOT use numbered lists for the section titles):
        
        {Personas.AI_COACH['overall_impression_header']}
        (1 short paragraph)
        
        {Personas.AI_COACH['key_strengths_header']}
        (Use standard bullet points)
        
        {Personas.AI_COACH['areas_for_improvement_header']}
        (Use standard bullet points)
        
        {Personas.AI_COACH['star_framework_header']}
        (Short paragraph evaluating if they used Situation, Task, Action, Result)
        """
