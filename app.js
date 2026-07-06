// app.js - Enhanced Medical Appointment System with LLM Information Extraction
// Uses LLM for intelligent patient information extraction
const DEFAULT_COUNTRY_CODE = '+91';
// Patient Data Class
class PatientData {
    constructor() {
        this.name = "";
        this.age = 0;
        this.gender = "";
        this.contact = "";
        this.symptoms = "";
        this.preferredDoctor = "N/A";
        this.preferredTime = null;
        this.detailedAssessmentDone = false;
        this.extractionHistory = []; // Track what's been extracted
        this.diagnosis = "";           // AI-generated diagnosis
        this.diagnosisConfidence = ""; // low / medium / high
        this.differential = [];        // running differential: [{condition, probability, evidence_for, evidence_against, discriminating_question}]
        this.askedQuestions = [];      // questions the AI has already asked (to prevent repetition)
        this.symptomsSummary = '';     // running summary: initial complaint + all follow-up answers
    }

    hasBasicInfo() {
        // contact and diagnosis are best-effort — don't block booking if AI skipped them
        return !!(this.name && this.age && this.gender && this.symptoms);
    }

    isComplete() {
        return this.hasBasicInfo() && this.detailedAssessmentDone;
    }

    getExtractedFields() {
        const fields = [];
        if (this.name) fields.push('name');
        if (this.age) fields.push('age');
        if (this.gender) fields.push('gender');
        if (this.contact) fields.push('contact');
        if (this.symptoms) fields.push('symptoms');
        if (this.diagnosis) fields.push('diagnosis');
        if (this.preferredDoctor && this.preferredDoctor !== "N/A") fields.push('preferredDoctor');
        return fields;
    }

    getMissingFields() {
        const required = ['name', 'age', 'gender', 'symptoms'];
        const extracted = this.getExtractedFields();
        return required.filter(field => !extracted.includes(field));
    }
}

// Enhanced LLM Interface with Information Extraction
class LLMPatientInterface {
    constructor(provider, apiKey = null) {
        this.provider = provider;
        this.apiKey = apiKey;
        this.conversationHistory = [];
        this.patientData = new PatientData();
        this.followUpQuestionsAsked = 0;
        this.maxFollowUpQuestions = 10;
        this.baseUrl = window.location.origin;

        this.systemPrompt = `You are a medical assistant who helps create appointments for patients at a hospital.

You MUST collect information in this sequence:

STEP 1 — Collect name, age, and gender. Ask ONLY for whichever of these are still missing, grouped into one question. Never ask again for something already collected. If all three are unknown, ask together: "Could you please share your name, age, and gender?" If only age and gender are missing, ask: "Could you share your age and gender?" If only age is missing, ask: "How old are you?" And so on.

STEP 1b — Once name/age/gender are known, ask for phone number (if not yet known).

STEP 1c — Once phone is known, ask: "What brings you in today?" (if symptoms not yet known).

STEP 1d — After symptoms are known, ask: "What time of day works best — morning, afternoon, or evening?" (optional, only once).

Do NOT ask about symptoms until you have name, age, gender, and phone number.
Do NOT ask follow-up symptom questions until STEP 1 is fully complete.

STEP 2 — Once ALL of name, age, gender, phone, symptoms, and preferred time are known, ask up to 5 relevant follow-up questions about the patient's ACTUAL stated symptoms only.

STEP 3 — After STEP 2 (or reaching 10 total questions), provide ONLY this structured assessment:
   Diagnosis: [clinical condition name]
   Confidence: [Low / Medium / High]
   Reason: [one sentence based on the symptoms and answers]
   I have sufficient information to book your appointment now.

   The diagnosis MUST be a clinical condition (e.g. "Possible myocardial infarction", "Likely acute gastritis", "Suspected lumbar disc herniation") — NOT a restatement of symptoms.
   NEVER include appointment date, time, doctor name, hospital, or placeholder text like [Date] or [Location].
   The booking system will handle all appointment details automatically after you say the above.

STRICT RULES:
- NEVER ask multiple questions EXCEPT name+age+gender which must be asked together in STEP 1.
- NEVER invent or assume symptoms. Only discuss what the patient has told you.
- Do NOT ask for information you already have.
- Do NOT repeat questions already asked.
- If the patient says "I don't know", acknowledge and move on.
- Ask follow-up questions about the patient's ACTUAL symptoms ONLY — never introduce conditions or symptoms they did not mention.
- Ask a MAXIMUM of 10 questions total. Stop and wrap up after 10 questions.
- This is an ONLINE appointment booking system. NEVER suggest physical exams, lab tests, blood tests, urine tests, or any in-person procedures.
- GENDER RULES: If the patient is MALE, NEVER ask about periods, menstruation, pregnancy, or any female-specific conditions. If the patient is FEMALE, NEVER ask about prostate issues or male-specific conditions.`;

        this.extractionPrompt = `You are an expert at extracting patient information from conversations. Start your work only after the conversation with the patient is complete.
Analyze the conversation and extract the following information if present:

REQUIRED FIELDS:
- name: Patient's full name
- age: Patient's age (number only)
- gender: Male/Female/Other
- contact: phone number
- symptoms: The patient's reported symptoms and complaints (what they feel)
- diagnosis: A CLINICAL DIAGNOSIS — a medical condition name, NOT a restatement of symptoms.
  Examples of valid diagnosis: "Possible myocardial infarction", "Likely acute gastritis", "Suspected lumbar disc herniation", "Probable anxiety disorder".
  Examples of INVALID diagnosis (reject these): "chest pain", "difficulty breathing", "stomach ache" — these are symptoms, not diagnoses.
  ONLY extract diagnosis after at least 3 follow-up questions have been asked. If insufficient info, return null.
- diagnosisConfidence: The confidence level for the diagnosis — MUST be exactly one of: "Low", "Medium", or "High".
  Base this on how many relevant symptoms and follow-up answers were gathered:
  Low = 1-2 data points, Medium = 3-4 data points, High = 5+ data points with consistent picture.
  Return null if no diagnosis yet.

OPTIONAL FIELDS:
- preferredDoctor: Specific doctor they want to see (if mentioned)
- preferredTime: Preferred appointment time — one of: "morning", "afternoon", "evening" (if mentioned)

RULES:
1. Only extract information that is clearly stated
2. For age, extract only numbers (e.g., "25" not "25 years old")
3. For gender, standardize to: Male, Female, or Other
4. For contact, prefer phone 
5. For symptoms, capture the main health concerns in their own words
6. If information is unclear or not provided, ask for it again. DO NOT MARK AS EXTRACTED UNLESS YOU GET A VALID INFORMATION FOR THE FIELD.
7. Be conservative - don't guess or infer information
8. Form an intelligent diagnosis from the patient conversation about the follow up questions.
9. Do not complete assessment before all the information has been extracted. None should be null (EXCEPT PREFERRED DOCTOR)


Return a JSON object with the extracted information.

`;
    }

    async callLLM(message) {
        if (this.provider === 'gemini') {
            return await this.callGeminiBackend(message);
        } else {
            return this.getFallbackResponse(message);
        }
    }

    async callGeminiBackend(message) {
        try {
            console.log('🔄 Calling Gemini backend...');

            const response = await fetch(`${this.baseUrl}/api/medical-chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message,
                    conversationHistory: this.conversationHistory,
                    systemPrompt: this.systemPrompt,
                    patientData: this.patientData
                })
            });

            if (!response.ok) {
                throw new Error(`Backend error: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.error) {
                console.warn('⚠️ Backend error, using fallback:', data.error);
                return data.fallback || this.getFallbackResponse(message);
            }
            
            console.log('✅ Gemini response received');
            return data.text.trim();
            
        } catch (error) {
            console.error('❌ Error calling backend:', error);
            console.log('🔄 Falling back to local responses');
            return this.getFallbackResponse(message);
        }
    }

    // Enhanced information extraction using LLM
    async extractPatientInfoWithLLM(conversationText) {
        console.log('🧠 Extracting patient information using LLM...');
        console.log('📝 Current conversation:', conversationText);

        if (this.provider === 'gemini') {
            try {
                const response = await fetch(`${this.baseUrl}/api/extract-patient-info`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        conversationText,
                        extractionPrompt: this.extractionPrompt,
                        currentData: this.patientData
                    })
                });

                if (!response.ok) {
                    console.warn(`⚠️ Extraction API error: ${response.status}, falling back to local extraction`);
                    return this.fallbackExtraction(conversationText);
                }

                const data = await response.json();
                
                if (data.error) {
                    console.warn('⚠️ LLM extraction failed, using fallback:', data.error);
                    return this.fallbackExtraction(conversationText);
                }

                console.log('✅ LLM extraction result:', data.extractedInfo);
                return this.validateAndUpdatePatientData(data.extractedInfo);

            } catch (error) {
                console.error('❌ Error in LLM extraction:', error);
                console.log('🔄 Using fallback extraction instead');
                return this.fallbackExtraction(conversationText);
            }
        } else {
            console.log('📋 Using fallback extraction (no LLM provider)');
            return this.fallbackExtraction(conversationText);
        }
    }

    validateAndUpdatePatientData(extractedInfo) {
        const updates = {};
        let hasUpdates = false;

        // Validate and update name
        if (extractedInfo.name && typeof extractedInfo.name === 'string' && 
            extractedInfo.name.trim().length > 1 && !this.patientData.name) {
            const name = this.validateName(extractedInfo.name.trim());
            if (name) {
                this.patientData.name = name;
                updates.name = name;
                hasUpdates = true;
                console.log('✅ Name extracted:', name);
            }
        }

        // Validate and update age
        if (extractedInfo.age && !this.patientData.age) {
            const age = this.validateAge(extractedInfo.age);
            if (age) {
                this.patientData.age = age;
                updates.age = age;
                hasUpdates = true;
                console.log('✅ Age extracted:', age);
            }
        }

        // Validate and update gender
        if (extractedInfo.gender && !this.patientData.gender) {
            const gender = this.validateGender(extractedInfo.gender);
            if (gender) {
                this.patientData.gender = gender;
                updates.gender = gender;
                hasUpdates = true;
                console.log('✅ Gender extracted:', gender);
            }
        }

        // Validate and update contact
        if (extractedInfo.contact && !this.patientData.contact) {
            const contact = this.validateContact(extractedInfo.contact);
            if (contact) {
                this.patientData.contact = contact;
                updates.contact = contact;
                hasUpdates = true;
                console.log('✅ Contact extracted:', contact);
            }
        }

        // Validate and update symptoms
        if (extractedInfo.symptoms && !this.patientData.symptoms) {
            const symptoms = this.validateSymptoms(extractedInfo.symptoms);
            if (symptoms) {
                this.patientData.symptoms = symptoms;
                updates.symptoms = symptoms;
                hasUpdates = true;
                console.log('✅ Symptoms extracted:', symptoms);
            }
        }

        // Validate and update diagnosis + confidence
        if (extractedInfo.diagnosis && !this.patientData.diagnosis) {
            const hasBasicInfo = this.patientData.name && this.patientData.age &&
                            this.patientData.gender && this.patientData.contact &&
                            this.patientData.symptoms;

            if (hasBasicInfo) {
                const diagnosis = this.validateDiagnosis(extractedInfo.diagnosis);
                if (diagnosis) {
                    this.patientData.diagnosis = diagnosis;
                    updates.diagnosis = diagnosis;
                    hasUpdates = true;
                    console.log('✅ Diagnosis extracted:', diagnosis);

                    // Extract confidence level
                    const rawConf = (extractedInfo.diagnosisConfidence || '').toString().trim().toLowerCase();
                    if (rawConf.includes('high')) {
                        this.patientData.diagnosisConfidence = 'High';
                    } else if (rawConf.includes('medium') || rawConf.includes('moderate')) {
                        this.patientData.diagnosisConfidence = 'Medium';
                    } else if (rawConf.includes('low')) {
                        this.patientData.diagnosisConfidence = 'Low';
                    } else {
                        this.patientData.diagnosisConfidence = 'Low'; // default
                    }
                    updates.diagnosisConfidence = this.patientData.diagnosisConfidence;
                    console.log('✅ Confidence:', this.patientData.diagnosisConfidence);
                }
            } else {
                console.log('⏳ Diagnosis extraction delayed - waiting for basic info');
            }
        }

        // Validate and update preferred doctor
        if (extractedInfo.preferredDoctor && this.patientData.preferredDoctor === "N/A") {
            const preferredDoctor = extractedInfo.preferredDoctor.trim();
            if (preferredDoctor && preferredDoctor.toLowerCase() !== 'none' &&
                preferredDoctor.toLowerCase() !== 'no preference') {
                this.patientData.preferredDoctor = preferredDoctor;
                updates.preferredDoctor = preferredDoctor;
                hasUpdates = true;
                console.log('✅ Preferred doctor extracted:', preferredDoctor);
            }
        }

        // Capture preferred time
        if (extractedInfo.preferredTime && !this.patientData.preferredTime) {
            const t = extractedInfo.preferredTime.trim().toLowerCase();
            if (['morning', 'afternoon', 'evening'].includes(t)) {
                this.patientData.preferredTime = t;
                updates.preferredTime = t;
                hasUpdates = true;
                console.log('✅ Preferred time extracted:', t);
            }
        }

        if (hasUpdates) {
            this.patientData.extractionHistory.push({
                timestamp: new Date().toISOString(),
                updates: updates,
                source: 'LLM'
            });
        }

        return hasUpdates;
    }

    validateName(name) {
        // Clean and validate name
        const cleanName = name.replace(/[^a-zA-Z\s]/g, '').trim();
        const words = cleanName.split(/\s+/).filter(word => word.length > 0);

        // Check for valid name patterns — allow 1 to 4 words
        if (words.length >= 1 && words.length <= 4) {
            // Check if it doesn't contain medical keywords
            const medicalWords = ['pain', 'ache', 'hurt', 'problem', 'trouble', 'issue', 'symptom', 'feel'];
            const hasmedicalWords = words.some(word =>
                medicalWords.includes(word.toLowerCase())
            );

            if (!hasmedicalWords) {
                return words.map(word =>
                    word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
                ).join(' ');
            }
        }
        return null;
    }

    validateAge(age) {
        let numAge;
        if (typeof age === 'string') {
            numAge = parseInt(age.replace(/[^0-9]/g, ''));
        } else {
            numAge = parseInt(age);
        }
        
        if (!isNaN(numAge) && numAge >= 1 && numAge <= 120) {
            return numAge;
        }
        return null;
    }

    validateGender(gender) {
        const genderStr = gender.toLowerCase().trim();
        if (['male', 'man', 'm'].includes(genderStr)) return 'Male';
        if (['female', 'woman', 'f'].includes(genderStr)) return 'Female';
        if (['other', 'non-binary', 'nonbinary', 'prefer not to say'].includes(genderStr)) return 'Other';
        return null;
    }

    validateContact(contact) {
    if (!contact) return null;
    const contactStr = contact.toString().trim();


    // PHONE => normalize digits
    const digits = contactStr.replace(/\D/g, '');
    // Too short or too long -> reject
    if (digits.length < 8 || digits.length > 15) return null;

    // If user entered 10 digits (likely local number), prepend default country
    let normalized = digits;
    if (digits.length === 10 && DEFAULT_COUNTRY_CODE) {
        // remove + from default and prepend its digits
        const countryDigits = DEFAULT_COUNTRY_CODE.replace('+', '');
        normalized = countryDigits + digits;
    }

    // Ensure we return with leading '+'
    if (!normalized.startsWith('+')) normalized = '+' + normalized;

    return normalized; // e.g. '+919876543210' or email string
    }


    validateSymptoms(symptoms) {
        const symptomsStr = symptoms.trim();
        if (symptomsStr.length < 5 || symptomsStr.length >= 500) return null;
        const lower = symptomsStr.toLowerCase();
        // Reject placeholder / extraction-failure text and AI question phrases
        const PLACEHOLDERS = [
            'not provided', 'not specified', 'not mentioned', 'not stated', 'not given',
            'not yet provided', 'not collected', 'not available', 'not applicable',
            'unknown', 'none', 'null', 'n/a', 'no symptoms', 'no complaints',
            'not reported', 'unspecified', 'to be determined', 'tbd',
            'please describe', 'describe your symptoms', 'what brings you in',
            'what are your symptoms', 'what seems to be', 'how can i help',
            'tell me about your', 'could you describe', 'can you describe'
        ];
        if (PLACEHOLDERS.some(p => lower.startsWith(p))) {
            console.warn('⚠️ Rejected placeholder/AI-phrase symptoms:', symptomsStr);
            return null;
        }
        // Must not be a phone number
        if (/^\+?[\d\s\-\(\)]{7,}$/.test(symptomsStr)) {
            console.warn('⚠️ Rejected numeric-only symptoms:', symptomsStr);
            return null;
        }
        return symptomsStr;
    }

    validateDiagnosis(diagnosis) {
        const diagnosisStr = diagnosis.trim();
        if (diagnosisStr.length < 10 || diagnosisStr.length > 1000) return null;

        const lower = diagnosisStr.toLowerCase();

        // Reject bare "unknown" or "unknown condition" — not a clinical diagnosis
        if (/^unknown(\s+condition)?$/i.test(lower)) {
            console.warn('⚠️ Rejected non-specific diagnosis:', diagnosisStr);
            return null;
        }

        // Reject if it looks like a raw symptom list rather than a clinical diagnosis
        const symptomOnlyWords = ['pain', 'ache', 'difficulty', 'feeling', 'having', 'experiencing', 'complains of', 'presents with'];
        const diagnosisIndicators = ['possible', 'probable', 'likely', 'suspected', 'acute', 'chronic', 'syndrome', 'disorder', 'disease', 'infection', 'infarction', 'injury', 'failure', 'deficiency', 'inflammation', 'unknown'];
        const hasDiagnosisIndicator = diagnosisIndicators.some(w => lower.includes(w));
        const isSymptomOnly = symptomOnlyWords.some(w => lower.startsWith(w)) && !hasDiagnosisIndicator;

        if (isSymptomOnly) {
            console.warn('⚠️ Rejected symptom-only diagnosis:', diagnosisStr);
            return null;
        }
        return diagnosisStr;
    }

    // Fallback extraction for when LLM is not available
    fallbackExtraction(conversationText) {
        console.log('📝 Using fallback extraction methods...');
        console.log('🔍 Analyzing text:', conversationText);
        
        let hasUpdates = false;
        const updates = {};

        // Extract name if not already present
        if (!this.patientData.name) {
            // Pattern 1: explicit phrasing
            const nameMatch = conversationText.match(/(?:my name is|i am|i'm|name is|call me)\s+([a-zA-Z\s]{2,40})/i);
            if (nameMatch) {
                const name = this.validateName(nameMatch[1]);
                if (name) {
                    this.patientData.name = name;
                    updates.name = name;
                    hasUpdates = true;
                    console.log('✅ Name extracted (fallback pattern):', name);
                }
            }

            // Pattern 2: if the last AI message asked for name, treat the last user reply as the name
            if (!this.patientData.name && this.conversationHistory.length >= 2) {
                const lastAI = [...this.conversationHistory].reverse().find(m => m.role === 'assistant');
                const lastUser = [...this.conversationHistory].reverse().find(m => m.role === 'user');
                if (lastAI && lastUser &&
                    /name|what.*call|who.*are/i.test(lastAI.content) &&
                    lastUser.content.trim().split(/\s+/).length <= 4) {
                    const name = this.validateName(lastUser.content.trim());
                    if (name) {
                        this.patientData.name = name;
                        updates.name = name;
                        hasUpdates = true;
                        console.log('✅ Name extracted (context reply):', name);
                    }
                }
            }
        }

        // Extract age if not already present
        if (!this.patientData.age) {
            const agePatterns = [
                /(?:i am|i'm|age)\s*(\d{1,3})/i,
                /(\d{1,3})\s*years?\s*old/i,
                /age.*?(\d{1,3})/i,
                /\b(\d{1,3})\b/g // Look for standalone numbers
            ];
            
            for (const pattern of agePatterns) {
                const matches = conversationText.matchAll(pattern);
                for (const match of matches) {
                    const age = this.validateAge(match[1]);
                    if (age) {
                        this.patientData.age = age;
                        updates.age = age;
                        hasUpdates = true;
                        console.log('✅ Age extracted (fallback):', age);
                        break;
                    }
                }
                if (this.patientData.age) break;
            }
        }

        // Extract gender if not already present
        if (!this.patientData.gender) {
            const text = conversationText.toLowerCase();
            if (['male', 'man', 'boy', 'he', 'him'].some(word => text.includes(word))) {
                this.patientData.gender = "Male";
                updates.gender = "Male";
                hasUpdates = true;
                console.log('✅ Gender extracted (fallback): Male');
            } else if (['female', 'woman', 'girl', 'she', 'her'].some(word => text.includes(word))) {
                this.patientData.gender = "Female";
                updates.gender = "Female";
                hasUpdates = true;
                console.log('✅ Gender extracted (fallback): Female');
            }
        }

        // Extract contact if not already present
       if (!this.patientData.contact) {
    // Phone pattern first (since WhatsApp needs phone)
             const phoneMatch = conversationText.match(/\b(\+?[\d\s\-\(\)]{8,})\b/);
             if (phoneMatch && phoneMatch[1].replace(/\D/g, '').length >= 8) {
               const phone = phoneMatch[1].trim();
               this.patientData.contact = phone;
               updates.contact = phone;
               hasUpdates = true;
               console.log('✅ Phone extracted (fallback):', phone);
            } 
            }


        // Extract symptoms if not already present — scan only user lines to avoid picking up AI questions
        if (!this.patientData.symptoms) {
            const medicalKeywords = [
                'pain', 'hurt', 'ache', 'fever', 'headache', 'cough', 'cold',
                'dizzy', 'nausea', 'sick', 'tired', 'sore', 'swollen', 'rash',
                'burn', 'trouble', 'problem', 'difficulty', 'feel',
                'stomach', 'chest', 'back', 'head', 'eye', 'ear', 'throat'
            ];
            // Only look at user-spoken lines, never assistant lines
            const userLines = conversationText.split('\n')
                .filter(l => l.toLowerCase().startsWith('user:'))
                .map(l => l.replace(/^user:\s*/i, '').trim());
            for (const line of userLines) {
                const lineLower = line.toLowerCase();
                if (medicalKeywords.some(k => lineLower.includes(k)) && line.length > 10) {
                    const validated = this.validateSymptoms(line);
                    if (validated) {
                        this.patientData.symptoms = validated;
                        updates.symptoms = validated;
                        hasUpdates = true;
                        console.log('✅ Symptoms extracted (fallback):', validated);
                        break;
                    }
                }
            }
        }

        if (hasUpdates) {
            this.patientData.extractionHistory.push({
                timestamp: new Date().toISOString(),
                updates: updates,
                source: 'Fallback'
            });
            console.log('📊 Fallback extraction updates:', updates);
        } else {
            console.log('❌ No new information extracted from fallback');
        }

        return hasUpdates;
    }

    getFallbackResponse(message) {
        const data = this.patientData;
        void message; // message used via patientData flow

        // Initial greeting
        if (this.conversationHistory.length === 0) {
            return "Hello! I'm here to help you book a medical appointment. To get started, I'll need some basic information. Could you please provide your full name, age, gender, phone number, and describe the health issues or symptoms you're experiencing?";
        }

        // Check what information is still missing
        const missingFields = data.getMissingFields();
        
        // If we have missing required information, ask for it
        if (missingFields.length > 0) {
            const needsBasic = missingFields.includes('name') || missingFields.includes('age') || missingFields.includes('gender');
            if (needsBasic) {
                return "Could you please share your full name, age, and gender?";
            }

            if (missingFields.includes('contact')) {
                return "I need your phone number to confirm your appointment. Could you please provide your mobile number (with country code if possible)?";
            }

            if (missingFields.includes('symptoms')) {
                return "What brings you in today? Please describe your symptoms or health concern.";
            }
        }

        // If we have all basic information, proceed with follow-up questions
        if (data.hasBasicInfo() && !data.detailedAssessmentDone) {
            const followUpQuestions = [
                "Thank you for providing all the basic information. Let me ask a few follow-up questions about your symptoms. How long have you been experiencing these symptoms?",
                "On a scale of 1-10, how would you rate your pain or discomfort?",
                "Is there anything that makes your symptoms better or worse?",
                "Do you have a preferred doctor you'd like to see, or would you like me to recommend one based on your symptoms?",
                "I have all the information needed. Let me book your appointment."
            ];

            const questionIndex = Math.min(this.followUpQuestionsAsked, followUpQuestions.length - 1);
            return followUpQuestions[questionIndex];
        }

        // Default response
        return "Thank you for that information. Is there anything else about your symptoms that you'd like to mention?";
    }

    checkIfAssessmentComplete(aiResponse) {
        // ONLY trigger on the exact completion phrase — do NOT match partial phrases like
        // 'book your appointment' which appears in every mid-conversation message.
        const lower = aiResponse.toLowerCase();
        return lower.includes('i have sufficient information to book your appointment now');
    }

    // Get conversation summary for extraction
    getConversationSummary() {
        return this.conversationHistory.map(msg => 
            `${msg.role}: ${msg.content}`
        ).join('\n');
    }
}

// Medical Matcher with Backend Integration (unchanged)
class GeminiMedicalMatcher {
    constructor(llmInterface) {
        this.llmInterface = llmInterface;
        this.baseUrl = window.location.origin;
        this.medicalKnowledge = {
            'General Physician': 'Primary care doctor who treats common illnesses like fever, cold, flu, headache, stomach pain, body aches, routine checkups, diabetes, hypertension, general health issues',
            'Cardiologist': 'Heart specialist who treats chest pain, heart problems, high blood pressure, heart palpitations, cardiovascular disease, heart attack symptoms',
            'Dermatologist': 'Skin specialist who treats all skin conditions — rash, itching, acne, eczema, psoriasis, skin lesions, hives, dermatitis, skin infections, hair loss, nail problems',
            'Urologist': 'Urinary system specialist who treats kidney stones, bladder problems, urinary infections, prostate issues, kidney disease, urinary incontinence',
            'Gynaecologist': 'Women\'s health specialist who treats menstrual problems, pregnancy care, reproductive health, ovarian cysts, pelvic pain',
            'Ophthalmologist': 'Eye specialist who treats vision problems, eye pain, cataracts, glaucoma, eye infections, retinal problems, vision loss'
        };
    }

    // In app.js - Update the matchSpecializationWithGemini method in GeminiMedicalMatcher class:

    async matchSpecializationWithGemini(patientIssues, availableSpecializations, knowledgeBaseString, preferredTime) {
        console.log(`🔍 Analyzing patient issues: ${patientIssues}`);

        if (!this.llmInterface || this.llmInterface.provider !== 'gemini') {
            return this.fallbackMatching(patientIssues, availableSpecializations);
        }

        try {
            console.log('🔄 Calling enhanced specialization matching and scheduling backend...');

            const response = await fetch(`${this.baseUrl}/api/match-specialization`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    patientIssues,
                    availableSpecializations,
                    knowledgeBaseString,
                    preferredTime: preferredTime || null
                })
            });

            if (!response.ok) {
                throw new Error(`Backend error: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.error) {
                console.warn('⚠️ Enhanced matching failed, using fallback');
                return this.fallbackMatching(patientIssues, availableSpecializations);
            }

            const bestMatch = availableSpecializations.find(spec => 
                data.specialization && (spec.toLowerCase().includes(data.specialization.toLowerCase()) || 
                data.specialization.toLowerCase().includes(spec.toLowerCase()))
            );

            if (bestMatch && data.appointmentDate && data.appointmentTime) {
                console.log(`✅ Gemini recommended: ${bestMatch} (${(data.confidence * 100).toFixed(0)}% confidence)`);
                console.log(`👨‍⚕️ Selected Doctor: ${data.doctorName}`);
                console.log(`📅 Scheduled: ${data.appointmentDate} at ${data.appointmentTime}`);
                console.log(`💡 Reasoning: ${data.reason}`);
                console.log(`⏰ Scheduling Logic: ${data.schedulingReason}`);
                
                return { 
                    specialization: bestMatch, 
                    confidence: data.confidence,
                    doctorName: data.doctorName,
                    appointmentDate: data.appointmentDate,
                    appointmentTime: data.appointmentTime,
                    schedulingReason: data.schedulingReason
                };
            } else {
                return this.fallbackMatching(patientIssues, availableSpecializations);
            }

        } catch (error) {
            console.error('❌ Error in enhanced backend matching:', error);
            return this.fallbackMatching(patientIssues, availableSpecializations);
        }
    }

    fallbackMatching(patientIssues, availableSpecializations) {
        const issues = patientIssues.toLowerCase();

        // Skin conditions — check FIRST, before generic pain/stomach rules
        if (['skin', 'rash', 'itch', 'acne', 'eczema', 'lesion', 'hive', 'dermatit', 'psoriasis', 'wound'].some(symptom => issues.includes(symptom))) {
            const dermatologist = availableSpecializations.find(spec => spec.toLowerCase().includes('dermat'));
            if (dermatologist) return { specialization: dermatologist, confidence: 0.92, reason: 'Patient has skin-related symptoms' };
        }

        // Rule-based matching as fallback
        if (['headache', 'head', 'fever', 'cold', 'flu', 'general', 'pain'].some(symptom => issues.includes(symptom))) {
            const generalPhysician = availableSpecializations.find(spec => 
                spec.toLowerCase().includes('general') || spec.toLowerCase().includes('physician')
            );
            if (generalPhysician) return { specialization: generalPhysician, confidence: 0.85 };
        }

        if (['stomach', 'abdominal', 'digestive', 'nausea', 'vomiting'].some(symptom => issues.includes(symptom))) {
            const generalPhysician = availableSpecializations.find(spec => 
                spec.toLowerCase().includes('general') || spec.toLowerCase().includes('physician')
            );
            if (generalPhysician) return { specialization: generalPhysician, confidence: 0.90 };
        }

        if (['chest pain', 'heart', 'cardiac', 'palpitation'].some(symptom => issues.includes(symptom))) {
            const cardiologist = availableSpecializations.find(spec => spec.toLowerCase().includes('cardio'));
            if (cardiologist) return { specialization: cardiologist, confidence: 0.90 };
        }

        if (['eye', 'vision', 'sight', 'visual'].some(symptom => issues.includes(symptom))) {
            const ophthalmologist = availableSpecializations.find(spec => spec.toLowerCase().includes('ophthalm'));
            if (ophthalmologist) return { specialization: ophthalmologist, confidence: 0.90 };
        }

        if (['urin', 'bladder', 'kidney', 'uti'].some(symptom => issues.includes(symptom))) {
            const urologist = availableSpecializations.find(spec => spec.toLowerCase().includes('urolog'));
            if (urologist) return { specialization: urologist, confidence: 0.90 };
        }

        if (['menstrual', 'period', 'pregnancy', 'gynec'].some(symptom => issues.includes(symptom))) {
            const gynaecologist = availableSpecializations.find(spec => 
                spec.toLowerCase().includes('gynec') || spec.toLowerCase().includes('gynaec')
            );
            if (gynaecologist) return { specialization: gynaecologist, confidence: 0.90 };
        }

        // Default to General Physician
        const generalPhysician = availableSpecializations.find(spec => 
            spec.toLowerCase().includes('general') || spec.toLowerCase().includes('physician')
        );
        if (generalPhysician) return { specialization: generalPhysician, confidence: 0.75 };

        return { specialization: availableSpecializations[0], confidence: 0.60 };
    }
}

// Enhanced Appointment Creator with LLM Integration
class EnhancedAppointmentCreator {
    constructor(llmProvider, apiKey) {
        this.llmInterface = new LLMPatientInterface(llmProvider, apiKey);
        this.matcher = new GeminiMedicalMatcher(this.llmInterface);
        this.conversationCount = 0;
        this.maxConversations = 20;
        this.extractionInterval = null;
        this.bookingInProgress = false;  // locked once the doctor-selection flow starts
    }

    async chatWithPatient() {
        // Hardcoded warm greeting — reliable and friendly, not AI-generated
        const initialResponse = "Hello! Welcome to MedEasy. Could you please share your name, age, and gender?";
        this.addMessage('ai', initialResponse);
        // Seed history so the AI knows the greeting was already sent and what was asked
        this.llmInterface.conversationHistory.push({ role: 'assistant', content: initialResponse });

        // Start periodic extraction
        this.startPeriodicExtraction();

        return new Promise((resolve) => {
            this.resolveChat = resolve;
        });
    }

    startPeriodicExtraction() {
        let lastExtractionLength = 0;
        this.extractionInterval = setInterval(async () => {
            // Stop polling once all required fields are known
            if (this.llmInterface.patientData.hasBasicInfo()) {
                this.stopPeriodicExtraction();
                console.log('⏹ Periodic extraction stopped — all required fields captured');
                return;
            }
            const currentLength = this.llmInterface.conversationHistory.length;
            if (currentLength > lastExtractionLength) {
                lastExtractionLength = currentLength;
                console.log('⏰ Periodic extraction check...');
                const conversationText = this.llmInterface.getConversationSummary();
                const extractionResult = await this.llmInterface.extractPatientInfoWithLLM(conversationText);
                if (extractionResult) {
                    console.log('✅ Periodic extraction found new information');
                    this.updateProgress();
                }
            }
        }, 20000);
    }

    stopPeriodicExtraction() {
        if (this.extractionInterval) {
            clearInterval(this.extractionInterval);
            this.extractionInterval = null;
        }
    }

    async handleUserMessage(message) {
        if (!message.trim()) return;
        // Prevent any new input once the booking flow has started
        if (this.bookingInProgress) return;

        // Direct booking intent — patient already knows what they need
        const DIRECT_BOOK_PHRASES = [
            'just book', 'book now', 'book appointment', 'book it', 'skip',
            'i know the doctor', 'i know my doctor', 'already know', 'direct booking',
            'no need for questions', 'skip questions', 'skip assessment',
            'book directly', 'direct book', 'go to booking', 'proceed to booking',
            'i want to book', 'book an appointment now'
        ];
        const msgLower = message.trim().toLowerCase();
        if (this.llmInterface.patientData.hasBasicInfo() &&
            DIRECT_BOOK_PHRASES.some(p => msgLower.includes(p))) {
            this.addMessage('user', message);
            this.llmInterface.conversationHistory.push({ role: 'user', content: message });
            await this.skipToBooking();
            return;
        }

        // Add user message
        this.addMessage('user', message);

        // Add to conversation history FIRST
        this.llmInterface.conversationHistory.push({ role: 'user', content: message });

        // Fast regex extraction for simple single-field answers — no Ollama needed
        this.fastExtract(message);

        // Always build conversationText here so the completion-signal handler can use it
        const conversationText = this.llmInterface.getConversationSummary();

        // Only run full LLM extraction if we still have missing required fields
        // (skips the Ollama call once name+age+gender+symptoms are all known)
        if (!this.llmInterface.patientData.hasBasicInfo()) {
            console.log('🔍 Running LLM extraction (missing fields)...');
            const extractionResult = await this.llmInterface.extractPatientInfoWithLLM(conversationText);
            if (extractionResult) {
                this.updateProgress();
            }
        } else {
            console.log('⚡ Skipping LLM extraction — all required fields already captured');
        }

        // Update differential BEFORE calling the LLM so it has fresh data this turn
        const pd = this.llmInterface.patientData;
        if (pd.hasBasicInfo() && !pd.detailedAssessmentDone) {
            try {
                const diffRes = await fetch(`${window.location.origin}/api/update-differential`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        conversationText,
                        symptoms: pd.symptoms,
                        gender: pd.gender,
                        currentDifferential: pd.differential
                    })
                });
                const diffData = await diffRes.json();
                if (diffData.differential && diffData.differential.length) {
                    pd.differential = diffData.differential;
                    // Fresh differential built — clear the escalation flag so the server's
                    // follow-up counter returns to normal from the next turn onwards.
                    pd.escalationDetected = false;
                    console.log('🧬 Differential (pre-LLM):', diffData.differential.map(d =>
                        `${d.condition}(p=${d.probability}, for=${(d.evidence_for||[]).length}, against=${(d.evidence_against||[]).length})`
                    ).join(' | '));
                }
            } catch (e) {
                console.warn('⚠️ Pre-LLM differential update failed:', e.message);
            }

            // Build symptomsSummary: initial complaint + contextual Q&A pairs
            // Each follow-up answer is stored with the question that prompted it,
            // so "No, not really." becomes "Does the pain worsen after eating? No, not really."
            const history = this.llmInterface.conversationHistory;
            const qaPairs = [];
            let pastInitialSymptoms = false;
            let pendingQuestion = null;
            const symptomsKeyword = pd.symptoms
                ? pd.symptoms.toLowerCase().split(/[\s,]+/).find(w => w.length > 3) || ''
                : '';
            for (const msg of history) {
                // Find the user turn where symptoms were first mentioned
                if (!pastInitialSymptoms && msg.role === 'user' &&
                    symptomsKeyword && msg.content.toLowerCase().includes(symptomsKeyword)) {
                    pastInitialSymptoms = true;
                    continue;
                }
                if (!pastInitialSymptoms) continue;

                if (msg.role === 'assistant') {
                    // Extract the question sentence(s) from the AI response
                    const questionMatch = msg.content.match(/[^.!?\n]*\?/g);
                    if (questionMatch && questionMatch.length > 0) {
                        // Use the last question sentence (most specific)
                        pendingQuestion = questionMatch[questionMatch.length - 1].trim();
                    }
                } else if (msg.role === 'user' && msg.content.trim().length > 2) {
                    const answer = msg.content.trim();
                    if (pendingQuestion) {
                        qaPairs.push(`${pendingQuestion} → ${answer}`);
                        pendingQuestion = null;
                    } else {
                        qaPairs.push(answer);
                    }
                }
            }
            console.log('📝 symptomsSummary Q&A pairs:', qaPairs);
            pd.symptomsSummary = pd.symptoms
                ? [pd.symptoms, ...qaPairs].join('\n')
                : '';

            // Escalation detection: if the current message introduces new critical symptoms
            // that weren't in the original complaint, update the symptom profile immediately
            if (pd.symptoms) {
                const escalation = this.detectSymptomEscalation(message, pd.symptoms);
                if (escalation.detected) {
                    pd.symptoms = `${pd.symptoms}; ${escalation.newSymptoms}`;
                    pd.symptomsSummary = [pd.symptoms, ...qaPairs].join('\n');
                    // Reset differential so it re-evaluates with the full symptom picture next turn.
                    // Set escalationDetected so the server resets the follow-up counter, ensuring
                    // it asks fresh questions about the new critical symptoms.
                    pd.differential = [];
                    pd.escalationDetected = true;
                    console.log('🚨 Symptom escalation detected:', escalation.newSymptoms);
                    this.addMessage('ai', escalation.alertMessage);
                }
            }
        }

        // Get AI response
        const aiResponse = await this.llmInterface.callLLM(message);
        this.addMessage('ai', aiResponse);

        // Add AI response to history
        this.llmInterface.conversationHistory.push({ role: 'assistant', content: aiResponse });

        // Track questions asked — extract any sentence ending in '?' from AI response
        const questionSentences = aiResponse.match(/[^.!?]*\?/g);
        if (questionSentences) {
            questionSentences.forEach(q => {
                const trimmed = q.trim();
                if (trimmed.length > 10 && !this.llmInterface.patientData.askedQuestions.includes(trimmed)) {
                    this.llmInterface.patientData.askedQuestions.push(trimmed);
                }
            });
        }

        // Check if AI provided the exact completion signal
        if (aiResponse.toLowerCase().includes('i have sufficient information to book your appointment now')) {
            
            console.log('🎯 Detected completion signal in AI response');
            // Lock input immediately — prevent user messages from re-entering the AI chat
            this.bookingInProgress = true;
            const _inp = document.getElementById('userInput');
            const _btn = document.getElementById('sendBtn');
            if (_inp) _inp.disabled = true;
            if (_btn) _btn.disabled = true;
            
            // FORCE diagnosis + confidence extraction from the AI response directly
            if (aiResponse.toLowerCase().includes('diagnosis:')) {
                const diagnosisMatch = aiResponse.match(/diagnosis:\s*(.+?)(?:\n|$)/i);
                if (diagnosisMatch && !this.llmInterface.patientData.diagnosis) {
                    const diag = diagnosisMatch[1].trim();
                    const validated = this.llmInterface.validateDiagnosis(diag);
                    if (validated) {
                        this.llmInterface.patientData.diagnosis = validated;
                        console.log('✅ Diagnosis from AI response:', validated);
                    }
                }

                // Extract confidence from AI response text
                const confMatch = aiResponse.match(/confidence:\s*(low|medium|high)/i);
                if (confMatch) {
                    const conf = confMatch[1].charAt(0).toUpperCase() + confMatch[1].slice(1).toLowerCase();
                    this.llmInterface.patientData.diagnosisConfidence = conf;
                    console.log('✅ Confidence from AI response:', conf);
                }
                this.updateProgress();
            }
            
            this.llmInterface.patientData.detailedAssessmentDone = true;

            // Await extraction fully before booking — Ollama is slow, don't race
            (async () => {
                this.addMessage('ai', "⏳ Finalising your information...");
                const finalExtractionResult = await this.llmInterface.extractPatientInfoWithLLM(conversationText);
                if (finalExtractionResult) {
                    this.updateProgress();
                }

                // If symptoms still missing after extraction, pull from conversation history
                if (!this.llmInterface.patientData.symptoms) {
                    // Find the user message that came after symptoms were asked (usually after contact)
                    const history = this.llmInterface.conversationHistory;
                    for (let i = 0; i < history.length; i++) {
                        const msg = history[i];
                        if (msg.role === 'user') {
                            const lc = msg.content.toLowerCase();
                            // Skip messages that are clearly just names, numbers, or single words
                            const isShort = msg.content.trim().split(/\s+/).length <= 2;
                            const isNumber = /^\d+$/.test(msg.content.trim());
                            const isGender = /^(male|female|other|m|f)$/i.test(msg.content.trim());
                            if (!isShort && !isNumber && !isGender && msg.content.length > 5) {
                                // Likely a symptom message
                                if (lc.includes('pain') || lc.includes('ache') || lc.includes('bleed') ||
                                    lc.includes('head') || lc.includes('fever') || lc.includes('cough') ||
                                    lc.includes('rash') || lc.includes('skin') || lc.includes('breath') ||
                                    lc.includes('pressure') || lc.includes('bp') || lc.includes('chest') ||
                                    lc.includes('stomach') || lc.includes('nausea') || lc.includes('dizz') ||
                                    msg.content.length > 15) {
                                    const validated = this.llmInterface.validateSymptoms(msg.content.trim());
                                    if (validated) {
                                        this.llmInterface.patientData.symptoms = validated;
                                        console.log('⚠️ Symptoms fallback from conversation:', validated);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                if (!this.llmInterface.patientData.hasBasicInfo()) {
                    this.addMessage('ai', "❌ I still need your symptoms to book an appointment. Could you describe what you are experiencing?");
                    return;
                }

                // If diagnosis still missing, make a dedicated call to get it
                if (!this.llmInterface.patientData.diagnosis) {
                    try {
                        const diagRes = await fetch(`${window.location.origin}/api/diagnose`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                conversationText: this.llmInterface.getConversationSummary(),
                                symptoms: this.llmInterface.patientData.symptoms,
                                gender: this.llmInterface.patientData.gender
                            })
                        });
                        const diagData = await diagRes.json();
                        if (diagData.diagnosis) {
                            const validated = this.llmInterface.validateDiagnosis(diagData.diagnosis);
                            if (validated) {
                                this.llmInterface.patientData.diagnosis = validated;
                                this.llmInterface.patientData.diagnosisConfidence = diagData.confidence || 'Low';
                                console.log('✅ Diagnosis from /api/diagnose:', validated);
                                this.updateProgress();
                            }
                        }
                    } catch (e) {
                        console.warn('⚠️ /api/diagnose call failed:', e.message);
                    }
                }

                this.addMessage('ai', "✅ Medical assessment complete! Let me show you the available doctors.");
                await this.showPreferredDoctorSelection();
            })();
            return;
        }

        // Check if assessment complete
        // Check if assessment complete
        if (this.llmInterface.checkIfAssessmentComplete(aiResponse) && this.llmInterface.patientData.hasBasicInfo()) {
            this.llmInterface.patientData.detailedAssessmentDone = true;
            this.addMessage('ai', "✅ Assessment complete! Processing appointment...");
        }

        // Track follow-up questions
        if (this.llmInterface.patientData.hasBasicInfo() && !this.llmInterface.patientData.detailedAssessmentDone) {
            this.llmInterface.followUpQuestionsAsked++;

            if (this.llmInterface.followUpQuestionsAsked >= this.llmInterface.maxFollowUpQuestions) {
                this.addMessage('ai', "✅ Sufficient information collected. Processing appointment...");
                this.llmInterface.patientData.detailedAssessmentDone = true;
            }
        }

        this.conversationCount++;

        // Check completion
        if (this.llmInterface.patientData.isComplete() || this.conversationCount >= this.maxConversations) {
            this.stopPeriodicExtraction();
            if (this.llmInterface.patientData.hasBasicInfo()) {
                this.bookingInProgress = true;
                const _i = document.getElementById('userInput');
                const _b = document.getElementById('sendBtn');
                if (_i) _i.disabled = true;
                if (_b) _b.disabled = true;
                setTimeout(() => this.showPreferredDoctorSelection(), 1000);
            } else {
                this.addMessage('ai', "❌ I need more information to book your appointment. Please provide your details.");
            }
        }

        // Debug: Log current patient data
        console.log('📊 Current patient data:', this.llmInterface.patientData);

        // Show quick-book bar as soon as name + age + gender + contact are collected
        // (don't wait for symptoms — patient may already know their doctor)
        const _pd = this.llmInterface.patientData;
        const _hasDemo = !!(_pd.name && _pd.age && _pd.gender && _pd.contact);
        const _bar = document.getElementById('quickBookBar');
        if (_bar && !this.bookingInProgress) {
            _bar.style.display = _hasDemo ? 'block' : 'none';
        }
    }

    // Skip assessment and go straight to doctor selection
    async skipToBooking() {
        if (this.bookingInProgress) return;
        this.bookingInProgress = true;
        const inp = document.getElementById('userInput');
        const btn = document.getElementById('sendBtn');
        const bar = document.getElementById('quickBookBar');
        if (inp) inp.disabled = true;
        if (btn) btn.disabled = true;
        if (bar) bar.style.display = 'none';
        // Default symptoms if patient skipped the symptom step
        if (!this.llmInterface.patientData.symptoms) {
            this.llmInterface.patientData.symptoms = 'General consultation';
        }
        this.llmInterface.patientData.detailedAssessmentDone = true;
        this.addMessage('ai', "Got it — taking you straight to booking.");
        await this.showPreferredDoctorSelection();
    }

    detectSymptomEscalation(message, currentSymptoms) {
        const ESCALATION_CATEGORIES = [
            { terms: ['chest pain', 'chest tightness', 'chest pressure', 'chest discomfort', 'pain in chest', 'chest hurts', 'chest is hurting', 'chest ache'], label: 'chest pain' },
            { terms: ['faint', 'fainting', 'passed out', 'blacked out', 'syncope', 'lost consciousness', 'feel faint', 'going to faint', 'nearly fainted'], label: 'fainting' },
            { terms: ["can't breathe", 'cannot breathe', 'difficulty breathing', 'shortness of breath', 'short of breath', 'trouble breathing', 'breathless', 'unable to breathe'], label: 'difficulty breathing' },
            { terms: ['coughing blood', 'vomiting blood', 'blood in urine', 'blood in stool', 'rectal bleeding', 'haemoptysis', 'coughing up blood'], label: 'bleeding' },
            { terms: ['seizure', 'convulsion', 'fits', 'shaking uncontrollably', 'epileptic'], label: 'seizures' },
            { terms: ['rapid heartbeat', 'heart racing', 'palpitation', 'heart pounding', 'irregular heartbeat', 'skipping beats', 'racing heart'], label: 'heart palpitations' },
            { terms: ['crushing pain', 'radiating pain', 'pain radiating', 'pain down arm', 'left arm pain', 'jaw pain', 'pain in jaw', 'arm going numb'], label: 'radiating cardiac-type pain' },
            { terms: ['sudden weakness', 'sudden numbness', 'face drooping', 'arm weakness', 'sudden confusion', 'sudden severe headache', 'worst headache of my life'], label: 'possible stroke symptoms' },
        ];

        const msgLower = message.toLowerCase();
        const existingLower = currentSymptoms.toLowerCase();

        // Return true if the keyword at position `idx` in `text` is preceded by a negation
        const NEGATIONS = ['not', "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
                           'never', 'no', 'without', "can't", "cant", 'cannot', "isn't", "isnt",
                           "aren't", "arent", "won't", "wont", "i'm not", "im not"];
        const isNegated = (text, idx) => {
            const before = text.slice(Math.max(0, idx - 60), idx);
            const words = before.trim().split(/\s+/).slice(-6);
            return NEGATIONS.some(n => words.join(' ').endsWith(n) || words.includes(n));
        };

        const newCritical = [];
        for (const cat of ESCALATION_CATEGORIES) {
            const inMessage = cat.terms.some(t => {
                const idx = msgLower.indexOf(t);
                return idx !== -1 && !isNegated(msgLower, idx);
            });
            const alreadyKnown = cat.terms.some(t => existingLower.includes(t));
            if (inMessage && !alreadyKnown) newCritical.push(cat.label);
        }

        if (newCritical.length === 0) return { detected: false };

        const symptomList = newCritical.join(' and ');
        const alertMessage = `I've noticed you've mentioned ${symptomList}. These can indicate a serious condition — I'm updating your assessment now so we find the right specialist for you.`;
        return { detected: true, newSymptoms: newCritical.join(', '), alertMessage };
    }

    addMessage(sender, text) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.textContent = text;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Speak AI messages aloud using browser TTS
        if (sender === 'ai') {
            this.speak(text);
        }
    }

    speak(text) {
        if (!window.speechSynthesis) return;
        const voiceBtn = document.getElementById('voiceToggleBtn');
        if (voiceBtn && voiceBtn.dataset.muted === 'true') return;

        window.speechSynthesis.cancel();

        // Remove emoji and markdown, then split into sentences
        // Chrome has a bug where long utterances are silently dropped
        const clean = text.replace(/[\u{1F300}-\u{1FAFF}✅❌⚠️🔄⏳👋]/gu, '').trim();
        const sentences = clean.match(/[^.!?]+[.!?]+/g) || [clean];

        const voices = window.speechSynthesis.getVoices();
        const preferred =
            voices.find(v => v.name === 'Samantha') ||
            voices.find(v => v.lang.startsWith('en') && v.localService) ||
            null;

        sentences.forEach(sentence => {
            const utterance = new SpeechSynthesisUtterance(sentence.trim());
            utterance.rate = 1.15;
            utterance.pitch = 1.05;
            utterance.volume = 1.0;
            if (preferred) utterance.voice = preferred;
            window.speechSynthesis.speak(utterance);
        });
    }

    // Instant regex-based extraction — no AI call, handles simple single-field answers
    fastExtract(message) {
        const pd = this.llmInterface.patientData;
        const msg = message.trim();
        const lc = msg.toLowerCase();
        let changed = false;

        // Name: first message that is 2-4 words, no digits, no medical keywords
        if (!pd.name) {
            const wordCount = msg.split(/\s+/).length;
            const hasDigits = /\d/.test(msg);
            const medicalWords = /fever|pain|cough|cold|headache|rash|bleed|bp|pressure|nausea/i.test(msg);
            if (wordCount >= 1 && wordCount <= 4 && !hasDigits && !medicalWords && msg.length >= 2) {
                pd.name = msg;
                console.log('⚡ Fast extracted name:', msg);
                changed = true;
            }
        }

        // Age: pure number or "N years old"
        if (!pd.age) {
            const ageMatch = msg.match(/^(\d{1,3})(\s*(years?\s*old)?)?$/i);
            if (ageMatch) {
                const age = parseInt(ageMatch[1]);
                if (age > 0 && age < 120) {
                    pd.age = age;
                    console.log('⚡ Fast extracted age:', age);
                    changed = true;
                }
            }
        }

        // Gender
        if (!pd.gender) {
            if (/^(male|m)$/i.test(msg))        { pd.gender = 'Male';   changed = true; console.log('⚡ Fast extracted gender: Male'); }
            else if (/^(female|f)$/i.test(msg)) { pd.gender = 'Female'; changed = true; console.log('⚡ Fast extracted gender: Female'); }
            else if (/^other$/i.test(msg))       { pd.gender = 'Other';  changed = true; console.log('⚡ Fast extracted gender: Other'); }
        }

        // Phone: 7+ digit string (with optional +, spaces, dashes)
        if (!pd.contact) {
            const phoneMatch = msg.match(/^[\+\d][\d\s\-\(\)]{6,}$/);
            if (phoneMatch) {
                pd.contact = msg.replace(/\s+/g, '');
                console.log('⚡ Fast extracted contact:', pd.contact);
                changed = true;
            }
        }

        // Preferred time
        if (!pd.preferredTime) {
            if (lc.includes('morning'))           { pd.preferredTime = 'morning';   changed = true; }
            else if (lc.includes('afternoon'))    { pd.preferredTime = 'afternoon'; changed = true; }
            else if (lc.includes('evening') || lc.includes('pm')) { pd.preferredTime = 'evening'; changed = true; }
            if (pd.preferredTime) console.log('⚡ Fast extracted preferredTime:', pd.preferredTime);
        }

        if (changed) this.updateProgress();
    }

    initVoiceToggle() {
        const btn = document.getElementById('voiceToggleBtn');
        if (!btn) return;
        btn.dataset.muted = 'false';
        btn.addEventListener('click', () => {
            const muted = btn.dataset.muted === 'true';
            btn.dataset.muted = muted ? 'false' : 'true';
            btn.textContent = muted ? '🔊' : '🔇';
            btn.title = muted ? 'Voice on' : 'Voice off';
            if (!muted) window.speechSynthesis.cancel();
        });

        // Pre-load voices so first utterance isn't silent
        if (window.speechSynthesis.getVoices().length === 0) {
            window.speechSynthesis.addEventListener('voiceschanged', () => {
                const voices = window.speechSynthesis.getVoices();
                const samantha = voices.find(v => v.name === 'Samantha');
                if (samantha) console.log('✅ Samantha voice ready');
            }, { once: true });
        }
    }

    updateProgress() {
        const data = this.llmInterface.patientData;
        
        console.log('🔄 Updating progress with data:', {
            name: data.name,
            age: data.age, 
            gender: data.gender,
            contact: data.contact,
            symptoms: data.symptoms,
            diagnosis: data.diagnosis
        });
        
        this.updateProgressItem('nameProgress', data.name, 'Name');
        this.updateProgressItem('ageProgress', data.age, 'Age');
        this.updateProgressItem('genderProgress', data.gender, 'Gender');
        this.updateProgressItem('contactProgress', data.contact, 'Contact');
        this.updateProgressItem('symptomsProgress', data.symptoms, 'Symptoms');

        // Diagnosis with confidence level
        const diagnosisElement = document.getElementById('diagnosisProgress');
        if (diagnosisElement) {
            if (data.diagnosis) {
                const short = data.diagnosis.length > 35 ? data.diagnosis.substring(0, 35) + '…' : data.diagnosis;
                const conf = data.diagnosisConfidence || 'Low';
                const confIcon = conf === 'High' ? '🟢' : conf === 'Medium' ? '🟡' : '🔴';
                diagnosisElement.className = 'progress-item complete';
                diagnosisElement.textContent = `✅ ${short} (${confIcon} ${conf} confidence)`;
            } else {
                const msgCount = this.llmInterface.conversationHistory.filter(m => m.role === 'user').length;
                const hasSymptoms = !!data.symptoms;
                const hasBasic = !!(data.name && data.age && data.gender && data.contact);
                let confidence, icon;
                if (!hasSymptoms || msgCount < 2) {
                    confidence = 'Awaiting symptoms';
                    icon = '⏳';
                } else if (!hasBasic || msgCount < 4) {
                    confidence = 'Low confidence — gathering info';
                    icon = '🔴';
                } else if (msgCount < 7) {
                    confidence = 'Medium confidence — assessing';
                    icon = '🟡';
                } else {
                    confidence = 'High confidence — finalising';
                    icon = '🟢';
                }
                diagnosisElement.className = 'progress-item in-progress';
                diagnosisElement.textContent = `${icon} ${confidence}`;
            }
        }

        const assessmentElement = document.getElementById('assessmentProgress');
        if (assessmentElement) {
            if (data.detailedAssessmentDone) {
                assessmentElement.className = 'progress-item complete';
                assessmentElement.textContent = '✅ Assessment Complete';
            } else if (data.hasBasicInfo()) {
                assessmentElement.className = 'progress-item in-progress';
                assessmentElement.textContent = '🔄 Assessment In Progress';
            } else {
                assessmentElement.className = 'progress-item incomplete';
                assessmentElement.textContent = '⏳ Assessment Pending';
            }
        }

        // Show "Book Now" quick-action bar as soon as demographic info is collected
        const quickBookBar = document.getElementById('quickBookBar');
        if (quickBookBar && !this.bookingInProgress) {
            const hasDemo = !!(data.name && data.age && data.gender && data.contact);
            quickBookBar.style.display = hasDemo ? 'block' : 'none';
        }

        // Show extraction history if available
        if (data.extractionHistory.length > 0) {
            console.log('📊 Extraction History:', data.extractionHistory);
        }

        // Log progress for debugging
        const extractedFields = data.getExtractedFields();
        const missingFields = data.getMissingFields();
        console.log(`📈 Progress: ${extractedFields.length}/6 fields extracted`);
        console.log(`✅ Extracted: ${extractedFields.join(', ')}`);
        console.log(`❌ Missing: ${missingFields.join(', ')}`);
    }

    updateProgressItem(elementId, value, label) {
        const element = document.getElementById(elementId);
        if (!element) {
            console.warn(`⚠️ Progress element not found: ${elementId}`);
            return;
        }
        
        if (value) {
            element.className = 'progress-item complete';
            element.textContent = `✅ ${label}: ${typeof value === 'string' && value.length > 20 ? value.substring(0, 20) + '...' : value}`;
            console.log(`✅ Updated ${label}: ${value}`);
        } else {
            element.className = 'progress-item incomplete';
            element.textContent = `❌ ${label}`;
        }
    }

    // ── Doctor Calendar: show available slots and let patient pick one ──
    async showDoctorCalendar(doctor) {
        document.getElementById('loadingPanel').style.display = 'block';

        try {
            const res = await fetch(`${window.location.origin}/api/doctors/slots?doctorName=${encodeURIComponent(doctor.name)}&count=15`);
            const data = await res.json();
            const slots = data.slots || [];

            document.getElementById('loadingPanel').style.display = 'none';

            if (!slots.length) {
                this.llmInterface.patientData.preferredDoctor = doctor.name;
                this.llmInterface.patientData.detailedAssessmentDone = true;
                await this.createAppointment();
                return;
            }

            // Group slots by date
            const byDate = {};
            slots.forEach(s => {
                if (!byDate[s.appointmentDate]) byDate[s.appointmentDate] = [];
                byDate[s.appointmentDate].push(s.appointmentTime);
            });

            document.getElementById('calendarDoctorName').textContent =
                `Dr. ${doctor.name} — ${doctor.specialization}`;

            const container = document.getElementById('calendarSlots');
            container.innerHTML = Object.entries(byDate).map(([date, times]) => {
                const label = new Date(date + 'T12:00:00').toLocaleDateString('en-US',
                    { weekday: 'long', month: 'long', day: 'numeric' });
                const btns = times.map(t =>
                    `<button class="slot-btn" onclick="window._pickSlot('${date}','${t}')">${t}</button>`
                ).join('');
                return `<div class="calendar-date-group">
                    <div class="calendar-date-label">${label}</div>
                    <div class="calendar-time-slots">${btns}</div>
                </div>`;
            }).join('');

            document.getElementById('calendarPanel').style.display = 'block';

            window._pickSlot = async (date, time) => {
                this._selectedSlot = { date, time };
                document.getElementById('calendarPanel').style.display = 'none';
                document.getElementById('loadingPanel').style.display = 'block';
                this.llmInterface.patientData.preferredDoctor = doctor.name;
                this.llmInterface.patientData.detailedAssessmentDone = true;
                await this.createAppointment();
            };

        } catch (err) {
            console.error('❌ showDoctorCalendar error:', err);
            document.getElementById('loadingPanel').style.display = 'none';
            this.llmInterface.patientData.preferredDoctor = doctor.name;
            this.llmInterface.patientData.detailedAssessmentDone = true;
            await this.createAppointment();
        }
    }

    // ── Step 1: Ask if patient has a preferred doctor (show specialty-filtered doctors) ──
    async showPreferredDoctorSelection() {
        document.getElementById('chatContainer').style.display = 'none';
        document.getElementById('loadingPanel').style.display = 'block';

        try {
            const data = this.llmInterface.patientData;

            // Match specialty from the patient's full symptom summary before fetching doctors,
            // so we only show relevant specialists (not ophthalmologists for GERD symptoms etc.)
            let matchedSpec = '';
            try {
                const kbRes  = await fetch(`${window.location.origin}/api/knowledge-base`);
                const kbData = await kbRes.json();
                const matchRes = await fetch(`${window.location.origin}/api/match-specialization`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        patientIssues: data.symptomsSummary || data.symptoms || '',
                        availableSpecializations: kbData.availableSpecializations || [],
                        knowledgeBaseString: kbData.knowledgeBaseString || '',
                        preferredTime: data.preferredTime || null
                    })
                });
                const matchData = await matchRes.json();
                matchedSpec = matchData.specialization || '';
                console.log('🩺 Specialty matched for preferred doctor panel:', matchedSpec);
            } catch (_) {}

            const params = new URLSearchParams();
            if (matchedSpec) params.set('specialization', matchedSpec);
            if (data.preferredTime) params.set('preferredTime', data.preferredTime);

            const docRes = await fetch(`${window.location.origin}/api/doctors/available?${params}`);
            const docData = await docRes.json();
            const doctors = docData.doctors || [];

            document.getElementById('loadingPanel').style.display = 'none';

            if (!doctors.length) {
                await this.showDoctorSelection();
                return;
            }

            const grid = document.getElementById('preferredDoctorCards');
            grid.innerHTML = doctors.map((doc, idx) => {
                const stars = this._renderStars(doc.rating);
                const slotText = doc.nextSlot
                    ? `Next slot: ${doc.nextSlot.appointmentDate} at ${doc.nextSlot.appointmentTime}`
                    : 'Availability to be confirmed';
                return `<div class="doctor-card" onclick="window._selectPreferredDoctor(${idx})">
                    <div class="doctor-card-left">
                        <div class="doctor-avatar">👨‍⚕️</div>
                        <div class="doctor-info">
                            <div class="doctor-name">Dr. ${doc.name}</div>
                            <div class="doctor-spec">${doc.specialization}</div>
                            <div class="doctor-slot${doc.nextSlot ? '' : ' no-slot'}">${slotText}</div>
                        </div>
                    </div>
                    <div class="doctor-card-right">
                        <div class="star-rating">${stars}<span class="rating-num">${doc.rating.toFixed(1)}</span></div>
                        <button class="select-btn">Select</button>
                    </div>
                </div>`;
            }).join('');

            this._allDoctorList = doctors;
            document.getElementById('preferredDoctorPanel').style.display = 'block';

            window._selectPreferredDoctor = async (idx) => {
                const chosen = this._allDoctorList[idx];
                document.getElementById('preferredDoctorPanel').style.display = 'none';
                await this.showDoctorCalendar(chosen);
            };

            window._noPreferredDoctor = async () => {
                document.getElementById('preferredDoctorPanel').style.display = 'none';
                await this.showDoctorSelection();
            };

        } catch (err) {
            console.error('❌ showPreferredDoctorSelection error:', err);
            document.getElementById('loadingPanel').style.display = 'none';
            await this.showDoctorSelection();
        }
    }

    // ── Step 2: Specialty-based doctor selection (shown when patient has no preference) ──
    async showDoctorSelection() {
        const data = this.llmInterface.patientData;
        document.getElementById('chatContainer').style.display = 'none';
        document.getElementById('loadingPanel').style.display = 'block';

        try {
            // Determine specialization from diagnosis or let server infer
            const symptoms  = data.symptoms  || '';

            // Load knowledge base to get specializations
            const kbRes  = await fetch(`${window.location.origin}/api/knowledge-base`);
            const kbData = await kbRes.json();
            const specs  = kbData.availableSpecializations || [];

            // Match specialization from the full symptom summary (not just initial complaint)
            let matchedSpec = '';
            try {
                const matchRes  = await fetch(`${window.location.origin}/api/match-specialization`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        patientIssues: data.symptomsSummary || symptoms,
                        availableSpecializations: specs,
                        knowledgeBaseString: kbData.knowledgeBaseString,
                        preferredTime: data.preferredTime || null
                    })
                });
                const matchData = await matchRes.json();
                matchedSpec = matchData.specialization || '';
            } catch (_) {}

            // Fetch available doctors filtered by specialization + preferred time
            const params = new URLSearchParams();
            if (matchedSpec) params.set('specialization', matchedSpec);
            if (data.preferredTime) params.set('preferredTime', data.preferredTime);

            const docRes  = await fetch(`${window.location.origin}/api/doctors/available?${params}`);
            const docData = await docRes.json();
            const doctors = docData.doctors || [];

            document.getElementById('loadingPanel').style.display = 'none';

            if (!doctors.length) {
                // No doctors found — fall back to auto-booking
                document.getElementById('chatContainer').style.display = 'block';
                this.addMessage('ai', "⚠️ No doctors found for your condition. Booking automatically...");
                await this.createAppointment();
                return;
            }

            // Build sub header
            let subText = `Based on your symptoms`;
            if (matchedSpec) subText += ` (${matchedSpec})`;
            if (data.preferredTime) subText += ` · Preferred: ${data.preferredTime}`;
            document.getElementById('doctorSelectionSub').textContent = subText;

            // Render cards — top card is recommended (AI's first pick)
            const grid = document.getElementById('doctorCards');
            grid.innerHTML = doctors.map((doc, idx) => {
                const stars = this._renderStars(doc.rating);
                const slotText = doc.nextSlot
                    ? `Next slot: ${doc.nextSlot.appointmentDate} at ${doc.nextSlot.appointmentTime}`
                    : 'Availability to be confirmed';
                const isRec = idx === 0;
                return `<div class="doctor-card${isRec ? ' recommended' : ''}" onclick="window._selectDoctor(${idx})">
                    <div class="doctor-card-left">
                        <div class="doctor-avatar">👨‍⚕️</div>
                        <div class="doctor-info">
                            <div class="doctor-name">Dr. ${doc.name}</div>
                            <div class="doctor-spec">${doc.specialization}</div>
                            <div class="doctor-slot${doc.nextSlot ? '' : ' no-slot'}">${slotText}</div>
                        </div>
                    </div>
                    <div class="doctor-card-right">
                        <div class="star-rating">${stars}<span class="rating-num">${doc.rating.toFixed(1)}</span></div>
                        ${isRec ? '<span class="rec-badge">Recommended</span>' : ''}
                        <button class="select-btn">Select</button>
                    </div>
                </div>`;
            }).join('');

            document.getElementById('doctorSelectionPanel').style.display = 'block';

            // Store doctors list so the click handler can access it
            this._doctorList = doctors;

            // Global click handler
            window._selectDoctor = async (idx) => {
                const chosen = this._doctorList[idx];
                document.getElementById('doctorSelectionPanel').style.display = 'none';
                await this.showDoctorCalendar(chosen);
            };

        } catch (err) {
            console.error('❌ showDoctorSelection error:', err);
            document.getElementById('loadingPanel').style.display = 'none';
            document.getElementById('chatContainer').style.display = 'block';
            this.addMessage('ai', '⚠️ Could not load doctors. Booking automatically...');
            await this.createAppointment();
        }
    }

    _renderStars(rating) {
        const full  = Math.floor(rating);
        const half  = (rating - full) >= 0.3 ? 1 : 0;
        const empty = 5 - full - half;
        return '★'.repeat(full).split('').map(() => '<span class="star filled">★</span>').join('') +
               (half ? '<span class="star half">★</span>' : '') +
               '☆'.repeat(empty).split('').map(() => '<span class="star empty">☆</span>').join('');
    }

    // In the createAppointment method of EnhancedAppointmentCreator class, replace the hardcoded knowledge base section:

    async createAppointment() {
        document.getElementById('loadingPanel').style.display = 'block';

        try {
            // Load knowledge base from file instead of hardcoded string
            console.log('📚 Loading knowledge base from file...');
            const knowledgeBaseResponse = await fetch(`${window.location.origin}/api/knowledge-base`);
            
            if (!knowledgeBaseResponse.ok) {
                throw new Error('Failed to load knowledge base from file');
            }
            
            const knowledgeBaseData = await knowledgeBaseResponse.json();
            
            if (!knowledgeBaseData.success) {
                throw new Error(knowledgeBaseData.error || 'Failed to load knowledge base');
            }
            
            const doctors = knowledgeBaseData.doctors;
            console.log(`✅ Loaded ${doctors.length} doctors from knowledge base file`);
            
            const data = this.llmInterface.patientData;

            let selectedDoctor = null;
            let matchingConfidence = 0.0;
            let appointmentSlot = null;
            let specializationReason = '';

            // Check for preferred doctor first
            if (data.preferredDoctor && data.preferredDoctor.toLowerCase() !== "n/a" && 
                data.preferredDoctor.toLowerCase() !== "no preference" && 
                data.preferredDoctor.toLowerCase() !== "none") {
                
                const preferredDoctor = doctors.find(doc => 
                    doc.name.toLowerCase().includes(data.preferredDoctor.toLowerCase())
                );
                if (preferredDoctor) {
                    selectedDoctor = preferredDoctor;
                    matchingConfidence = 1.0;
                }
            }

            // In app.js - Replace the doctor matching section in createAppointment() method:

        // Use AI matching if no preferred doctor
        if (!selectedDoctor) {
            const availableSpecializations = knowledgeBaseData.availableSpecializations;
            const knowledgeBaseString = knowledgeBaseData.knowledgeBaseString;
            
            const matchResult = await this.matcher.matchSpecializationWithGemini(
                data.symptomsSummary || data.symptoms, availableSpecializations, knowledgeBaseString, data.preferredTime
            );

            // Find the doctor by name from the enhanced result
            if (matchResult.doctorName) {
                selectedDoctor = doctors.find(doc => 
                    doc.name.toLowerCase().includes(matchResult.doctorName.toLowerCase())
                );
                
                if (selectedDoctor) {
                    matchingConfidence = matchResult.confidence;
                    specializationReason = matchResult.reason || '';
                    // Use the appointment slot from Gemini instead of findNextAvailableSlot
                    appointmentSlot = {
                        date: matchResult.appointmentDate,
                        time: matchResult.appointmentTime
                    };
                    console.log(`✅ Selected doctor: Dr. ${selectedDoctor.name} (${selectedDoctor.specialization})`);
                    console.log(`📅 AI-scheduled appointment: ${appointmentSlot.date} at ${appointmentSlot.time}`);
                }
            }
            
            // Fallback logic if AI matching failed
            if (!selectedDoctor) {
                const specialistDoctors = doctors.filter(doc => doc.specialization === matchResult.specialization);
                
                if (specialistDoctors.length > 0) {
                    selectedDoctor = specialistDoctors[0];
                    matchingConfidence = matchResult.confidence;
                    console.log(`✅ Selected doctor: Dr. ${selectedDoctor.name} (${selectedDoctor.specialization})`);
                } else {
                    // Fallback to General Physician
                    const generalDoctors = doctors.filter(doc => 
                        doc.specialization.toLowerCase().includes('general') || 
                        doc.specialization.toLowerCase().includes('physician')
                    );
                    if (generalDoctors.length > 0) {
                        selectedDoctor = generalDoctors[0];
                        matchingConfidence = 0.75;
                        console.log(`⚠️ Fallback to General Physician: Dr. ${selectedDoctor.name}`);
                    } else {
                        selectedDoctor = doctors[0];
                        matchingConfidence = 0.60;
                        console.log(`⚠️ Fallback to first available doctor: Dr. ${selectedDoctor.name}`);
                    }
                }
            }
        }

        // Only call findNextAvailableSlot if we don't have appointment slot from AI
        if (!appointmentSlot) {
            appointmentSlot = this.findNextAvailableSlot(selectedDoctor);
        }

        if (!appointmentSlot.date) {
            throw new Error(`No available slots for Dr. ${selectedDoctor.name}`);
        }

            const appointmentResult = {
                   patient: {
                   name: data.name,
                   age: data.age,
                   gender: data.gender,
                   contact: data.contact || null,
                   email: data.email || null,
                   issues: await this._buildSymptomSummary(this.llmInterface.getConversationSummary(), data.symptoms || ''),
                   diagnosis: data.diagnosis || this._extractLatestDiagnosis() || "Under assessment",
                   symptomsSummary: data.symptomsSummary || ''
                  },

                doctor: {
                    name: selectedDoctor.name,
                    specialization: selectedDoctor.specialization,
                    email: selectedDoctor.email
                },
                appointment: {
                    date: appointmentSlot.date,
                    time: appointmentSlot.time,
                    duration: "15 minutes"
                },
                analysis: {
                    confidence: (matchingConfidence * 100).toFixed(0) + "%",
                    reasoning: specializationReason || "Based on LLM analysis of symptoms and conversation",
                    extractionMethod: data.extractionHistory.length > 0 ? "LLM Enhanced" : "Fallback"
                }
            };

            setTimeout(async () => {
              document.getElementById('loadingPanel').style.display = 'none';

        try {
            // Prefer a phone-number contact for WhatsApp. Try to normalize existing contact/email.
            const rawContact = (data && (data.contact || data.email || appointmentResult.patient.contact)) || '';
            let normalizedContact = this.normalizeContact ? this.normalizeContact(rawContact) : rawContact;

            // If normalizedContact looks like an email (no usable phone), prompt the user to enter phone for WhatsApp
            const looksLikePhone = normalizedContact && /\d/.test(normalizedContact) && normalizedContact.replace(/\D/g, '').length >= 8;

            if (!looksLikePhone) {
                // Ask operator if they want to provide a phone number (so WhatsApp messages can be sent)
                // This keeps the flow automated when phone is available, but asks only when necessary.
                const supplyPhone = confirm('To send WhatsApp confirmations we need the patient phone number. Do you want to enter a phone number now? (Cancel = skip WhatsApp)');
                if (supplyPhone) {
                const entered = prompt('Enter patient phone number (include country code e.g. +91 or local 10-digit):');
                if (entered) {
                    const normalizedEntered = this.normalizeContact ? this.normalizeContact(entered) : entered;
                    if (normalizedEntered) normalizedContact = normalizedEntered;
                }
            }
        }

        // Update appointmentResult patient contact to the chosen normalized contact (if any)
        appointmentResult.patient.contact = normalizedContact || appointmentResult.patient.contact || null;

        // Build booking payload and call booking endpoint on server
        // Build booking payload and call booking endpoint on server
               const bookingPayload = {
                    doctorName: selectedDoctor.name,
                    preferredTime: data.preferredTime || null,
                    requestedDate: this._selectedSlot ? this._selectedSlot.date : null,
                    requestedTime: this._selectedSlot ? this._selectedSlot.time : null,
                    patient: {
                            name: appointmentResult.patient.name,
                            age: appointmentResult.patient.age,
                            gender: appointmentResult.patient.gender,
                            contact: appointmentResult.patient.contact,
                            issues: appointmentResult.patient.issues,
                            diagnosis: appointmentResult.patient.diagnosis,
                            symptomsSummary: data.symptomsSummary || ''
                                }
                        };

                console.log('📤 Sending booking request to server:', bookingPayload);

                // Make sure server URL is correct
                const apiUrl = `${window.location.origin}/api/book-appointment`;
                console.log('🌐 Using backend API:', apiUrl);

                let resp;
                try {
                    resp = await fetch(apiUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(bookingPayload)
                    });
                    } catch (err) {
                         console.error('❌ Failed to reach backend:', err);
                        this.addMessage('ai', '❌ Unable to reach backend server. Make sure backend is running.');
                    return;
                    }

                console.log('📩 Received booking response with status:', resp.status);

                    let json = null;
                     try {
                         json = await resp.json();
                        console.log('🧾 Booking response JSON:', json);
                        } catch (e) {
                                console.warn('⚠️ Booking response not valid JSON:', e);
                        this.addMessage('ai', '⚠️ Received invalid response from server.');
                    }


        if (resp.ok && json && json.success && json.appointment) {
            // Server stored and returned appointment — merge server result
            console.log('✅ Booking saved on server:', json.appointment);
            appointmentResult.id = json.appointment.id || appointmentResult.id;
            appointmentResult.createdAt = json.appointment.createdAt || appointmentResult.createdAt;
            // server uses appointmentDate/appointmentTime fields
            appointmentResult.appointment.date = json.appointment.appointmentDate || appointmentResult.appointment.date;
            appointmentResult.appointment.time = json.appointment.appointmentTime || appointmentResult.appointment.time;

            this.addMessage('ai', '✅ Appointment booked and saved on server.');
        } else {
            console.warn('⚠️ Server booking failed or returned error:', json || resp.statusText);
            this.addMessage('ai', '⚠️ Could not book appointment on server; falling back to local update & email.');

            // fallback: keep local appointmentSlot and still attempt to update knowledge base
            try {
                await this.updateKnowledgeBaseFile(selectedDoctor.name, appointmentSlot.date, appointmentSlot.time);
            } catch (e) {
                console.warn('⚠️ Fallback knowledge base update failed:', e);
            }
        }

        // Send email notification to doctor in both success & fallback cases
        try {
            await this.sendEmailNotification(appointmentResult);
        } catch (e) {
            console.warn('⚠️ sendEmailNotification failed:', e);
        }

        // Finally show result in UI
        this.showAppointmentResult(appointmentResult);

    } catch (err) {
        console.error('❌ Error during final booking flow:', err);

        // Final fallback (old flow)
        try {
            await this.sendEmailNotification(appointmentResult);
        } catch (e) {
            console.warn('⚠️ sendEmailNotification fallback failed:', e);
        }
        try {
            await this.updateKnowledgeBaseFile(selectedDoctor.name, appointmentSlot.date, appointmentSlot.time);
        } catch (e) {
            console.warn('⚠️ updateKnowledgeBaseFile fallback failed:', e);
        }
        this.showAppointmentResult(appointmentResult);
        }
       }, 2000);


        } catch (error) {
            console.error('Error creating appointment:', error);
            document.getElementById('loadingPanel').style.display = 'none';
            this.addMessage('ai', `❌ Error creating appointment: ${error.message}`);
        }
    }



    // Send email notification to doctor
   // Send email notification to doctor
   async sendEmailNotification(appointmentResult) {
       try {
        console.log('📧 Sending email notification to doctor...');

        const response = await fetch(`${window.location.origin}/api/send-appointment-email`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ appointmentData: appointmentResult })
        });

        const emailResult = await response.json();

        if (emailResult.success) {
            console.log('✅ Email notification sent successfully to doctor');
            this.addMessage('ai', `✅ Email notification sent to Dr. ${appointmentResult.doctor.name}`);

            appointmentResult.emailNotification = {
                sent: true,
                sentAt: emailResult.emailResult?.sentAt || new Date().toISOString(),
                messageId: emailResult.emailResult?.messageId || null
            };
        } else {
            console.error('❌ Failed to send email notification:', emailResult.error);
            this.addMessage('ai', `⚠️ Appointment created but email notification failed. Please contact the doctor directly.`);

            appointmentResult.emailNotification = {
                sent: false,
                error: emailResult.error || 'Unknown error'
            };
        }
    } catch (error) {
        console.error('❌ Error sending email notification:', error);
        this.addMessage('ai', `⚠️ Appointment created but email notification failed. Please contact the doctor directly.`);

        appointmentResult.emailNotification = {
            sent: false,
            error: error.message
        };
       }
     }

    // Add new method to update knowledge base file after appointment booking
    async updateKnowledgeBaseFile(doctorName, appointmentDate, appointmentTime) {
        try {
            console.log('🔄 Updating knowledge base file with new appointment...');
            
            const response = await fetch(`${window.location.origin}/api/update-knowledge-base`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    doctorName: doctorName,
                    appointmentDate: appointmentDate,
                    appointmentTime: appointmentTime
                })
            });

            const updateResult = await response.json();
            
            if (updateResult.success) {
                console.log('✅ Knowledge base file updated successfully');
                this.addMessage('ai', `✅ Knowledge base updated with Dr. ${doctorName}'s appointment`);
            } else {
                console.error('❌ Failed to update knowledge base:', updateResult.error);
                this.addMessage('ai', `⚠️ Appointment created but knowledge base update failed`);
            }

        } catch (error) {
            console.error('❌ Error updating knowledge base file:', error);
            this.addMessage('ai', `⚠️ Appointment created but knowledge base update failed`);
        }
    }
    // Normalize contact: return phone-like string or email string, or null if none
    normalizeContact(raw) {
    if (!raw) return null;
    const s = String(raw).trim();

    

    // Extract phone-like digits
    const digits = s.replace(/\D/g, '');
    // Accept 8-15 digits (loose), prefer returning original raw string so server formatting will handle it
    if (digits.length >= 8 && digits.length <= 15) {
        return s;
    }

    // If too long, return last 10 digits as last-resort (useful for pasted numbers with country code + formatting)
    if (digits.length > 15) {
        return digits.slice(-10);
    }

    return null;
}

    parseKnowledgeBase(knowledgeBaseText) {
        console.log("📋 Parsing knowledge base...");
        const doctors = [];
        const lines = knowledgeBaseText.trim().split('\n');

        for (const line of lines) {
            if (line.includes('Doctor name') && line.includes('Doctor specialization')) {
                const nameMatch = line.match(/Doctor name\s*:\s*([^,]+)/);
                const specMatch = line.match(/Doctor specialization\s*:\s*([^,]+)/);
                const emailMatch = line.match(/Doctor email\s*:\s*([^,]+)/);
                const availMatch = line.match(/Availability\s*:\s*(.+)/);

                if (nameMatch && specMatch && emailMatch && availMatch) {
                    const doctor = {
                        name: nameMatch[1].trim(),
                        specialization: specMatch[1].trim(),
                        email: emailMatch[1].trim(),
                        availability: availMatch[1].trim()
                    };
                    doctors.push(doctor);
                    console.log(`✅ Found doctor: Dr. ${doctor.name} (${doctor.specialization})`);
                }
            }
        }

        console.log(`📊 Total doctors loaded: ${doctors.length}`);
        return doctors;
    }

    findNextAvailableSlot(_doctor) {
        // Simple implementation - returns tomorrow at 10:00 AM
        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 1);
        
        return {
            date: tomorrow.toISOString().split('T')[0],
            time: "10:00"
        };
    }

    // Call backend to extract a clean comma-separated symptom list from the conversation
    async _buildSymptomSummary(conversationText, symptoms) {
        try {
            const res = await fetch(`${window.location.origin}/api/extract-symptoms`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conversationText, symptoms })
            });
            const data = await res.json();
            return data.symptoms || symptoms || 'Not captured';
        } catch (_) {
            return symptoms || 'Not captured';
        }
    }

    // Scan conversation history in reverse to find the chatbot's latest diagnosis statement
    _extractLatestDiagnosis() {
        const history = this.llmInterface.conversationHistory;
        for (let i = history.length - 1; i >= 0; i--) {
            const msg = history[i];
            if (msg.role !== 'assistant') continue;
            const match = msg.content.match(/diagnosis:\s*(.+?)(?:\n|$)/i);
            if (match) {
                const candidate = match[1].trim();
                const validated = this.llmInterface.validateDiagnosis(candidate);
                if (validated) return validated;
            }
        }
        return null;
    }

    // Find the showAppointmentResult function in app.js and update the patient details section

showAppointmentResult(appointmentData) {
    document.getElementById('chatContainer').style.display = 'none';
    document.getElementById('resultPanel').style.display = 'block';

    const detailsHTML = `
        <div style="margin-bottom: 20px;">
            <h4>👤 Patient Details:</h4>
            <p><strong>Name:</strong> ${appointmentData.patient.name}</p>
            <p><strong>Age:</strong> ${appointmentData.patient.age}</p>
            <p><strong>Gender:</strong> ${appointmentData.patient.gender}</p>
            <p><strong>Contact:</strong> ${appointmentData.patient.contact || 'Not provided'}</p>
            <p><strong>Symptoms:</strong> ${appointmentData.patient.issues || 'Not captured'}</p>
            <p><strong>Preliminary Diagnosis:</strong> ${appointmentData.patient.diagnosis || 'Under assessment'}</p>
        </div>

        <div style="margin-bottom: 20px;">
            <h4>👩‍⚕️ Doctor Details:</h4>
            <p><strong>Doctor:</strong> ${appointmentData.doctor.name.startsWith('Dr.') ? appointmentData.doctor.name : 'Dr. ' + appointmentData.doctor.name}</p>
            <p><strong>Specialization:</strong> ${appointmentData.doctor.specialization}</p>
            <p><strong>Email:</strong> ${appointmentData.doctor.email}</p>
        </div>

        <div style="margin-bottom: 20px;">
            <h4>📅 Appointment Details:</h4>
            <p><strong>Date:</strong> ${appointmentData.appointment.date}</p>
            <p><strong>Time:</strong> ${appointmentData.appointment.time}</p>
            <p><strong>Duration:</strong> ${appointmentData.appointment.duration}</p>
        </div>

        <div style="margin-bottom: 20px;">
            <h4>🤖 AI Analysis:</h4>
            <p><strong>Method:</strong> ${appointmentData.analysis.extractionMethod}</p>
            <p><strong>Confidence:</strong> ${appointmentData.analysis.confidence}</p>
            <p><strong>Reasoning:</strong> ${appointmentData.analysis.reasoning}</p>
        </div>

        <div style="margin-bottom: 20px;">
            <h4>📧 Email Notification:</h4>
            ${appointmentData.emailNotification?.sent ? 
                `<p style="color: #28a745;">✅ Email sent successfully to Dr. ${appointmentData.doctor.name}</p>
                 <p><strong>Sent at:</strong> ${new Date(appointmentData.emailNotification.sentAt).toLocaleString()}</p>
                 <p><strong>Message ID:</strong> ${appointmentData.emailNotification.messageId}</p>` :
                `<p style="color: #dc3545;">❌ Email notification failed</p>
                 <p><strong>Error:</strong> ${appointmentData.emailNotification?.error || 'Unknown error'}</p>
                 <p style="color: #6c757d;">Please contact the doctor directly at ${appointmentData.doctor.email}</p>`
            }
        </div>

        <div style="text-align: center; margin-top: 30px;">
            <button onclick="restartBooking()" style="background: #4facfe; color: white; border: none; padding: 15px 30px; border-radius: 8px; font-size: 16px; cursor: pointer;">
                Book Another Appointment
            </button>
            <button onclick="cancelAppointment('${appointmentData.id || ''}','${(appointmentData.patient.contact||'').replace(/\D/g,'').slice(-4)}')"
                style="background: #ef4444; color: white; border: none; padding: 15px 20px; border-radius: 8px; font-size: 16px; cursor: pointer; margin-left: 10px;"
                ${appointmentData.id ? '' : 'disabled'}>
                Cancel
            </button>
        </div>

        <div id="ratingSection" style="margin-top:24px;text-align:center;background:#f8fafc;border-radius:10px;padding:16px;">
            <p style="font-weight:600;margin-bottom:8px">⭐ Rate your booking experience</p>
            <div id="starRow" style="font-size:1.8rem;cursor:pointer;user-select:none">${[1,2,3,4,5].map(n=>`<span onclick="submitRating(${n},'${appointmentData.id||''}')">☆</span>`).join('')}</div>
            <p id="ratingMsg" style="font-size:0.85rem;color:#6b7280;margin-top:4px"></p>
        </div>

        <div id="queueSection" style="margin-top:16px;text-align:center;background:#eef2ff;border-radius:10px;padding:14px;display:none">
            <p id="queuePosition" style="font-weight:600;font-size:1rem;color:#4f46e5">Checking your queue position…</p>
        </div>
    `;

    document.getElementById('appointmentDetails').innerHTML = detailsHTML;

    // Start polling queue position
    if (appointmentData.id && appointmentData.doctor && appointmentData.appointment) {
        pollQueuePosition(appointmentData.id, appointmentData.doctor.name, appointmentData.appointment.date);
    }
}
}


// ── Voice Input (STT) ─────────────────────────────────────────────────────────
let _recognition = null;
let _voiceActive = false;

function initVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const btn = document.getElementById('voiceToggleBtn');
    if (!SpeechRecognition) {
        if (btn) { btn.title = 'Voice not supported in this browser'; btn.style.opacity = '0.4'; }
        return;
    }
    _recognition = new SpeechRecognition();
    _recognition.continuous = false;
    _recognition.interimResults = false;
    _recognition.lang = 'en-IN';

    _recognition.onresult = (e) => {
        const transcript = e.results[0][0].transcript;
        const confidence = e.results[0][0].confidence;
        const input = document.getElementById('userInput') || document.getElementById('messageInput');
        if (input) {
            input.value = transcript;
            input.dispatchEvent(new Event('input'));
            if (confidence > 0.65) {
                // Auto-submit after short delay so user can see what was recognised
                setTimeout(() => {
                    const form = input.closest('form');
                    const sendBtn = document.getElementById('sendBtn') || document.getElementById('sendButton');
                    if (sendBtn) sendBtn.click();
                    else if (form) form.requestSubmit();
                }, 600);
            }
        }
        stopVoice();
    };

    _recognition.onerror = () => stopVoice();
    _recognition.onend   = () => stopVoice();

    if (btn) btn.addEventListener('click', toggleVoice);
}

function toggleVoice() {
    if (_voiceActive) stopVoice();
    else startVoice();
}

function startVoice() {
    if (!_recognition) return;
    _voiceActive = true;
    _recognition.start();
    const btn = document.getElementById('voiceToggleBtn');
    if (btn) { btn.style.background = '#ef4444'; btn.title = 'Listening… click to stop'; }
}

function stopVoice() {
    if (!_recognition) return;
    _voiceActive = false;
    try { _recognition.stop(); } catch (_) {}
    const btn = document.getElementById('voiceToggleBtn');
    if (btn) { btn.style.background = ''; btn.title = 'Click to speak'; }
}

// ── Rating submission ─────────────────────────────────────────────────────────
async function submitRating(score, appointmentId) {
    if (!appointmentId) return;
    const stars = document.getElementById('starRow');
    const msg   = document.getElementById('ratingMsg');
    if (stars) stars.innerHTML = [1,2,3,4,5].map(n => `<span style="color:${n<=score?'#f59e0b':'#d1d5db'}">${n<=score?'★':'☆'}</span>`).join('');
    try {
        const res = await fetch(`/api/appointments/${appointmentId}/rate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ score })
        });
        const data = await res.json();
        if (data.success) {
            if (msg) msg.textContent = `Thanks for rating! Your ${score}★ helps others choose.`;
            if (stars) stars.style.pointerEvents = 'none';
        } else {
            if (msg) msg.textContent = data.error || 'Could not submit rating.';
        }
    } catch (e) {
        if (msg) msg.textContent = 'Rating failed — please try again.';
    }
}

// ── Queue position polling ────────────────────────────────────────────────────
let _queueInterval = null;

async function pollQueuePosition(appointmentId, doctorName, date) {
    const section = document.getElementById('queueSection');
    const posEl   = document.getElementById('queuePosition');
    if (!section || !posEl || !doctorName || !date) return;
    section.style.display = 'block';

    async function check() {
        try {
            const res  = await fetch(`/api/queue/${encodeURIComponent(doctorName)}/${date}`);
            const data = await res.json();
            if (!data.success) return;
            const entry = data.queue.find(q => q.appointmentId === appointmentId);
            if (entry) {
                posEl.textContent = `You are #${entry.position} in queue · Estimated wait: ${(entry.position - 1) * 15} min`;
            } else {
                posEl.textContent = `Queue: ${data.totalConfirmed} confirmed today`;
            }
        } catch (_) {}
    }
    check();
    _queueInterval = setInterval(check, 60000);
}

// ── Patient cancel ────────────────────────────────────────────────────────────
async function cancelAppointment(appointmentId, last4) {
    if (!appointmentId) { alert('Appointment ID not available.'); return; }
    const confirm = window.confirm('Are you sure you want to cancel this appointment?');
    if (!confirm) return;
    const contact = last4 || prompt('Enter last 4 digits of your contact number to verify:');
    if (!contact) return;
    try {
        const res = await fetch(`/api/appointments/${appointmentId}/cancel`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ last4: String(contact).slice(-4) })
        });
        const data = await res.json();
        if (data.success) {
            alert('✅ Appointment cancelled successfully.');
            if (_queueInterval) clearInterval(_queueInterval);
        } else {
            alert('❌ ' + (data.error || 'Could not cancel appointment.'));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// Global variables
let appointmentSystem = null;
let selectedProvider = 'gemini';

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Enhanced Medical Appointment System - LLM Information Extraction');
    console.log('🔗 Backend URL:', window.location.origin);
    
    // Automatically set Gemini as the provider
    selectedProvider = 'gemini';
    console.log('🤖 Auto-selected provider: Gemini AI');
    
    // Check backend connectivity
    checkBackendConnection();

    // Init voice STT
    initVoiceInput();

    // Start button
    document.getElementById('startBtn').addEventListener('click', async function() {
        console.log('▶️ Starting LLM-enhanced appointment booking with provider:', selectedProvider);

        // Hide setup panel and show chat
        document.getElementById('setupPanel').style.display = 'none';
        document.getElementById('chatContainer').style.display = 'flex';

        // Initialize appointment system
        appointmentSystem = new EnhancedAppointmentCreator(selectedProvider, 'demo-key');
        appointmentSystem.initVoiceToggle();

        try {
            await appointmentSystem.chatWithPatient();
        } catch (error) {
            console.error('❌ Error starting chat:', error);
            alert('Error starting the appointment system. Please try again.');
            document.getElementById('setupPanel').style.display = 'block';
            document.getElementById('chatContainer').style.display = 'none';
        }
    });

    // Send message
    document.getElementById('sendBtn').addEventListener('click', sendMessage);
    document.getElementById('userInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') { sendMessage(); }
    });

    // Quick-book button
    const quickBookBtn = document.getElementById('quickBookBtn');
    if (quickBookBtn) {
        quickBookBtn.addEventListener('click', () => {
            if (appointmentSystem) appointmentSystem.skipToBooking();
        });
    }

    // ── Upload panel logic ──
    const attachBtn       = document.getElementById('attachBtn');
    const uploadPanel     = document.getElementById('uploadPanel');
    const reportTypeSelect= document.getElementById('reportTypeSelect');
    const fileInput       = document.getElementById('fileInput');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const uploadConfirmBtn= document.getElementById('uploadConfirmBtn');
    const uploadCancelBtn = document.getElementById('uploadCancelBtn');

    const UPLOAD_ICONS = {
        'Report':      '📄',
        'X-Ray':       '🦴',
        'MRI':         '🧠',
        'Ultrasound':  '🔊',
        'Prescription':'💊',
        'Other':       '📎'
    };

    function closeUploadPanel() {
        uploadPanel.style.display = 'none';
        attachBtn.classList.remove('active');
        reportTypeSelect.value = '';
        fileInput.value = '';
        fileNameDisplay.textContent = 'No file chosen';
        uploadConfirmBtn.disabled = true;
    }

    function maybeEnableAttach() {
        uploadConfirmBtn.disabled = !(reportTypeSelect.value && fileInput.files.length > 0);
    }

    attachBtn.addEventListener('click', () => {
        const isOpen = uploadPanel.style.display !== 'none';
        if (isOpen) {
            closeUploadPanel();
        } else {
            uploadPanel.style.display = 'block';
            attachBtn.classList.add('active');
        }
    });

    reportTypeSelect.addEventListener('change', maybeEnableAttach);

    fileInput.addEventListener('change', () => {
        fileNameDisplay.textContent = fileInput.files[0]?.name || 'No file chosen';
        maybeEnableAttach();
    });

    uploadCancelBtn.addEventListener('click', closeUploadPanel);

    uploadConfirmBtn.addEventListener('click', async () => {
        const type = reportTypeSelect.value;
        const file = fileInput.files[0];
        if (!type || !file) return;

        const messagesContainer = document.getElementById('chatMessages');

        // ── Symptom Photo: send to vision model for analysis ──────────────────
        if (type === 'SymptomPhoto') {
            closeUploadPanel();
            // Show image thumbnail in chat as a user message
            const reader = new FileReader();
            reader.onload = async (e) => {
                const dataUrl = e.target.result;
                const base64  = dataUrl.split(',')[1]; // strip "data:image/...;base64,"

                // Show thumbnail bubble
                const thumbBubble = document.createElement('div');
                thumbBubble.className = 'message user';
                thumbBubble.innerHTML = `
                    <div style="display:flex;flex-direction:column;gap:0.4rem;align-items:flex-end">
                        <img src="${dataUrl}" alt="Symptom photo"
                             style="max-width:200px;max-height:200px;border-radius:8px;border:1px solid #e5e7eb;">
                        <span style="font-size:0.75rem;color:#9ca3af">📷 Analysing image…</span>
                    </div>`;
                messagesContainer.appendChild(thumbBubble);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;

                // Show typing indicator
                const typingBubble = document.createElement('div');
                typingBubble.className = 'message ai';
                typingBubble.innerHTML = '<span style="color:#9ca3af">🤖 Analysing your photo…</span>';
                messagesContainer.appendChild(typingBubble);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;

                try {
                    const resp = await fetch('/api/analyze-image', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: base64, context: 'Patient submitted a symptom photo during appointment booking.' })
                    });
                    const data = await resp.json();

                    if (data.error) {
                        typingBubble.innerHTML = `⚠️ Image analysis unavailable: ${data.hint || data.error}`;
                        return;
                    }

                    const description = data.description;

                    // Replace typing indicator with AI response
                    typingBubble.innerHTML = `<strong>📷 Image analysis:</strong><br>${description.replace(/\n/g, '<br>')}`;

                    // Update the thumbnail caption
                    thumbBubble.querySelector('span').textContent = '📷 Symptom photo attached';

                    // Feed description into the conversation as a user message
                    // so the LLM and differential can use it
                    if (appointmentSystem) {
                        const summaryMsg = `[Patient uploaded a symptom photo. Visual observation: ${description}]`;
                        appointmentSystem.llmInterface.conversationHistory.push({
                            role: 'user', content: summaryMsg
                        });
                        // If symptoms not yet captured, use image description as seed
                        const pd = appointmentSystem.llmInterface.patientData;
                        if (!pd.symptoms) {
                            pd.symptoms = description.split('.')[0].trim();
                            console.log('📷 Symptoms seeded from image:', pd.symptoms);
                        }

                        // Immediately update differential using the image description
                        // so the text model starts reasoning without waiting for next message
                        if (pd.hasBasicInfo()) {
                            try {
                                const convText = appointmentSystem.llmInterface.getConversationSummary();
                                const diffRes = await fetch('/api/update-differential', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({
                                        conversationText: convText,
                                        symptoms: pd.symptoms,
                                        gender: pd.gender,
                                        currentDifferential: pd.differential
                                    })
                                });
                                const diffData = await diffRes.json();
                                if (diffData.differential && diffData.differential.length) {
                                    pd.differential = diffData.differential;
                                    console.log('🧬 Differential updated from image:', diffData.differential.map(d =>
                                        `${d.condition}(p=${d.probability}, for=${(d.evidence_for||[]).length}, against=${(d.evidence_against||[]).length})`
                                    ).join(' | '));
                                    // Show a follow-up question from the new differential
                                    const nextQ = diffData.differential
                                        .map(d => d.discriminating_question)
                                        .find(q => q && !pd.askedQuestions.includes(q));
                                    if (nextQ) {
                                        const qBubble = document.createElement('div');
                                        qBubble.className = 'message ai';
                                        qBubble.textContent = nextQ;
                                        messagesContainer.appendChild(qBubble);
                                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                                        appointmentSystem.llmInterface.conversationHistory.push({
                                            role: 'assistant', content: nextQ
                                        });
                                        pd.askedQuestions.push(nextQ);
                                    }
                                }
                            } catch (e) {
                                console.warn('⚠️ Post-image differential update failed:', e.message);
                            }
                        }
                    }
                } catch (err) {
                    typingBubble.innerHTML = `⚠️ Could not analyse image: ${err.message}`;
                }
            };
            reader.readAsDataURL(file);
            return;
        }

        // ── Other document types: just show attachment bubble ─────────────────
        const bubble = document.createElement('div');
        bubble.className = 'message attachment';
        bubble.innerHTML = `
            <span class="attachment-icon">${UPLOAD_ICONS[type] || '📎'}</span>
            <span class="attachment-info">
                <span class="attachment-type">${type}</span>
                <span class="attachment-name">${file.name}</span>
            </span>`;
        messagesContainer.appendChild(bubble);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Store on patientData for later use
        if (appointmentSystem && appointmentSystem.llmInterface) {
            if (!appointmentSystem.llmInterface.patientData.attachments) {
                appointmentSystem.llmInterface.patientData.attachments = [];
            }
            appointmentSystem.llmInterface.patientData.attachments.push({ type, name: file.name });
        }

        console.log(`📎 Attachment recorded: ${type} — ${file.name}`);
        closeUploadPanel();
    });
});

async function checkBackendConnection() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('✅ Backend connected:', data.message);
        
        // Update UI to show backend status
        const infoBox = document.querySelector('.info-box');
        if (infoBox) {
            // Safely check for email and gemini status
            const emailStatus = (data.emailService && data.emailService.configured) ? '✅ Configured' : '⚠️ Not Configured';
            const geminiStatus = data.geminiAvailable ? '✅ Available' : '⚠️ Not Available';
            
            infoBox.innerHTML = `
                <p><strong>✅ Backend Status:</strong> Connected to server</p>
                <p><strong>🧠 LLM Extraction:</strong> Advanced AI-powered information extraction enabled</p>
                <p><strong>🔑 Gemini API:</strong> ${geminiStatus}</p>
                <p><strong>📧 Email Service:</strong> ${emailStatus}</p>
                <p><strong>⚡ Server:</strong> ${data.message}</p>
            `;
            
            // Only change styling if there are issues
            if (!data.geminiAvailable || !(data.emailService && data.emailService.configured)) {
                infoBox.style.backgroundColor = '#fff3cd';
                infoBox.style.borderLeft = '4px solid #ffc107';
            } else {
                // Reset to default styling if everything is configured
                infoBox.style.backgroundColor = '';
                infoBox.style.borderLeft = '';
            }
        }
    } catch (error) {
        console.warn('⚠️ Backend not available, will use fallback mode');
        
        // Update UI to show fallback mode
        const infoBox = document.querySelector('.info-box');
        if (infoBox) {
            infoBox.innerHTML = `
                <p><strong>⚠️ Backend Status:</strong> Using fallback extraction mode</p>
                <p><strong>💡 Note:</strong> Start the Node.js server for full functionality</p>
                <p><strong>🔧 Command:</strong> <code>npm run dev</code></p>
                <p><strong>📝 Fallback:</strong> Basic regex patterns will be used</p>
                <p><strong>📧 Email:</strong> Email notifications will not work</p>
            `;
            infoBox.style.backgroundColor = '#f8d7da';
            infoBox.style.borderLeft = '4px solid #dc3545';
        }
    }
}

async function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    
    if (!message || !appointmentSystem) return;

    console.log('📤 Sending message for LLM analysis:', message);
    
    input.value = '';
    input.disabled = true;
    document.getElementById('sendBtn').disabled = true;

    try {
        await appointmentSystem.handleUserMessage(message);
    } catch (error) {
        console.error('❌ Error handling message:', error);
        appointmentSystem.addMessage('ai', 'Sorry, I encountered an error. Please try again.');
    }

    // Only re-enable if booking has NOT started
    if (!appointmentSystem || !appointmentSystem.bookingInProgress) {
        input.disabled = false;
        document.getElementById('sendBtn').disabled = false;
        input.focus();
    }
}

function restartBooking() {
    console.log('🔄 Restarting LLM-enhanced appointment booking');
    
    // Stop any running extraction processes
    if (appointmentSystem) {
        appointmentSystem.stopPeriodicExtraction();
    }
    
    // Reset appointment system but keep provider selection
    appointmentSystem = null;
    
    // Clear chat
    document.getElementById('chatMessages').innerHTML = '';
    document.getElementById('userInput').value = '';
    
    // Reset progress
    ['nameProgress', 'ageProgress', 'genderProgress', 'contactProgress', 'symptomsProgress', 'diagnosisProgress', 'assessmentProgress'].forEach(id => {
        const element = document.getElementById(id);
        element.className = 'progress-item incomplete';
        element.textContent = element.textContent.replace('✅', '❌').replace('🔄', '❌');
    });
    
    // Show setup panel
    document.getElementById('setupPanel').style.display = 'block';
    document.getElementById('chatContainer').style.display = 'none';
    document.getElementById('resultPanel').style.display = 'none';
    document.getElementById('loadingPanel').style.display = 'none';
    
    // Re-check backend connection
    checkBackendConnection();
}

// Handle browser refresh
window.addEventListener('beforeunload', function() {
    if (appointmentSystem) {
        appointmentSystem.stopPeriodicExtraction();
    }
    console.log('👋 Closing Enhanced Medical Appointment System');
});

// Error handling for unhandled promises
window.addEventListener('unhandledrejection', function(event) {
    console.error('🚨 Unhandled promise rejection:', event.reason);
    event.preventDefault();
});