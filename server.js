// server.js - Enhanced Medical Appointment Server with LLM Information Extraction and Email Notifications
// Node.js backend with Gemini AI integration for patient information extraction and doctor email notifications

const express = require('express');
const cors = require('cors');
const path = require('path');

// Doctor auth (Phase 1)
const doctorAuthMiddleware = require('./doctorAuthMiddleware');
const { verifyDoctor, generateToken, getDoctorById, changePassword, updateProfile } = require('./doctorAuth');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const EmailService = require('./emailService'); // Import email service
const KnowledgeBaseReader = require('./knowledgeBaseReader');
const KnowledgeBaseUpdater = require('./updateKnowledgeBase');

const schedule = require('node-schedule');         // for scheduled reminders
const WhatsAppService = require('./whatsappService'); // new file we added
const whatsappService = new WhatsAppService();
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Initialize email service
const emailService = new EmailService();
// Initialize knowledge base services
const knowledgeBaseReader = new KnowledgeBaseReader('./doctors.xlsx'); // or './doctors.csv'
const knowledgeBaseUpdater = new KnowledgeBaseUpdater('./doctors.xlsx'); // or './doctors.csv'

// ---- Added for appointment manager (do not remove) ----
const fsPromises = require('fs').promises;
const fs = require('fs');
const APPT_FILE = path.resolve('./appointments.json');
// in-memory enqueue to serialize booking operations
let bookingQueue = Promise.resolve();
// -------------------------------------------------------

// =============================================================================
// RAG — Embedding-based semantic retrieval from prompts/<SpecialtyName>/*.txt
// Falls back to keyword question banks when embeddings.json is not present.
// =============================================================================
const SPECIALTIES = {
    'Cardiologist':     { keywords: ['chest', 'heart', 'palpitation', 'cardiac', 'angina', 'blood pressure', 'hypertension'] },
    'Dermatologist':    { keywords: ['skin', 'rash', 'itch', 'acne', 'lesion', 'eczema', 'psoriasis', 'hives', 'dermatitis', 'pigment', 'hair loss'] },
    'General Physician':{ keywords: ['stomach', 'abdomen', 'nausea', 'vomit', 'bowel', 'diarrhea', 'constipation', 'bloat', 'gas', 'fever', 'cold', 'flu', 'fatigue', 'tired', 'headache', 'back', 'joint', 'muscle', 'knee', 'shoulder', 'cough', 'breath', 'throat', 'general'] },
    'Urologist':        { keywords: ['urine', 'urinary', 'kidney', 'bladder', 'prostate', 'urination', 'incontinence'] },
    'Gynaecologist':    { keywords: ['period', 'menstrual', 'menstruation', 'pregnancy', 'pregnant', 'ovarian', 'uterus', 'cervix', 'vaginal', 'pelvic', 'contraception'] },
    'Ophthalmologist':  { keywords: ['eye', 'vision', 'sight', 'cataract', 'glaucoma', 'retina', 'blind', 'glasses'] }
};

const PROMPTS_DIR   = path.resolve('./prompts');
const EMBEDDINGS_FILE = path.resolve('./embeddings.json');
const EMBED_MODEL   = 'nomic-embed-text';
const RAG_TOP_K     = 7;   // passages retrieved per query (raised for better coverage)

// ── In-memory stores ──────────────────────────────────────────────────────────
let ragStore = [];          // [{ specialty, file, chunkIndex, text, embedding }]
let ragReady = false;       // true when embeddings.json loaded successfully

const questionBanks = {};   // keyword fallback
const exampleConversations = {};

// ── Cosine similarity (pure JS, no native deps) ───────────────────────────────
function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot   += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
}

// ── Embed a text string via Ollama /api/embeddings ────────────────────────────
async function embedText(text) {
    const res = await axios.post(
        `${process.env.OLLAMA_URL || 'http://localhost:11434'}/api/embeddings`,
        { model: EMBED_MODEL, prompt: text },
        { timeout: 15000 }
    );
    return res.data.embedding; // float[]
}

// ── Load pre-built embeddings.json at startup (async — avoids blocking the event loop) ──
async function loadEmbeddings() {
    if (!fs.existsSync(EMBEDDINGS_FILE)) {
        console.warn('⚠️  embeddings.json not found — falling back to keyword question banks');
        console.warn('   Run: node generate-embeddings.js   to build it');
        loadQuestionBanks();
        return;
    }
    try {
        console.log('⏳ Loading embeddings.json asynchronously (this may take a moment)...');
        const raw = await fsPromises.readFile(EMBEDDINGS_FILE, 'utf8');
        ragStore = JSON.parse(raw);
        ragReady = true;
        const counts = {};
        ragStore.forEach(r => { counts[r.specialty] = (counts[r.specialty] || 0) + 1; });
        console.log('✅ RAG embeddings loaded:', Object.entries(counts).map(([k,v]) => `${k}(${v})`).join(', '));
    } catch (e) {
        console.warn('⚠️ Failed to load/parse embeddings.json:', e.message, '— using keyword fallback');
        loadQuestionBanks();
    }
}

// ── Semantic retrieval: embed the query, score every chunk, return top-K ──────
async function retrieveRelevantContext(symptomsText, gender, topK = RAG_TOP_K) {
    let queryVec;
    try {
        queryVec = await embedText(symptomsText);
    } catch (e) {
        console.warn('⚠️ Embed query failed:', e.message, '— keyword fallback');
        return keywordFallback(symptomsText, gender);
    }

    // Score all chunks and pick top-K
    const scored = ragStore.map(chunk => ({
        ...chunk,
        score: cosineSimilarity(queryVec, chunk.embedding)
    }));
    scored.sort((a, b) => b.score - a.score);

    let topChunks = scored.slice(0, topK * 3); // over-fetch to allow gender filtering

    // Filter gender-inappropriate lines within each chunk text
    const FEMALE_KW = ['period', 'menstrual', 'menstruation', 'pregnancy', 'pregnant', 'ovarian', 'uterus', 'cervix', 'vaginal', 'pelvic', 'contraception'];
    const MALE_KW   = ['prostate', 'erectile', 'testicular'];
    topChunks = topChunks.filter(chunk => {
        const lower = chunk.text.toLowerCase();
        if (gender === 'Male'   && FEMALE_KW.some(kw => lower.includes(kw))) return false;
        if (gender === 'Female' && MALE_KW.some(kw => lower.includes(kw)))   return false;
        return true;
    }).slice(0, topK);

    // Detect specialty from top chunk (most representative)
    const specialty = topChunks[0]?.specialty || detectSpecialty(symptomsText) || 'General';

    const context = topChunks.map(c => c.text).join('\n---\n');
    return { specialty, context, source: 'rag' };
}

// ── Keyword question bank (fallback when no embeddings.json) ──────────────────
function loadQuestionBanks() {
    try {
        for (const [specialtyName] of Object.entries(SPECIALTIES)) {
            const specialtyDir = path.join(PROMPTS_DIR, specialtyName);
            if (!fs.existsSync(specialtyDir)) continue;

            const files = fs.readdirSync(specialtyDir)
                .filter(f => f.endsWith('.txt'))
                .sort()
                .map(f => path.join(specialtyDir, f));
            if (files.length === 0) continue;

            try {
                const lines = fs.readFileSync(files[0], 'utf8').split('\n').filter(l => l.trim());
                const dialogLines = lines.filter(l => l.startsWith('D:') || l.startsWith('P:')).slice(0, 30);
                exampleConversations[specialtyName] = dialogLines.join('\n');
            } catch (_) {}

            const questions = new Set();
            for (const file of files.slice(0, 5)) {
                try {
                    for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
                        if (line.startsWith('D:') && line.includes('?')) {
                            const q = line.replace('D:', '').trim();
                            if (q.length > 15 && q.length < 200) questions.add(q);
                        }
                    }
                } catch (_) {}
            }
            questionBanks[specialtyName] = [...questions].slice(0, 12);
        }
        console.log('✅ Keyword fallback loaded:', Object.entries(questionBanks).map(([k,v]) => `${k}(${v.length})`).join(', '));
    } catch (e) {
        console.warn('⚠️ Could not load question banks:', e.message);
    }
}

function detectSpecialty(symptoms) {
    if (!symptoms) return 'General';
    const lower = symptoms.toLowerCase();
    for (const [name, info] of Object.entries(SPECIALTIES)) {
        if (info.keywords.some(kw => lower.includes(kw))) return name;
    }
    return 'General';
}

const FEMALE_ONLY_KEYWORDS = ['period', 'menstrual', 'menstruation', 'pregnancy', 'pregnant', 'ovarian', 'uterus', 'cervix', 'vaginal', 'pelvic', 'last period', 'contraception'];
const MALE_ONLY_KEYWORDS   = ['prostate', 'erectile', 'testicular'];

function keywordFallback(symptoms, gender) {
    const specialty = detectSpecialty(symptoms);
    let questions = questionBanks[specialty] || questionBanks['General'] || [];
    let example   = exampleConversations[specialty] || exampleConversations['General'] || '';
    if (gender === 'Male') {
        questions = questions.filter(q => !FEMALE_ONLY_KEYWORDS.some(kw => q.toLowerCase().includes(kw)));
        example   = example.split('\n').filter(l => !FEMALE_ONLY_KEYWORDS.some(kw => l.toLowerCase().includes(kw))).join('\n');
    } else if (gender === 'Female') {
        questions = questions.filter(q => !MALE_ONLY_KEYWORDS.some(kw => q.toLowerCase().includes(kw)));
        example   = example.split('\n').filter(l => !MALE_ONLY_KEYWORDS.some(kw => l.toLowerCase().includes(kw))).join('\n');
    }
    const context = example
        ? example
        : questions.slice(0, 8).map((q, i) => `D: ${q}`).join('\n');
    return { specialty, context, source: 'keyword' };
}

// Unified entry point used by the chat handler
async function getSymptomContext(symptoms, gender) {
    if (!symptoms) return { specialty: 'General', context: '', source: 'none' };
    if (ragReady) {
        return retrieveRelevantContext(symptoms, gender);
    }
    return keywordFallback(symptoms, gender);
}

loadEmbeddings().catch(e => {
    console.warn('⚠️ loadEmbeddings failed unexpectedly:', e.message, '— using keyword fallback');
    loadQuestionBanks();
});
// =============================================================================

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));


// Serve static files
app.use(express.static('.'));

const axios = require('axios');

// =============================================================================
// OPTION A: HuggingFace Inference API (DISABLED — model not eligible for free
//           serverless inference, returns 410 Gone)
// -----------------------------------------------------------------------------
// const HF_TOKEN = process.env.HUGGINGFACE_HUB_TOKEN;
// const HF_MODEL = 'medeasy/med-gemma-finetune';
// const HF_API_URL = `https://api-inference.huggingface.co/models/${HF_MODEL}`;
// let hfServiceAvailable = !!HF_TOKEN;
//
// async function checkHFService() {
//     if (!HF_TOKEN) {
//         console.warn('⚠️ HUGGINGFACE_HUB_TOKEN not set — HF inference disabled');
//         hfServiceAvailable = false;
//         return;
//     }
//     try {
//         await axios.post(HF_API_URL,
//             { inputs: 'Hello', parameters: { max_new_tokens: 5 } },
//             { headers: { Authorization: `Bearer ${HF_TOKEN}` }, timeout: 15000 }
//         );
//         hfServiceAvailable = true;
//         console.log(`✅ HuggingFace Inference API available (${HF_MODEL})`);
//     } catch (e) {
//         if (e.response && e.response.status === 503) {
//             hfServiceAvailable = true;
//             console.log(`⏳ HuggingFace model warming up on HF servers`);
//         } else {
//             hfServiceAvailable = false;
//             console.warn(`⚠️ HuggingFace Inference API not available: ${e.message}`);
//         }
//     }
// }
//
// async function callHFModel(prompt, maxLength = 300) {
//     const res = await axios.post(HF_API_URL, {
//         inputs: prompt,
//         parameters: { max_new_tokens: maxLength, temperature: 0.7, return_full_text: false }
//     }, { headers: { Authorization: `Bearer ${HF_TOKEN}` }, timeout: 60000 });
//     const data = res.data;
//     if (Array.isArray(data) && data[0] && data[0].generated_text !== undefined)
//         return data[0].generated_text;
//     if (data && data.generated_text) return data.generated_text;
//     throw new Error('Unexpected HF API response format');
// }
// =============================================================================

// =============================================================================
// OPTION B: Ollama local inference (ACTIVE)
// Requires: brew install ollama && ollama pull gemma2:2b && ollama serve
// Uses ~1.5GB RAM on M1 Mac (4-bit quantized via llama.cpp)
// -----------------------------------------------------------------------------
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'gemma2:2b';
let hfServiceAvailable = false;

async function checkHFService() {
    try {
        const res = await axios.get(`${OLLAMA_URL}/api/tags`, { timeout: 3000 });
        const models = res.data && res.data.models ? res.data.models.map(m => m.name) : [];
        if (models.some(m => m.startsWith('gemma2'))) {
            hfServiceAvailable = true;
            console.log(`✅ Ollama available — using model: ${OLLAMA_MODEL}`);
        } else {
            hfServiceAvailable = false;
            console.warn(`⚠️ Ollama running but ${OLLAMA_MODEL} not found. Run: ollama pull ${OLLAMA_MODEL}`);
        }
    } catch (e) {
        hfServiceAvailable = false;
        console.warn(`⚠️ Ollama not reachable. Run: ollama serve`);
    }
}

// callHFModel: accepts either a string prompt or a messages array
async function callHFModel(promptOrMessages, maxLength = 300) {
    let messages;
    if (Array.isArray(promptOrMessages)) {
        messages = promptOrMessages;
    } else {
        messages = [{ role: 'user', content: promptOrMessages }];
    }
    const res = await axios.post(`${OLLAMA_URL}/api/chat`, {
        model: OLLAMA_MODEL,
        messages: messages,
        stream: false,
        options: { num_predict: maxLength, temperature: 0.7 }
    }, { timeout: 120000 });
    return res.data.message.content;
}
// =============================================================================

// Unified AI text generation: Ollama fallback when Gemini is unavailable
async function generateAIText(promptOrMessages, maxLength = 300) {
    if (hfServiceAvailable) {
        try {
            console.log(`🦙 Using Ollama (${OLLAMA_MODEL})`);
            return await callHFModel(promptOrMessages, maxLength);
        } catch (e) {
            console.warn('⚠️ Ollama call failed:', e.message);
        }
    }
    throw new Error('AI service unavailable. Run: ollama serve && ollama pull gemma2:2b');
}

// Initialize Gemini AI — checked ONCE at startup; permanently disabled if unavailable
let genAI = null;
let model = null;
let geminiDisabled = false;  // set true permanently if startup probe fails

function markGeminiQuotaExhausted() {
    if (!geminiDisabled) {
        geminiDisabled = true;
        console.warn('🚫 Gemini unavailable — permanently falling back to Ollama for this session');
    }
}

function isGeminiAvailable() {
    return genAI !== null && !geminiDisabled;
}

async function initGemini() {
    if (!process.env.GEMINI_API_KEY) {
        console.warn('⚠️ GEMINI_API_KEY not set — using Ollama (gemma2)');
        geminiDisabled = true;
        return;
    }
    try {
        genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
        model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });
        // Probe with a minimal request to confirm the key and quota are valid
        await model.generateContent({ contents: [{ role: 'user', parts: [{ text: 'ping' }] }], generationConfig: { maxOutputTokens: 1 } });
        console.log('✅ Gemini AI available — using Gemini as primary');
    } catch (error) {
        genAI = null;
        model = null;
        geminiDisabled = true;
        console.warn(`⚠️ Gemini unavailable (${error.message}) — falling back to Ollama (gemma2) for this session`);
    }
}

// Check Ollama and probe Gemini once at startup; no further retries
checkHFService();
initGemini();

// Health check endpoint with email service status
app.get('/api/health', (req, res) => {
    const emailStatus = emailService.getStatus();
    
    const status = {
        message: 'Medical Appointment Server is running',
        timestamp: new Date().toISOString(),
        geminiAvailable: !!model,
        apiKey: process.env.GEMINI_API_KEY ? 'Configured' : 'Missing',
        emailService: {
            initialized: emailStatus.initialized,
            configured: emailStatus.configured,
            emailUser: emailStatus.emailUser
        }
    };
    res.json(status);
});

// Enhanced medical chat endpoint — Gemini primary, Ollama fallback
app.post('/api/medical-chat', async (req, res) => {
    try {
        const { message, conversationHistory = [], systemPrompt, patientData } = req.body;

        if (!model && !hfServiceAvailable) {
            return res.json({
                error: 'AI service not available',
                fallback: getFallbackResponse(message, patientData)
            });
        }

        console.log('🤖 AI Chat Request:', {
            message: message.substring(0, 100),
            historyLength: conversationHistory.length,
            patientFields: Object.keys(patientData).filter(key => patientData[key])
        });

        // --- Build enriched system instruction ---
        const extractedFields = Object.keys(patientData).filter(key =>
            patientData[key] && patientData[key] !== "" && patientData[key] !== "N/A" && patientData[key] !== 0
        );
        const missingFields = ['name', 'age', 'gender', 'contact', 'symptoms', 'diagnosis']
            .filter(f => !extractedFields.includes(f));

        const patientContext = extractedFields.length > 0
            ? `\nCOLLECTED SO FAR (DO NOT ASK FOR THESE AGAIN):\n${extractedFields.map(f => `- ${f}: ${patientData[f]}`).join('\n')}\nSTILL NEEDED: ${missingFields.join(', ')}\nOnly ask for fields listed under STILL NEEDED. Never ask for any field already listed under COLLECTED SO FAR.`
            : '\nNOTHING COLLECTED YET. Begin with name, age, and gender together.';

        const gender = patientData.gender || null;
        const genderWarning = gender === 'Male'
            ? '\n⚠️ PATIENT IS MALE. NEVER ask about periods, menstruation, pregnancy, or female-specific conditions.'
            : gender === 'Female'
            ? '\n⚠️ PATIENT IS FEMALE. NEVER ask about prostate or male-specific conditions.'
            : '';

        // --- RAG context: symptom-specific clinical question examples ---
        const symptoms = (patientData.symptoms || '').toLowerCase();
        let questionContext = '';
        if (symptoms) {
            const { specialty, context: ragContext, source } = await getSymptomContext(patientData.symptoms, gender);
            const symptomWarning = `\n⚠️ PATIENT'S SYMPTOMS ARE: "${patientData.symptoms}". ONLY ask follow-up questions about THESE symptoms. Do NOT introduce any other condition.`;
            if (ragContext) {
                const label = source === 'rag'
                    ? `RELEVANT CLINICAL DIALOGUE EXAMPLES FOR ${specialty.toUpperCase()} (semantic match — use as inspiration only):`
                    : `SUGGESTED FOLLOW-UP QUESTIONS FOR ${specialty.toUpperCase()} (only use ones relevant to what THIS patient said):`;
                questionContext = `${symptomWarning}\n${label}\n${ragContext}`;
            } else {
                questionContext = symptomWarning;
            }
        }

        // --- Determine if we are in follow-up phase ---
        const hasSymptoms = !!(patientData.symptoms && patientData.symptoms.trim());
        const allBasicCollected = patientData.name && patientData.age && patientData.gender && patientData.contact && patientData.symptoms;
        const assistantTurns = conversationHistory.filter(m => m.role === 'assistant').length;
        // First ~4 assistant turns = intro + basic info. Follow-ups start after that.
        const followUpsDone = allBasicCollected ? Math.max(0, assistantTurns - 4) : 0;
        const needsMoreFollowUp = hasSymptoms && followUpsDone < 3;

        const followUpInstruction = needsMoreFollowUp
            ? `\n\n🔴 FOLLOW-UP PHASE — YOU MUST ASK MORE QUESTIONS:\n- You have the patient's basic info and symptoms.\n- You have asked ${followUpsDone} follow-up question(s) about the symptoms so far. You need at least 3.\n- Ask ONE specific follow-up question now about: duration, severity (1-10), onset, triggers, associated symptoms, or relevant medical history.\n- DO NOT ask about appointment time yet.\n- DO NOT say "I have sufficient information" yet.\n- DO NOT diagnose yet.`
            : hasSymptoms
            ? `\n\n✅ FOLLOW-UP COMPLETE — You have gathered enough symptom information. You may now:\n- Give a brief preliminary assessment or say "Diagnosis: [assessment]".\n- Ask the patient's preferred appointment time (morning/afternoon/evening).\n- Then say "I have sufficient information to book your appointment now."`
            : '';

        const responseGuidelines = `\n\nRESPONSE GUIDELINES:\n- Ask name, age, and gender TOGETHER in one question if any are missing.\n- NEVER ask for information you already have.\n- ONLY ask about symptoms the patient has mentioned.\n- If the patient says "I don't know" or denies a symptom, accept it and move on.\n- After 3+ follow-up symptom questions, then ask preferred appointment time, then conclude.\n- If you cannot determine a diagnosis, say: "Diagnosis is unknown".\n- Say "I have sufficient information to book your appointment now." ONLY after asking follow-up questions AND appointment time preference.`;

        const systemInstruction = `${systemPrompt}${patientContext}${genderWarning}${questionContext}${followUpInstruction}${responseGuidelines}`;

        let text;

        // --- PRIMARY: Gemini with multi-turn chat ---
        if (isGeminiAvailable()) {
            try {
                const chatModel = genAI.getGenerativeModel({
                    model: 'gemini-2.0-flash',
                    systemInstruction: systemInstruction,
                    generationConfig: { temperature: 0.7, maxOutputTokens: 300 }
                });

                const rawHistory = conversationHistory.slice(-10).map(msg => ({
                    role: msg.role === 'assistant' ? 'model' : 'user',
                    parts: [{ text: msg.content }]
                }));
                const geminiHistory = rawHistory[0]?.role === 'model' ? rawHistory.slice(1) : rawHistory;

                const chat = chatModel.startChat({ history: geminiHistory });
                const result = await chat.sendMessage(message);
                text = result.response.text().trim();
                console.log('✅ Gemini chat response generated');
            } catch (geminiErr) {
                if (geminiErr.message && geminiErr.message.includes('429')) markGeminiQuotaExhausted();
                console.warn('⚠️ Gemini chat failed, trying Ollama fallback:', geminiErr.message);
            }
        }

        // --- FALLBACK: Ollama ---
        if (!text && hfServiceAvailable) {
            const ollamaMessages = buildMedicalChatPrompt(message, conversationHistory, systemPrompt, patientData);
            text = await generateAIText(ollamaMessages);
            console.log('✅ Ollama fallback response generated');
        }

        if (!text) throw new Error('Both Gemini and Ollama unavailable');

        res.json({ text, timestamp: new Date().toISOString() });

    } catch (error) {
        console.error('❌ Error in medical chat:', error);
        res.json({
            error: error.message,
            fallback: getFallbackResponse(req.body.message, req.body.patientData)
        });
    }
});

// New endpoint for patient information extraction
app.post('/api/extract-patient-info', async (req, res) => {
    try {
        const { conversationText, extractionPrompt, currentData } = req.body;

        if (!model && !hfServiceAvailable) {
            return res.json({
                error: 'AI service not available for extraction'
            });
        }

        // Build structured extraction prompt
        const fullExtractionPrompt = buildExtractionPrompt(
            conversationText,
            extractionPrompt,
            currentData
        );

        console.log('🧠 AI Extraction Request:', {
            conversationLength: conversationText.length,
            currentFields: Object.keys(currentData).filter(key => currentData[key])
        });

        let text = '';

        // PRIMARY: Gemini
        if (isGeminiAvailable()) {
            try {
                const extractModel = genAI.getGenerativeModel({
                    model: 'gemini-2.0-flash',
                    generationConfig: { temperature: 0.1, maxOutputTokens: 300 }
                });
                const result = await extractModel.generateContent(fullExtractionPrompt);
                text = result.response.text().trim();
                console.log('✅ Gemini extraction response generated');
            } catch (geminiErr) {
                if (geminiErr.message && geminiErr.message.includes('429')) markGeminiQuotaExhausted();
                console.warn('⚠️ Gemini extraction failed, trying Ollama fallback:', geminiErr.message);
            }
        }

        // FALLBACK: Ollama
        if (!text && hfServiceAvailable) {
            text = await generateAIText(fullExtractionPrompt);
        }

        if (!text) throw new Error('Both Gemini and Ollama unavailable for extraction');

        // Parse JSON response from Gemini
        let extractedInfo;
        try {
            // Clean the response text more thoroughly
            let cleanText = text.replace(/```json\n?|\n?```/g, '').trim();
            
            // Remove any text before the first { and after the last }
            const firstBrace = cleanText.indexOf('{');
            const lastBrace = cleanText.lastIndexOf('}');
            
            if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
                cleanText = cleanText.substring(firstBrace, lastBrace + 1);
            }
            
            extractedInfo = JSON.parse(cleanText);
            console.log('✅ Successfully parsed extraction JSON:', extractedInfo);
            
        } catch (parseError) {
            console.warn('⚠️ Failed to parse JSON from Gemini response:', text.substring(0, 200));
            
            // More aggressive JSON extraction
            const jsonMatches = text.match(/\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}/g);
            if (jsonMatches && jsonMatches.length > 0) {
                try {
                    // Try the largest JSON object found
                    const largestJson = jsonMatches.reduce((a, b) => a.length > b.length ? a : b);
                    extractedInfo = JSON.parse(largestJson);
                    console.log('✅ Recovered JSON from response:', extractedInfo);
                } catch (e) {
                    console.error('❌ All JSON parsing attempts failed');
                    throw new Error('Invalid JSON in response: ' + parseError.message);
                }
            } else {
                throw new Error('No JSON found in response');
            }
        }

        console.log('✅ Extraction successful:', extractedInfo);

        res.json({
            extractedInfo: extractedInfo,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('❌ Error in patient information extraction:', error);
        res.json({
            error: error.message,
            extractedInfo: null
        });
    }
});

// New endpoint for sending appointment notification emails
app.post('/api/send-appointment-email', async (req, res) => {
    try {
        const { appointmentData } = req.body;

        if (!appointmentData) {
            return res.status(400).json({
                success: false,
                error: 'Missing appointment data'
            });
        }

        console.log('📧 Received email request for appointment:', {
            patient: appointmentData.patient?.name,
            doctor: appointmentData.doctor?.name,
            email: appointmentData.doctor?.email,
            date: appointmentData.appointment?.date
        });

        // Send email notification
        const emailResult = await emailService.sendAppointmentNotification(appointmentData);

        if (emailResult.success) {
            console.log('✅ Appointment email sent successfully');
            res.json({
                success: true,
                message: 'Appointment notification email sent successfully',
                emailResult: emailResult
            });
        } else {
            console.error('❌ Failed to send appointment email:', emailResult.error);
            res.status(500).json({
                success: false,
                error: 'Failed to send email notification',
                details: emailResult.error
            });
        }

    } catch (error) {
        console.error('❌ Error in send-appointment-email endpoint:', error);
        res.status(500).json({
            success: false,
            error: 'Internal server error while sending email',
            message: error.message
        });
    }
});

// Test email endpoint
app.post('/api/test-email', async (req, res) => {
    try {
        const { testEmail } = req.body;

        console.log('🧪 Testing email configuration...');
        const testResult = await emailService.sendTestEmail(testEmail);

        if (testResult.success) {
            res.json({
                success: true,
                message: 'Test email sent successfully',
                result: testResult
            });
        } else {
            res.status(500).json({
                success: false,
                error: 'Test email failed',
                details: testResult.error
            });
        }

    } catch (error) {
        console.error('❌ Error in test-email endpoint:', error);
        res.status(500).json({
            success: false,
            error: 'Internal server error during email test',
            message: error.message
        });
    }
});

// Get email service status endpoint
app.get('/api/email-status', (req, res) => {
    const status = emailService.getStatus();
    res.json(status);
});
// New endpoint for sending appointment notification emails
app.post('/api/send-appointment-email', async (req, res) => {
    try {
        const { appointmentData } = req.body;

        if (!appointmentData) {
            return res.status(400).json({
                success: false,
                error: 'Missing appointment data'
            });
        }

        console.log('📧 Received email request for appointment:', {
            patient: appointmentData.patient?.name,
            doctor: appointmentData.doctor?.name,
            email: appointmentData.doctor?.email,
            date: appointmentData.appointment?.date
        });

        // Send email notification
        const emailResult = await emailService.sendAppointmentNotification(appointmentData);

        if (emailResult.success) {
            console.log('✅ Appointment email sent successfully');
            res.json({
                success: true,
                message: 'Appointment notification email sent successfully',
                emailResult: emailResult
            });
        } else {
            console.error('❌ Failed to send appointment email:', emailResult.error);
            res.status(500).json({
                success: false,
                error: 'Failed to send email notification',
                details: emailResult.error
            });
        }

    } catch (error) {
        console.error('❌ Error in send-appointment-email endpoint:', error);
        res.status(500).json({
            success: false,
            error: 'Internal server error while sending email',
            message: error.message
        });
    }
});

// Test email endpoint
app.post('/api/test-email', async (req, res) => {
    try {
        const { testEmail } = req.body;

        console.log('🧪 Testing email configuration...');
        const testResult = await emailService.sendTestEmail(testEmail);

        if (testResult.success) {
            res.json({
                success: true,
                message: 'Test email sent successfully',
                result: testResult
            });
        } else {
            res.status(500).json({
                success: false,
                error: 'Test email failed',
                details: testResult.error
            });
        }

    } catch (error) {
        console.error('❌ Error in test-email endpoint:', error);
        res.status(500).json({
            success: false,
            error: 'Internal server error during email test',
            message: error.message
        });
    }
});

// Get email service status endpoint
app.get('/api/email-status', (req, res) => {
    const status = emailService.getStatus();
    res.json(status);
});

// In server.js - Replace the /api/match-specialization endpoint with this enhanced version:

app.post('/api/match-specialization', async (req, res) => {
    try {
        const { patientIssues, availableSpecializations, knowledgeBaseString, preferredTime } = req.body;

        if (!model && !hfServiceAvailable) {
            return res.json({
                error: 'AI service not available for specialization matching'
            });
        }

        const timePreferenceNote = preferredTime
            ? `\nPATIENT PREFERRED TIME: ${preferredTime} (prioritise slots in this part of the day)`
            : '';

        const specializationPrompt = `You are a medical expert helping to match patient symptoms with the most appropriate medical specialization AND find the next available appointment slot.

PATIENT SYMPTOMS/ISSUES:
${patientIssues}${timePreferenceNote}

AVAILABLE SPECIALIZATIONS:
${availableSpecializations.map((spec, index) => `${index + 1}. ${spec}`).join('\n')}

COMPLETE DOCTOR KNOWLEDGE BASE:
${knowledgeBaseString}

MEDICAL SPECIALIZATION GUIDELINES:
- General Physician: Treats ALL general and common illnesses — fever, cold, flu, fatigue, headache, stomach pain, nausea, vomiting, diarrhoea, constipation, gas, bloating, abdominal pain, back pain, body aches, joint pain, muscle pain, cough, shortness of breath, chest tightness (non-cardiac), diabetes, hypertension, routine checkups. Use this as the DEFAULT when no other specialist fits.
- Cardiologist: Heart specialist — chest pain (cardiac), heart palpitations, high blood pressure, cardiovascular disease, heart attack symptoms, irregular heartbeat.
- Dermatologist: Skin specialist — rash, itching, acne, eczema, psoriasis, skin lesions, skin infections, hives, dermatitis, pigmentation issues, hair loss, nail problems.
- Urologist: Urinary system — kidney stones, bladder problems, urinary tract infections (UTI), prostate issues, urinary incontinence.
- Gynaecologist: Women's health — menstrual problems, pregnancy care, ovarian cysts, pelvic pain, reproductive health.
- Ophthalmologist: Eye specialist — vision problems, eye pain, cataracts, glaucoma, eye infections, retinal problems.

CRITICAL ROUTING RULES:
1. Stomach pain, nausea, vomiting, diarrhoea, constipation, gas, bloating → ALWAYS General Physician (not Dermatologist).
2. Skin symptoms (rash, itch, acne, eczema, hives) → ALWAYS Dermatologist.
3. Back pain, joint pain, muscle pain, cough, breathlessness → General Physician (unless clearly cardiac).
4. Menstrual problems, period pain, irregular periods, vaginal discharge, pelvic pain, pregnancy, ovarian issues, reproductive health → ALWAYS Gynaecologist (if available and patient is female).
5. Eye problems, vision issues, eye pain, redness, cataracts → ALWAYS Ophthalmologist.
6. Urinary problems, kidney stones, UTI, prostate issues → ALWAYS Urologist.
7. If in doubt, route to General Physician.
8. ONLY pick a specialization that is present in the AVAILABLE SPECIALIZATIONS list above.

APPOINTMENT SCHEDULING GUIDELINES:
1. Each appointment slot is exactly 15 minutes
2. Check each doctor's "Latest Booked Slot" - book AFTER this time/date
3. If Latest Booked Slot is "15:00", next available slot is "15:15"
4. If Latest Booked Slot is "10:45", next available slot is "11:00"
5. Respect doctor's weekly "Availability" schedule (e.g., "Monday to Friday - 19:00 to 21:00")
6. If Latest Booked Slot is "NIL", book the first available slot according to their availability schedule
7. Book for the NEXT available 15-minute slot that respects both constraints
8. Consider today's date as ${new Date().toDateString()}
9. Book during doctor's available hours only
10. Time format: Use 24-hour format (e.g., "14:30" not "2:30 PM")

EXAMPLES:
- Latest Booked: "2025-06-21 09:15" → Next Slot: "2025-06-21 09:30" 
- Latest Booked: "2025-06-21 17:45" → Next Slot: "2025-06-22 [start time]" (if 17:45 is end of day)
- Latest Booked: "NIL" → Next Slot: First available slot in doctor's schedule

TASK:
1. Determine the most appropriate specialization from available options
2. Find the best doctor in that specialization
3. Calculate the next available appointment slot for that doctor
4. Provide confidence level (0.0 to 1.0) for the match
5. Give brief reasoning for your recommendation

Respond in JSON format:
{
  "specialization": "exact name from available list",
  "doctorName": "selected doctor's name",
  "confidence": 0.85,
  "reason": "brief explanation for the match",
  "appointmentDate": "YYYY-MM-DD",
  "appointmentTime": "HH:MM",
  "schedulingReason": "brief explanation of why this slot was chosen"
}`;

        console.log('🔍 Enhanced specialization matching and scheduling request:', {
            symptoms: patientIssues.substring(0, 100) + '...',
            availableSpecs: availableSpecializations
        });

        let text = '';
        if (isGeminiAvailable()) {
            try {
                const matchModel = genAI.getGenerativeModel({
                    model: 'gemini-2.0-flash',
                    generationConfig: { temperature: 0.1, maxOutputTokens: 500 }
                });
                const result = await matchModel.generateContent(specializationPrompt);
                text = result.response.text().trim();
                console.log('🤖 Gemini specialization match response received');
            } catch (geminiError) {
                if (geminiError.message && geminiError.message.includes('429')) markGeminiQuotaExhausted();
                console.warn('⚠️ Gemini specialization match failed, falling back to Ollama:', geminiError.message);
            }
        }
        if (!text && hfServiceAvailable) {
            text = await generateAIText(specializationPrompt, 500);
        }
        if (!text) throw new Error('No AI service available for specialization matching');

        // Parse JSON response
        let matchResult;
        try {
            const cleanText = text.replace(/```json\n?|\n?```/g, '').trim();
            matchResult = JSON.parse(cleanText);
        } catch (parseError) {
            const jsonMatch = text.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                matchResult = JSON.parse(jsonMatch[0]);
            } else {
                throw new Error('Invalid response format');
            }
        }

        console.log('✅ Enhanced specialization and scheduling result:', matchResult);

        // Immediately update the knowledge base so no overlap happens
        await knowledgeBaseUpdater.updateKnowledgeBase(
               matchResult.doctorName,
               matchResult.appointmentDate,
               matchResult.appointmentTime
              );


        res.json({
            specialization: matchResult.specialization,
            doctorName: matchResult.doctorName,
            confidence: matchResult.confidence,
            reason: matchResult.reason,
            appointmentDate: matchResult.appointmentDate,
            appointmentTime: matchResult.appointmentTime,
            schedulingReason: matchResult.schedulingReason,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('❌ Error in enhanced specialization matching:', error);
        res.json({
            error: error.message
        });
    }
});

// Helper function to build medical chat prompt with context
// Returns a messages array for Ollama /api/chat (proper role-based — prevents repetition)
function buildMedicalChatPrompt(message, conversationHistory, systemPrompt, patientData) {
    const extractedFields = Object.keys(patientData).filter(key =>
        patientData[key] && patientData[key] !== "" && patientData[key] !== "N/A" && patientData[key] !== 0
    );

    const missingFields = ['name', 'age', 'gender', 'contact', 'symptoms', 'diagnosis']
        .filter(field => !extractedFields.includes(field));

    const patientContext = extractedFields.length > 0 ?
        `\nCURRENT PATIENT INFORMATION COLLECTED:\n${extractedFields.map(field => `- ${field}: ${patientData[field]}`).join('\n')}\nSTILL MISSING: ${missingFields.join(', ')}\nDo NOT ask for information already collected.` :
        'NO PATIENT INFORMATION COLLECTED YET. You need: name, age, gender, contact, symptoms and diagnosis.';

    // Only inject symptom-specific questions once symptoms are known
    const symptoms = (patientData.symptoms || '').toLowerCase();
    let questionContext = '';
    if (symptoms) {
        const { specialty, context: kbContext } = keywordFallback(patientData.symptoms, patientData.gender || null);
        if (kbContext && kbContext.trim()) {
            questionContext = `\nSUGGESTED FOLLOW-UP QUESTIONS FOR ${specialty.toUpperCase()} (only use if relevant to what THIS patient said):\n${kbContext}`;
        }
    }

    const gender = patientData.gender || null;
    const genderWarning = gender === 'Male'
        ? '\n⚠️ PATIENT IS MALE. NEVER ask about periods, menstruation, pregnancy, or female-specific conditions.'
        : gender === 'Female'
        ? '\n⚠️ PATIENT IS FEMALE. NEVER ask about prostate or male-specific conditions.'
        : '';

    const needsBasic = !patientData.name || !patientData.age || !patientData.gender;
    const symptomWarning = symptoms
        ? `\n⚠️ PATIENT'S SYMPTOMS ARE: "${patientData.symptoms}". ONLY ask follow-up questions about THESE symptoms. Do NOT introduce or ask about any other medical condition or symptom the patient has NOT mentioned.`
        : needsBasic
        ? '\n⚠️ You do NOT yet have name, age, and gender. Ask for ALL THREE together in one question.'
        : '\n⚠️ You have name/age/gender. Now ask for phone number, then symptoms.';

    // Follow-up phase enforcement
    const hasSymptoms = !!(patientData.symptoms && patientData.symptoms.trim());
    const allBasicCollected = patientData.name && patientData.age && patientData.gender && patientData.contact && patientData.symptoms;
    const assistantTurns = conversationHistory.filter(m => m.role === 'assistant').length;
    const followUpsDone = allBasicCollected ? Math.max(0, assistantTurns - 4) : 0;
    const needsMoreFollowUp = hasSymptoms && followUpsDone < 3;

    const followUpInstruction = needsMoreFollowUp
        ? `\n\n🔴 FOLLOW-UP PHASE — YOU MUST ASK MORE QUESTIONS:\n- You have the patient's basic info and symptoms.\n- You have asked ${followUpsDone} follow-up question(s) so far. You need at least 3.\n- Ask ONE specific follow-up question now about: duration, severity (1-10), onset, triggers, associated symptoms, or relevant medical history.\n- DO NOT ask about appointment time yet.\n- DO NOT say "I have sufficient information" yet.`
        : hasSymptoms
        ? `\n\n✅ FOLLOW-UP COMPLETE — Ask appointment time preference (morning/afternoon/evening), then say "I have sufficient information to book your appointment now."`
        : '';

    const systemContent = `${systemPrompt}\n\n${patientContext}${genderWarning}${symptomWarning}${questionContext}${followUpInstruction}\n\nRESPONSE GUIDELINES:\n- Ask name, age, and gender TOGETHER in one question if any are missing\n- NEVER ask for information you already have\n- Do NOT repeat questions already asked\n- ONLY ask about symptoms the patient has explicitly mentioned\n- If the user says "I don't know" or denies a symptom, accept it and move on — do NOT repeat\n- Ask at least 3 follow-up symptom questions before concluding\n- If you cannot diagnose, say: "Diagnosis is unknown"\n- Say "I have sufficient information to book your appointment now." ONLY after 3+ follow-up questions AND appointment time preference is collected.`;

    // Build messages array with proper roles
    const messages = [{ role: 'system', content: systemContent }];

    // Add conversation history as proper user/assistant turns
    if (conversationHistory.length > 0) {
        conversationHistory.slice(-6).forEach(msg => {
            messages.push({
                role: msg.role === 'assistant' ? 'assistant' : 'user',
                content: msg.content
            });
        });
    }

    // Add current user message
    messages.push({ role: 'user', content: message });

    return messages;
}

// Helper function to build extraction prompt
function buildExtractionPrompt(conversationText, extractionPrompt, currentData) {
    const currentFields = Object.keys(currentData).filter(key => 
        currentData[key] && currentData[key] !== "" && currentData[key] !== "N/A" && currentData[key] !== 0
    );

    return `${extractionPrompt}

CURRENT PATIENT DATA ALREADY COLLECTED:
${currentFields.length > 0 ? 
    currentFields.map(field => `- ${field}: ${currentData[field]}`).join('\n') : 
    'No information collected yet'}

CONVERSATION TO ANALYZE:
${conversationText}

CRITICAL INSTRUCTIONS:
1. ONLY extract information that is EXPLICITLY stated by the patient
2. NEVER assume, infer, or guess any information
3. If a field is not clearly mentioned, return null
4. Only extract NEW information not already captured
5. Be extremely conservative - when in doubt, return null

Return ONLY a valid JSON object with the extracted information. Do not include any explanatory text before or after the JSON:
{
  "name": "string or null",
  "age": "number or null", 
  "gender": "Male/Female/Other or null",
  "contact": " phone or null",
  "symptoms": "string description or null",
  "preferredDoctor": "string or null",
  "diagnosis": "string description or null"
}`;
}

// Fallback response function
function getFallbackResponse(message, patientData) {
    const data = patientData || {};
    
    // Check what information is missing
    const missingFields = ['name', 'age', 'gender', 'contact', 'symptoms', 'diagnosis'].filter(field => 
        !data[field] || data[field] === "" || data[field] === 0
    );

    if (missingFields.length === 0) {
        return "Thank you for providing all the information. Let me ask a few follow-up questions about your symptoms to better understand your condition.";
    }

    // Ask for missing information
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
    
    if (missingFields.includes('diagnosis')) {
        return "Based on your symptoms, could you provide any additional details or your understanding of what might be causing these issues?";
    }

    return "Thank you for that information. Is there anything else about your symptoms that you'd like to mention?";
}
// Add new endpoint to get knowledge base from file
app.get('/api/knowledge-base', async (req, res) => {
    try {
        console.log('📚 Loading knowledge base from file...');
        
        const doctors = await knowledgeBaseReader.readKnowledgeBase();
        const knowledgeBaseString = knowledgeBaseReader.generateKnowledgeBaseString();
        
        res.json({
            success: true,
            doctors: doctors,
            knowledgeBaseString: knowledgeBaseString,
            totalDoctors: doctors.length,
            availableSpecializations: knowledgeBaseReader.getAvailableSpecializations()
        });
        
    } catch (error) {
        console.error('❌ Error loading knowledge base:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Add new endpoint to update knowledge base after appointment booking
app.post('/api/update-knowledge-base', async (req, res) => {
    try {
        const { doctorName, appointmentDate, appointmentTime } = req.body;
        
        if (!doctorName || !appointmentDate || !appointmentTime) {
            return res.status(400).json({
                success: false,
                error: 'Missing required fields: doctorName, appointmentDate, appointmentTime'
            });
        }
        
        console.log('🔄 Updating knowledge base after appointment booking...');
        
        // Create backup before updating
        await knowledgeBaseUpdater.createBackup();
        
        // Update the knowledge base
        const updateResult = await knowledgeBaseUpdater.updateKnowledgeBase(
            doctorName, 
            appointmentDate, 
            appointmentTime
        );
        
        if (updateResult.success) {
            res.json({
                success: true,
                message: 'Knowledge base updated successfully',
                updateResult: updateResult
            });
        } else {
            res.status(500).json({
                success: false,
                error: 'Failed to update knowledge base',
                details: updateResult.error
            });
        }
        
    } catch (error) {
        console.error('❌ Error updating knowledge base:', error);
        res.status(500).json({
            success: false,
            error: 'Internal server error while updating knowledge base',
            message: error.message
        });
    }
});

/* APPOINTMENT MANAGER BLOCK - START (ADDED) */

async function readAppointments() {
  try {
    const txt = await fsPromises.readFile(APPT_FILE, 'utf8');
    if (!txt || txt.trim() === '') {
      // empty file -> initialize with []
      await fsPromises.writeFile(APPT_FILE, '[]', 'utf8');
      return [];
    }
    try {
      return JSON.parse(txt);
    } catch (parseErr) {
      console.warn('⚠️ appointments.json contains invalid JSON — resetting to []', parseErr.message);
      await fsPromises.writeFile(APPT_FILE, '[]', 'utf8');
      return [];
    }
  } catch (e) {
    if (e.code === 'ENOENT') {
      // file doesn't exist -> create it
      await fsPromises.writeFile(APPT_FILE, '[]', 'utf8');
      return [];
    }
    throw e;
  }
}


async function writeAppointments(data) {
  const tmp = APPT_FILE + '.tmp';
  await fsPromises.writeFile(tmp, JSON.stringify(data, null, 2), 'utf8');
  await fsPromises.rename(tmp, APPT_FILE);
}

function enqueueBooking(fn) {
  bookingQueue = bookingQueue.then(() => fn()).catch(err => {
    console.error('Booking queue error', err);
  });
  return bookingQueue;
}

function pad(n) { return n < 10 ? '0' + n : '' + n; }
function formatDate(d) {
  const yyyy = d.getFullYear();
  const mm = pad(d.getMonth() + 1);
  const dd = pad(d.getDate());
  return `${yyyy}-${mm}-${dd}`;
}
function formatTime(d) {
  const hh = pad(d.getHours());
  const mm = pad(d.getMinutes());
  return `${hh}:${mm}`;
}
function addMinutes(d, mins) {
  return new Date(d.getTime() + mins * 60000);
}
function roundUpToNextQuarter(d) {
  const dt = new Date(d);
  dt.setSeconds(0, 0);
  const mins = dt.getMinutes();
  const rem = mins % 15;
  if (rem === 0) {
    dt.setMinutes(mins + 15);
  } else {
    dt.setMinutes(mins + (15 - rem));
  }
  return dt;
}
function parseDateTimeString(s) {
  if (!s) return null;
  let d = new Date(s);
  if (!isNaN(d)) return d;
  const m = s.match(/(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})/);
  if (m) {
    return new Date(m[1] + 'T' + m[2] + ':00');
  }
  return null;
}
// ---------- WhatsApp/contact helpers (paste after parseDateTimeString) ----------
/**
 * Return a WhatsApp destination string (e.g. 'whatsapp:+911234567890') or null.
 * Uses whatsappService.formatToWhatsApp (so it uses your DEFAULT_COUNTRY if needed).
 */
function getWhatsAppAddress(contact) {
  try {
    if (!contact) return null;
    if (whatsappService && typeof whatsappService.formatToWhatsApp === 'function') {
      return whatsappService.formatToWhatsApp(contact);
    }
    // fallback - simple digit check
    const digits = contact.toString().replace(/\D/g, '');
    if (digits.length >= 8 && digits.length <= 15) {
      // default +91 if no plus sign (keeps format used in whatsappService)
      return `whatsapp:${digits.length === 10 ? (process.env.DEFAULT_COUNTRY_CODE || '+91') + digits : '+' + digits}`;
    }
    return null;
  } catch (e) {
    console.warn('getWhatsAppAddress error', e && e.message);
    return null;
  }
}

/** Quick phone test (true if contact is phone-like and not an email) */
function isPhoneLike(contact) {
  if (!contact) return false;
  const s = contact.toString().trim();
  if (s.includes('@')) return false; // treat as email
  const digits = s.replace(/\D/g, '');
  return digits.length >= 8 && digits.length <= 15;
}

function parseAvailability(availabilityString) {
  if (!availabilityString || typeof availabilityString !== 'string') {
    return null;
  }
  const s = availabilityString.trim();
  try {
    const timeMatch = s.match(/(\d{1,2}:\d{2})\s*(?:to|-)\s*(\d{1,2}:\d{2})/i);
    const daysMatch = s.match(/([A-Za-z]+)(?:\s*(?:to|-)\s*([A-Za-z]+))?/i);
    let startTime = null, endTime = null;
    if (timeMatch) {
      startTime = timeMatch[1];
      endTime = timeMatch[2];
    }
    let dayFrom = null, dayTo = null;
    if (daysMatch) {
      const map = { sunday:0,monday:1,tuesday:2,wednesday:3,thursday:4,friday:5,saturday:6,
                    sun:0,mon:1,tue:2,wed:3,thu:4,fri:5,sat:6};
      const a = daysMatch[1].toLowerCase();
      dayFrom = map[a] !== undefined ? map[a] : null;
      if (daysMatch[2]) {
        const b = daysMatch[2].toLowerCase();
        dayTo = map[b] !== undefined ? map[b] : null;
      } else {
        dayTo = dayFrom;
      }
    }
    return {
      startTime,
      endTime,
      dayFrom,
      dayTo,
      isAvailableForDate: function(date) {
        if (!startTime || !endTime) return true;
        const dow = date.getDay();
        if (dayFrom !== null && dayTo !== null) {
          if (dayFrom <= dayTo) {
            if (dow < dayFrom || dow > dayTo) return false;
          } else {
            if (dow > dayTo && dow < dayFrom) return false;
          }
        }
        const [sh, sm] = startTime.split(':').map(Number);
        const [eh, em] = endTime.split(':').map(Number);
        const startDate = new Date(date);
        startDate.setHours(sh, sm, 0, 0);
        const endDate = new Date(date);
        endDate.setHours(eh, em, 0, 0);
        return date >= startDate && date < endDate;
      },
      getDayWindow: function(date) {
        if (!this.startTime || !this.endTime) return null;
        const [sh, sm] = this.startTime.split(':').map(Number);
        const [eh, em] = this.endTime.split(':').map(Number);
        const startDate = new Date(date);
        startDate.setHours(sh, sm, 0, 0);
        const endDate = new Date(date);
        endDate.setHours(eh, em, 0, 0);
        return { startDate, endDate };
      }
    };
  } catch (e) {
    return null;
  }
}

// preferredTime: 'morning' | 'afternoon' | 'evening' | null
function isInPreferredWindow(date, preferredTime) {
    if (!preferredTime) return true;
    const h = date.getHours();
    if (preferredTime === 'morning')   return h >= 8  && h < 12;
    if (preferredTime === 'afternoon') return h >= 12 && h < 17;
    if (preferredTime === 'evening')   return h >= 17 && h < 21;
    return true;
}

async function computeNextAvailableSlot(doctor, appointments, durationMinutes = 15, preferredTime = null) {
  // Prefer saved availability.json over Excel string
  try {
    const allAvail = await readAvailability();
    const { listDoctors: _ld } = require('./doctorAuth');
    const authDoc = _ld().find(d => d.name.toLowerCase() === (doctor.name || doctor.doctorName || doctor['Doctor Name'] || '').toLowerCase());
    if (authDoc && allAvail[authDoc.id]) {
      const saved = allAvail[authDoc.id];
      // Build an availability string compatible with parseAvailability e.g. "Monday to Friday 09:00 to 17:00"
      const dayNames = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
      const sortedDays = [...saved.days].sort((a,b)=>a-b);
      if (sortedDays.length > 0) {
        const dayStr = `${dayNames[sortedDays[0]]} to ${dayNames[sortedDays[sortedDays.length-1]]}`;
        doctor = { ...doctor, availability: `${dayStr} ${saved.startTime} to ${saved.endTime}` };
        durationMinutes = saved.slotDuration || durationMinutes;
      }
    }
  } catch (_) {}

  const latestCandidates = [
    doctor.latestBookedSlot,
    doctor['Latest Booked Slot'],
    doctor.latest_booked_slot,
    doctor.lastBooked,
    doctor['latest_booked_slot']
  ];
  let latest = null;
  for (const c of latestCandidates) {
    if (c) {
      const pd = parseDateTimeString(c);
      if (pd) { latest = pd; break; }
    }
  }

  const now = new Date();
  let candidate;
  if (latest && latest instanceof Date && !isNaN(latest)) {
    candidate = addMinutes(latest, durationMinutes);
    if (candidate < now) candidate = roundUpToNextQuarter(now);
  } else {
    candidate = roundUpToNextQuarter(now);
  }

  // Load saved availability (including dateOverrides) for this doctor
  let savedAvail = null;
  try {
    const allAvail2 = await readAvailability();
    const { listDoctors: _ld2 } = require('./doctorAuth');
    const authDoc2 = _ld2().find(d => d.name.toLowerCase() === (doctor.name || doctor.doctorName || doctor['Doctor Name'] || '').toLowerCase());
    if (authDoc2) savedAvail = allAvail2[authDoc2.id] || null;
  } catch (_) {}

  const availability = parseAvailability(doctor.availability || doctor.Availability || doctor['Availability'] || doctor.Avail || doctor.avail || '');

  function conflicts(candidate) {
    const cd = formatDate(candidate);
    const ct = formatTime(candidate);
    for (const a of appointments) {
      if (!a.doctorName) continue;
      if ((a.doctorName === doctor.name) || (a.doctorName === doctor.doctorName) || (a.doctorName === doctor['Doctor Name'])) {
        if (a.appointmentDate === cd && a.appointmentTime === ct) return true;
      }
    }
    return false;
  }

  const MAX_ITERS = 200;
  let iter = 0;
  while (iter++ < MAX_ITERS) {
    let withinAvailability = true;
    const candidateDateStr = formatDate(candidate);

    // Check per-date override first
    if (savedAvail && savedAvail.dateOverrides && savedAvail.dateOverrides[candidateDateStr] !== undefined) {
      const ovr = savedAvail.dateOverrides[candidateDateStr];
      if (!ovr.available) {
        withinAvailability = false;
      } else if (ovr.startTime && ovr.endTime) {
        const [sh, sm] = ovr.startTime.split(':').map(Number);
        const [eh, em] = ovr.endTime.split(':').map(Number);
        const windowStart = new Date(candidate); windowStart.setHours(sh, sm, 0, 0);
        const windowEnd   = new Date(candidate); windowEnd.setHours(eh, em, 0, 0);
        const slotEnd = addMinutes(candidate, durationMinutes);
        if (candidate < windowStart || slotEnd > windowEnd) withinAvailability = false;
      }
    } else if (availability) {
      const dayWindow = availability.getDayWindow(candidate);
      if (dayWindow) {
        const endCandidate = addMinutes(candidate, durationMinutes);
        if (candidate < dayWindow.startDate || endCandidate > dayWindow.endDate) {
          withinAvailability = false;
        }
      } else {
        if (availability.dayFrom !== null && availability.dayTo !== null) {
          const dow = candidate.getDay();
          const df = availability.dayFrom;
          const dt = availability.dayTo;
          if (df <= dt) {
            if (dow < df || dow > dt) withinAvailability = false;
          } else {
            if (dow > dt && dow < df) withinAvailability = false;
          }
        }
      }
    } else if (savedAvail && savedAvail.days) {
      // Fallback: use saved recurring days when no Excel availability string exists
      if (!savedAvail.days.includes(candidate.getDay())) withinAvailability = false;
    }

    if (withinAvailability && !conflicts(candidate) && isInPreferredWindow(candidate, preferredTime)) {
      return { appointmentDate: formatDate(candidate), appointmentTime: formatTime(candidate) };
    }
    candidate = addMinutes(candidate, durationMinutes);
  }

  // If no slot matched the preferred window, fall back without that constraint
  if (preferredTime) {
    return computeNextAvailableSlot(doctor, appointments, durationMinutes, null);
  }
  return null;
}

// booking endpoint: create appointment and persist and update excel
// booking endpoint: create appointment and persist and update excel
// GET /api/doctors/available?specialization=X&preferredTime=morning
// Returns matching doctors with their next available slot in the preferred window
app.get('/api/doctors/available', async (req, res) => {
  try {
    const { specialization, preferredTime } = req.query;
    const doctors = await knowledgeBaseReader.readKnowledgeBase();
    const appts   = await readAppointments();

    const matched = specialization
      ? doctors.filter(d => d.specialization && d.specialization.toLowerCase().includes(specialization.toLowerCase()))
      : doctors;

    const results = await Promise.all(matched.map(async d => {
      const slot = await computeNextAvailableSlot(d, appts, 15, preferredTime || null);
      return {
        name:           d.name,
        specialization: d.specialization,
        availability:   d.availability || '',
        rating:         d.rating || 4.0,
        nextSlot:       slot || null
      };
    }));

    // Sort by rating descending
    results.sort((a, b) => (b.rating || 0) - (a.rating || 0));
    res.json({ success: true, doctors: results });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

app.post('/api/book-appointment', async (req, res) => {
  try {
    const { doctorName, patient, preferredTime } = req.body;
    if (!doctorName || !patient) {
      return res.status(400).json({ success: false, error: 'doctorName and patient required' });
    }

    const result = await enqueueBooking(async () => {
      const currentAppts = await readAppointments();
      const doctors = await knowledgeBaseReader.readKnowledgeBase();
      const doctor = doctors.find(d => {
        const names = [d.name, d.doctorName, d['Doctor Name'], d.fullName].filter(Boolean).map(x => x.toString());
        return names.some(n => n.toLowerCase() === doctorName.toString().toLowerCase());
      }) || doctors.find(d => {
        const names = [d.name, d.doctorName, d['Doctor Name'], d.fullName].filter(Boolean).map(x => x.toString());
        return names.some(n => n.toLowerCase().includes(doctorName.toString().toLowerCase()));
      });

      if (!doctor) throw new Error('Doctor not found');

      const slot = await computeNextAvailableSlot(doctor, currentAppts, 15, preferredTime || null);
      if (!slot) throw new Error('No available slot found in the next few days/hours');

      // Resolve doctorId from auth DB so doctor portal can filter appointments
      const { listDoctors: listAuthDoctors } = require('./doctorAuth');
      const resolvedName = (doctor.name || doctor.doctorName || doctor['Doctor Name'] || doctorName).toString();
      const authDoctors = listAuthDoctors();
      const matched = authDoctors.find(d => d.name.toLowerCase() === resolvedName.toLowerCase());

      const appt = {
        id: Date.now().toString(),
        doctorId: matched ? matched.id : null,
        doctorName: resolvedName,
        patient: patient,
        appointmentDate: slot.appointmentDate,
        appointmentTime: slot.appointmentTime,
        status: 'pending',
        createdAt: new Date().toISOString()
      };

     // --- Save appointment locally ---
       try {
         currentAppts.push(appt);
         await writeAppointments(currentAppts);
         console.log('✅ Appointment saved to appointments.json:', appt);
         } catch (err) {
         console.error('❌ Failed to write appointment:', err);
         }


      try {
        if (typeof knowledgeBaseUpdater.updateKnowledgeBase === 'function') {
          await knowledgeBaseUpdater.updateKnowledgeBase(appt.doctorName, appt.appointmentDate, appt.appointmentTime);
        } else {
          console.warn('knowledgeBaseUpdater.updateKnowledgeBase function not found');
        }
      } catch (e) {
        console.warn('Failed to update knowledge base Excel:', e.message || e.toString());
      }

      return appt;
    });

    // --- WHATSAPP CONFIRMATION: only if contact is phone-like and WhatsApp address can be derived ---
    try {
       const contact = result.patient && result.patient.contact;
       const waAddr = getWhatsAppAddress(contact);
       console.log('📞 WhatsApp check:', { contact, waAddr });

        if (!waAddr) {
          console.log(`ℹ️ Skipping WhatsApp confirmation: patient has no WhatsApp-capable phone (contact="${contact}")`);
           } else if (!whatsappService || !whatsappService.client) {
             console.warn('⚠️ WhatsApp service not configured (Twilio missing) — cannot send confirmation now');
          } else {
           console.log(`📤 Sending WhatsApp confirmation to ${waAddr} ...`);
          const sendResult = await whatsappService.sendAppointmentConfirmation({
           ...result,
           patient: {
              ...result.patient,
              contact: waAddr
                 }
              });

          if (sendResult && sendResult.success) {
           console.log(`✅ WhatsApp confirmation sent successfully to ${waAddr}`);
          } else {
             console.warn('⚠️ WhatsApp confirmation send reported failure:', sendResult && sendResult.error ? sendResult.error : 'Unknown error');
                }
               }
                } catch (err) {
                     console.error('❌ Failed to send WhatsApp confirmation:', err && err.message ? err.message : String(err));
                }  

    // Schedule reminder (scheduling function will decide whether to actually schedule based on contact)
    scheduleReminderForAppointment(result);
    try {
       const finalAppointments = await readAppointments();
       const found = finalAppointments.find(a => a.id === result.id);
       if (found) {
       console.log('✅ Verified appointment persisted in file.');
      } else {
         console.warn('⚠️ Appointment not found after write, retrying...');
         finalAppointments.push(result);
         await writeAppointments(finalAppointments);
       }
           } catch (err) {
              console.error('❌ Verification write failed:', err);
                }
    res.json({ success: true, appointment: result });
  } catch (error) {
    console.error('Booking error', error);
    res.status(500).json({ success: false, error: error.message || String(error) });
  }
});

// endpoint to list appointments (optionally filter by doctor)
app.get('/api/appointments', async (req, res) => {
  try {
    const doctorName = req.query.doctor;
    const appts = await readAppointments();
    const out = doctorName ? appts.filter(a => a.doctorName && a.doctorName.toLowerCase().includes(doctorName.toLowerCase())) : appts;
    res.json({ success: true, total: out.length, appointments: out });
  } catch (e) {
    res.status(500).json({ success: false, error: e.message });
  }
});
// ----------------- WhatsApp reminder scheduling (ADD) -----------------
const scheduledJobs = new Map(); // appointmentId -> node-schedule job

// Determine lead minutes (priority: minutes var, else hours var, else default 240)
function getReminderLeadMinutes() {
  if (process.env.REMINDER_LEAD_MINUTES) return Number(process.env.REMINDER_LEAD_MINUTES);
  if (process.env.REMINDER_LEAD_HOURS) return Number(process.env.REMINDER_LEAD_HOURS) * 60;
  return 4 * 60; // default 4 hours
}

function getAppointmentDateTime(appt) {
  // Use server's parse helper if available, else fallback to constructing
  try {
    // The server's appointment manager defines parseDateTimeString and addMinutes
    const str = `${appt.appointmentDate} ${appt.appointmentTime}`;
    if (typeof parseDateTimeString === 'function') {
      return parseDateTimeString(str) || new Date(`${appt.appointmentDate}T${appt.appointmentTime}:00`);
    } else {
      return new Date(`${appt.appointmentDate}T${appt.appointmentTime}:00`);
    }
  } catch (e) {
    return null;
  }
}

function scheduleReminderForAppointment(appt) {
  try {
    if (!appt || !appt.id) return;
    if (scheduledJobs.has(appt.id)) {
      console.log('🔔 Reminder already scheduled for', appt.id);
      return;
    }

    const appointmentDt = getAppointmentDateTime(appt);
    if (!appointmentDt || isNaN(appointmentDt.getTime())) {
      console.warn('⚠️ Cannot schedule reminder: invalid appointment datetime for', appt.id, appt);
      return;
    }

    const leadMinutes = getReminderLeadMinutes();
    // negative minutes: reminder time = appointment - leadMinutes
    const reminderTime = (typeof addMinutes === 'function') ? addMinutes(appointmentDt, -leadMinutes) : new Date(appointmentDt.getTime() - leadMinutes * 60000);

    const now = new Date();
    if (reminderTime <= now) {
      console.log('⏭ Reminder time already passed for', appt.id, 'reminderTime=', reminderTime.toString());
      return;
    }

    const job = schedule.scheduleJob(reminderTime, async function() {
      console.log('⏰ Running scheduled reminder for', appt.id, appt.appointmentDate, appt.appointmentTime);
      try {
        if (whatsappService && whatsappService.client) {
          await whatsappService.sendAppointmentReminder(appt);
        } else {
          console.warn('⚠️ WhatsApp service not configured — cannot send reminder for', appt.id);
        }
      } catch (err) {
        console.error('❌ Error sending scheduled reminder for', appt.id, err);
      }
      scheduledJobs.delete(appt.id);
    });

    scheduledJobs.set(appt.id, job);
    console.log(`🗓 Scheduled reminder for appt ${appt.id} at ${reminderTime.toString()}`);
  } catch (e) {
    console.error('❌ scheduleReminderForAppointment error', e);
  }
}

async function scheduleAllReminders() {
  try {
    const appts = await readAppointments();
    for (const a of appts) {
      scheduleReminderForAppointment(a);
    }
    console.log('🔁 scheduleAllReminders completed, scheduled:', scheduledJobs.size);
  } catch (e) {
    console.warn('⚠️ scheduleAllReminders failed', e);
  }
}

// Admin endpoint to trigger a reminder immediately for testing (optional)
// Admin endpoint to trigger a reminder immediately for testing
app.post('/api/trigger-reminder', async (req, res) => {
  try {
    const { appointmentId } = req.body;
    if (!appointmentId) {
      return res.status(400).json({ success: false, error: 'appointmentId required' });
    }

    const appts = await readAppointments();
    const appt = appts.find(a => a.id === appointmentId);
    if (!appt) {
      return res.status(404).json({ success: false, error: 'appointment not found' });
    }

    if (!whatsappService || !whatsappService.client) {
      return res.status(500).json({ success: false, error: 'WhatsApp service not configured' });
    }

    // ✅ Ensure contact is a phone number usable for WhatsApp
    const contact = appt.patient && appt.patient.contact;
    const waAddr = getWhatsAppAddress(contact);
    if (!waAddr) {
      return res.status(400).json({
        success: false,
        error: `Appointment has no WhatsApp-capable phone number (contact="${contact}")`
      });
    }

    const result = await whatsappService.sendAppointmentReminder(appt);
    res.json({ success: true, result });
  } catch (error) {
    console.error('❌ /api/trigger-reminder error', error);
    res.status(500).json({
      success: false,
      error: error.message || String(error)
    });
  }
});

// ----------------- END WhatsApp reminder scheduling -----------------

/* APPOINTMENT MANAGER BLOCK - END (ADDED) */

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('🚨 Server Error:', error);
    res.status(500).json({
        error: 'Internal server error',
        message: error.message,
        timestamp: new Date().toISOString()
    });
});

// Catch-all handler for frontend routes


// Start server
app.listen(PORT, () => {
    console.log('🏥 Enhanced Medical Appointment Server Started');
    console.log(`🌐 Server running on http://localhost:${PORT}`);
    console.log(`🤖 Gemini AI: ${model ? 'Enabled' : 'Disabled (fallback mode)'}`);
    console.log(`📧 Email Service: ${emailService.getStatus().configured ? 'Configured' : 'Not Configured'}`);
    console.log(`📊 Features: Patient Info Extraction, Specialization Matching, Conversation AI, Email Notifications`);
    
    if (!process.env.GEMINI_API_KEY) {
        console.log('\n⚙️  GEMINI SETUP INSTRUCTIONS:');
        console.log('1. Get your Gemini API key from: https://makersuite.google.com/app/apikey');
        console.log('2. Add GEMINI_API_KEY=your_api_key_here to .env file');
    }

    if (!emailService.getStatus().configured) {
        console.log('\n📧 EMAIL SETUP INSTRUCTIONS:');
        console.log('1. Enable 2-Factor Authentication on your Gmail account');
        console.log('2. Generate an App Password: https://myaccount.google.com/apppasswords');
        console.log('3. Add to .env file:');
        console.log('   EMAIL_USER=your_email@gmail.com');
        console.log('   EMAIL_APP_PASSWORD=your_16_digit_app_password');
        console.log('4. Restart the server');
    }
    scheduleAllReminders().catch(err => {
  console.warn('⚠️ scheduleAllReminders failed at startup:', err && err.message ? err.message : err);
  });
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n👋 Shutting down Medical Appointment Server...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n👋 Shutting down Medical Appointment Server...');
    process.exit(0);
});

app.get('/api/test', (req, res) => {
  res.json({ success: true, message: "Backend route working fine!" });
});

// =============================================================================
// DOCTOR AUTH ROUTES  (Phase 1)
// =============================================================================

// POST /api/doctor/login — exchange email+password for a JWT
app.post('/api/doctor/login', (req, res) => {
    const { email, password } = req.body || {};

    if (!email || !password) {
        return res.status(400).json({ error: 'Email and password are required.' });
    }

    const doctor = verifyDoctor(email.trim().toLowerCase(), password);
    if (!doctor) {
        return res.status(401).json({ error: 'Invalid email or password.' });
    }

    const token = generateToken(doctor);
    return res.json({
        token,
        doctor: {
            id: doctor.id,
            name: doctor.name,
            email: doctor.email,
            mustChangePassword: doctor.mustChangePassword
        }
    });
});

// POST /api/doctor/logout — stateless JWT; client just discards the token.
// This endpoint exists so the client has a clean "logout" call.
app.post('/api/doctor/logout', doctorAuthMiddleware, (req, res) => {
    return res.json({ message: 'Logged out successfully.' });
});

// GET /api/doctor/me — return current doctor's profile (token verification)
app.get('/api/doctor/me', doctorAuthMiddleware, (req, res) => {
    const profile = getDoctorById(req.doctor.id);
    if (!profile) return res.status(404).json({ error: 'Doctor not found.' });
    return res.json({ doctor: profile });
});

// GET /api/doctor/profile — return full editable profile
app.get('/api/doctor/profile', doctorAuthMiddleware, (req, res) => {
    const profile = getDoctorById(req.doctor.id);
    if (!profile) return res.status(404).json({ error: 'Doctor not found.' });
    return res.json({ doctor: profile });
});

// PUT /api/doctor/profile — update name, specialization, clinic, contact_phone
app.put('/api/doctor/profile', doctorAuthMiddleware, (req, res) => {
    const { name, specialization, clinic, contact_phone } = req.body || {};
    if (!name || !name.trim()) {
        return res.status(400).json({ error: 'Name is required.' });
    }
    try {
        updateProfile(req.doctor.id, {
            name: name.trim(),
            specialization: (specialization || '').trim(),
            clinic: (clinic || '').trim(),
            contact_phone: (contact_phone || '').trim()
        });
        const updated = getDoctorById(req.doctor.id);
        return res.json({ doctor: updated });
    } catch (err) {
        console.error('update-profile error:', err);
        return res.status(500).json({ error: 'Failed to update profile.' });
    }
});

// POST /api/doctor/change-password — update own password
app.post('/api/doctor/change-password', doctorAuthMiddleware, (req, res) => {
    const { newPassword } = req.body || {};

    if (!newPassword || newPassword.length < 8) {
        return res.status(400).json({ error: 'Password must be at least 8 characters.' });
    }

    try {
        changePassword(req.doctor.id, newPassword);
        return res.json({ message: 'Password updated successfully.' });
    } catch (err) {
        console.error('change-password error:', err);
        return res.status(500).json({ error: 'Failed to update password.' });
    }
});

// GET /api/doctor/appointments — all appointments for the logged-in doctor
app.get('/api/doctor/appointments', doctorAuthMiddleware, async (req, res) => {
    try {
        const appts = await readAppointments();
        const mine = appts.filter(a =>
            a.doctorId === req.doctor.id ||
            (a.doctorName && a.doctorName.toLowerCase() === req.doctor.name.toLowerCase())
        );
        // Sort: upcoming first, then past
        const now = new Date();
        mine.sort((a, b) => {
            const da = new Date(`${a.appointmentDate}T${a.appointmentTime}`);
            const db = new Date(`${b.appointmentDate}T${b.appointmentTime}`);
            return da - db;
        });
        res.json({ success: true, appointments: mine });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

// PUT /api/doctor/appointments/:id/status — accept / reject / reschedule
app.put('/api/doctor/appointments/:id/status', doctorAuthMiddleware, async (req, res) => {
    const { id } = req.params;
    const { status, rescheduleDate, rescheduleTime } = req.body || {};
    const allowed = ['confirmed', 'rejected', 'rescheduled', 'cancelled'];

    if (!status || !allowed.includes(status)) {
        return res.status(400).json({ error: `status must be one of: ${allowed.join(', ')}` });
    }

    try {
        const appts = await readAppointments();
        const idx = appts.findIndex(a => a.id === id);
        if (idx === -1) return res.status(404).json({ error: 'Appointment not found.' });

        const appt = appts[idx];
        // Ensure doctor owns this appointment
        if (appt.doctorId !== req.doctor.id && appt.doctorName.toLowerCase() !== req.doctor.name.toLowerCase()) {
            return res.status(403).json({ error: 'Access denied.' });
        }

        appt.status = status;
        if (status === 'rescheduled') {
            if (!rescheduleDate || !rescheduleTime) {
                return res.status(400).json({ error: 'rescheduleDate and rescheduleTime are required for rescheduling.' });
            }
            appt.appointmentDate = rescheduleDate;
            appt.appointmentTime = rescheduleTime;
        }
        appt.updatedAt = new Date().toISOString();
        appts[idx] = appt;
        await writeAppointments(appts);
        res.json({ success: true, appointment: appt });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

// GET /api/doctor/patients/:patientId — all appointments for a patient (this doctor only)
app.get('/api/doctor/patients/:patientId', doctorAuthMiddleware, async (req, res) => {
    try {
        const { patientId } = req.params;
        const appts = await readAppointments();
        // Match by patientId field or normalized contact
        const history = appts.filter(a => {
            const mine = a.doctorId === req.doctor.id ||
                (a.doctorName && a.doctorName.toLowerCase() === req.doctor.name.toLowerCase());
            if (!mine) return false;
            const apptPatientId = a.patientId ||
                (a.patient && a.patient.contact ? a.patient.contact.replace(/\D/g,'').slice(-10) : null);
            return apptPatientId === patientId;
        });
        history.sort((a, b) => new Date(b.appointmentDate) - new Date(a.appointmentDate));
        const patient = history.length ? history[0].patient : null;
        res.json({ success: true, patient, appointments: history });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

// POST /api/doctor/appointments/:id/notes — add/update consultation notes
app.post('/api/doctor/appointments/:id/notes', doctorAuthMiddleware, async (req, res) => {
    const { id } = req.params;
    const { chiefComplaint, examination, diagnosis, prescription, followUp } = req.body || {};

    if (!diagnosis) return res.status(400).json({ error: 'Diagnosis is required.' });

    try {
        const appts = await readAppointments();
        const idx = appts.findIndex(a => a.id === id);
        if (idx === -1) return res.status(404).json({ error: 'Appointment not found.' });

        const appt = appts[idx];
        if (appt.doctorId !== req.doctor.id && appt.doctorName.toLowerCase() !== req.doctor.name.toLowerCase()) {
            return res.status(403).json({ error: 'Access denied.' });
        }

        appt.notes = { chiefComplaint, examination, diagnosis, prescription, followUp, addedAt: new Date().toISOString() };
        appt.updatedAt = new Date().toISOString();
        appts[idx] = appt;
        await writeAppointments(appts);
        res.json({ success: true, appointment: appt });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

// =============================================================================
// PHASE 5: PRESCRIPTION MANAGEMENT
// =============================================================================
const PRESC_FILE = path.resolve('./prescriptions.json');
const AVAIL_FILE = path.resolve('./availability.json');

async function readAvailability() {
    try {
        const raw = await fsPromises.readFile(AVAIL_FILE, 'utf8');
        return JSON.parse(raw);
    } catch (e) {
        if (e.code === 'ENOENT') return {};
        throw e;
    }
}

async function writeAvailability(data) {
    await fsPromises.writeFile(AVAIL_FILE, JSON.stringify(data, null, 2));
}

// GET /api/doctor/availability — return this doctor's availability settings
app.get('/api/doctor/availability', doctorAuthMiddleware, async (req, res) => {
    try {
        const all = await readAvailability();
        const avail = all[req.doctor.id] || {
            days: [1, 2, 3, 4, 5],
            startTime: '09:00',
            endTime: '17:00',
            slotDuration: 15
        };
        res.json({ success: true, availability: avail });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

// PUT /api/doctor/availability — save recurring availability settings
app.put('/api/doctor/availability', doctorAuthMiddleware, async (req, res) => {
    const { days, startTime, endTime, slotDuration } = req.body || {};
    if (!Array.isArray(days) || !startTime || !endTime)
        return res.status(400).json({ error: 'days, startTime, and endTime are required.' });
    try {
        const all = await readAvailability();
        const existing = all[req.doctor.id] || {};
        all[req.doctor.id] = {
            ...existing,
            days,
            startTime,
            endTime,
            slotDuration: slotDuration || 15,
            doctorName: req.doctor.name,
            updatedAt: new Date().toISOString()
        };
        await writeAvailability(all);
        res.json({ success: true, availability: all[req.doctor.id] });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

// PUT /api/doctor/availability/date/:date — set or clear a per-date override
// Body: { available: bool, startTime?: "HH:MM", endTime?: "HH:MM" }
app.put('/api/doctor/availability/date/:date', doctorAuthMiddleware, async (req, res) => {
    const { date } = req.params;
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date))
        return res.status(400).json({ error: 'date must be YYYY-MM-DD' });
    const { available, startTime, endTime, clear } = req.body || {};
    try {
        const all = await readAvailability();
        const avail = all[req.doctor.id] || { days:[1,2,3,4,5], startTime:'09:00', endTime:'17:00', slotDuration:15 };
        if (!avail.dateOverrides) avail.dateOverrides = {};
        if (clear) {
            delete avail.dateOverrides[date];
        } else {
            avail.dateOverrides[date] = {
                available: !!available,
                ...(startTime ? { startTime } : {}),
                ...(endTime   ? { endTime }   : {})
            };
        }
        avail.updatedAt = new Date().toISOString();
        all[req.doctor.id] = avail;
        await writeAvailability(all);
        res.json({ success: true, availability: avail });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

async function readPrescriptions() {
    try {
        const raw = await fsPromises.readFile(PRESC_FILE, 'utf8');
        return JSON.parse(raw);
    } catch (e) {
        if (e.code === 'ENOENT') return [];
        throw e;
    }
}

async function writePrescriptions(data) {
    await fsPromises.writeFile(PRESC_FILE, JSON.stringify(data, null, 2));
}

// POST /api/doctor/prescriptions — write a new prescription
app.post('/api/doctor/prescriptions', doctorAuthMiddleware, async (req, res) => {
    const { patientId, patientName, patientAge, patientGender, patientContact,
            diagnosis, medicines, notes, appointmentId } = req.body || {};

    if (!patientId) return res.status(400).json({ error: 'patientId is required.' });
    if (!medicines || !Array.isArray(medicines) || medicines.length === 0)
        return res.status(400).json({ error: 'At least one medicine is required.' });

    for (const m of medicines) {
        if (!m.name || !m.dosage || !m.duration)
            return res.status(400).json({ error: 'Each medicine must have name, dosage, and duration.' });
    }

    try {
        const prescriptions = await readPrescriptions();
        const prescription = {
            id: Date.now().toString(),
            patientId,
            patientName: patientName || '',
            patientAge: patientAge || null,
            patientGender: patientGender || '',
            patientContact: patientContact || '',
            doctorId: req.doctor.id,
            doctorName: req.doctor.name,
            diagnosis: diagnosis || '',
            medicines,
            notes: notes || '',
            appointmentId: appointmentId || null,
            createdAt: new Date().toISOString()
        };
        prescriptions.push(prescription);
        await writePrescriptions(prescriptions);
        res.json({ success: true, prescription });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

// GET /api/doctor/prescriptions/:patientId — all prescriptions for a patient (this doctor only)
app.get('/api/doctor/prescriptions/:patientId', doctorAuthMiddleware, async (req, res) => {
    try {
        const { patientId } = req.params;
        const prescriptions = await readPrescriptions();
        const result = prescriptions.filter(p =>
            p.patientId === patientId &&
            (p.doctorId === req.doctor.id || p.doctorName.toLowerCase() === req.doctor.name.toLowerCase())
        );
        result.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
        res.json({ success: true, prescriptions: result });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

// GET /api/prescriptions/:patientId — patient views their own prescriptions (no auth)
app.get('/api/prescriptions/:patientId', async (req, res) => {
    try {
        const { patientId } = req.params;
        const prescriptions = await readPrescriptions();
        const result = prescriptions.filter(p => p.patientId === patientId);
        result.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
        res.json({ success: true, prescriptions: result });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

// =============================================================================
// END DOCTOR AUTH ROUTES
// =============================================================================

app.use(express.static(path.join(__dirname, 'public')));

app.get('*', (req, res) => {
    if (req.path === '/doctor-login.html') {
        return res.sendFile(path.join(__dirname, 'doctor-login.html'));
    }
    if (req.path === '/doctor-dashboard.html') {
        return res.sendFile(path.join(__dirname, 'doctor-dashboard.html'));
    }
    res.sendFile(path.join(__dirname, 'index.html'));
});
