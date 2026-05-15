// server.js - Enhanced Medical Appointment Server with Railway FastAPI Integration
// Node.js backend with Railway FastAPI integration for patient information extraction and doctor email notifications

const fetch = require('node-fetch');
const { Headers } = require('node-fetch');

// Make fetch and Headers globally available
global.fetch = fetch;
global.Headers = Headers;
global.Request = require('node-fetch').Request;
global.Response = require('node-fetch').Response;

const express = require('express');
const cors = require('cors');
const path = require('path');
const EmailService = require('./emailService'); // Import email service
const KnowledgeBaseReader = require('./knowledgeBaseReader');
const KnowledgeBaseUpdater = require('./updateKnowledgeBase');

// Add these imports after your existing requires
const { router: authRouter, verifyToken, dbManager } = require('./auth-routes');

require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Initialize email service
const emailService = new EmailService();
// Initialize knowledge base services
const knowledgeBaseReader = new KnowledgeBaseReader('./doctors.csv');
const knowledgeBaseUpdater = new KnowledgeBaseUpdater('./doctors.csv');

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Serve static files
app.use(express.static('.'));
app.use('/api', authRouter);

// FastAPI service configuration
const FASTAPI_CONFIG = {
    baseUrl: process.env.FASTAPI_URL || 'https://your-railway-fastapi-url.railway.app',
    timeout: 120000, // 2 minutes timeout for model responses
    retryAttempts: 3
};

let modelReady = false;
let modelLoading = false;

// Function to call Railway FastAPI service
async function queryFastAPIModel(prompt, maxLength = 500, temperature = 0.7) {
    try {
        console.log('🚂 Querying Railway FastAPI model...');
        
        const response = await fetch(`${FASTAPI_CONFIG.baseUrl}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                max_length: maxLength,
                temperature: temperature
            }),
            timeout: FASTAPI_CONFIG.timeout
        });

        if (!response.ok) {
            const errorText = await response.text();
            
            // Handle specific error cases
            if (response.status === 503) {
                if (errorText.includes('loading')) {
                    modelLoading = true;
                    modelReady = false;
                    throw new Error('Model is still loading. Please try again in a few minutes.');
                } else {
                    throw new Error('Model failed to load. Please check FastAPI service.');
                }
            }
            
            throw new Error(`FastAPI Error ${response.status}: ${errorText}`);
        }

        const result = await response.json();
        
        if (result.status === 'success') {
            console.log('✅ Railway FastAPI response received');
            modelReady = true;
            modelLoading = false;
            
            return {
                response: {
                    text: () => result.text
                }
            };
        } else {
            throw new Error('Invalid response from FastAPI service');
        }

    } catch (error) {
        console.error('❌ Error querying FastAPI model:', error);
        
        // Update model status based on error
        if (error.message.includes('loading')) {
            modelLoading = true;
            modelReady = false;
        } else if (error.message.includes('ECONNREFUSED') || error.message.includes('fetch')) {
            modelReady = false;
            modelLoading = false;
        }
        
        throw error;
    }
}

// Function to check FastAPI service health
async function checkFastAPIHealth() {
    try {
        console.log('🏥 Checking FastAPI service health...');
        
        const response = await fetch(`${FASTAPI_CONFIG.baseUrl}/health`, {
            method: 'GET',
            timeout: 10000 // 10 second timeout for health check
        });

        if (response.ok) {
            const healthData = await response.json();
            
            modelReady = healthData.model_loaded;
            modelLoading = healthData.model_loading;
            
            console.log('✅ FastAPI service health:', {
                status: healthData.status,
                modelLoaded: healthData.model_loaded,
                modelLoading: healthData.model_loading,
                modelName: healthData.model_name
            });
            
            return healthData;
        } else {
            throw new Error(`Health check failed: ${response.status}`);
        }

    } catch (error) {
        console.error('❌ FastAPI health check failed:', error.message);
        modelReady = false;
        modelLoading = false;
        return null;
    }
}

// Initialize FastAPI connection
async function initializeFastAPIConnection() {
    try {
        console.log('🔄 Initializing connection to Railway FastAPI service...');
        console.log(`📍 FastAPI URL: ${FASTAPI_CONFIG.baseUrl}`);
        
        // Check if service is reachable
        const healthStatus = await checkFastAPIHealth();
        
        if (healthStatus) {
            console.log('✅ FastAPI service connection established');
            
            // If model is not loaded yet, trigger loading
            if (!healthStatus.model_loaded && !healthStatus.model_loading) {
                console.log('🔄 Triggering model loading...');
                try {
                    await fetch(`${FASTAPI_CONFIG.baseUrl}/load-model`, {
                        method: 'POST',
                        timeout: 5000
                    });
                    console.log('✅ Model loading triggered');
                } catch (loadError) {
                    console.warn('⚠️ Could not trigger model loading:', loadError.message);
                }
            }
            
            return { generateContent: queryFastAPIModel };
        } else {
            throw new Error('FastAPI service not reachable');
        }

    } catch (error) {
        console.error('❌ Error initializing FastAPI connection:', error.message);
        console.log('⚠️ Server will run in fallback mode');
        return null;
    }
}

// Initialize model connection on startup
let model = null;

initializeFastAPIConnection().then(initializedModel => {
    model = initializedModel;
    if (model) {
        console.log('✅ FastAPI model connection ready');
        
        // Start periodic health checks
        setInterval(async () => {
            await checkFastAPIHealth();
        }, 60000); // Check every minute
        
    } else {
        console.log('⚠️ Server running in fallback mode');
    }
}).catch(error => {
    console.error('❌ Failed to initialize FastAPI connection:', error);
    console.log('⚠️ Server running in fallback mode');
});

// Enhanced health check endpoint
app.get('/api/health', async (req, res) => {
    const emailStatus = emailService.getStatus();
    
    // Get latest FastAPI status
    const fastApiHealth = await checkFastAPIHealth();
    
    const status = {
        message: 'Medical Appointment Server is running',
        timestamp: new Date().toISOString(),
        modelAvailable: modelReady,
        modelLoading: modelLoading,
        modelType: 'Railway FastAPI Service',
        fastApiUrl: FASTAPI_CONFIG.baseUrl,
        fastApiStatus: fastApiHealth ? 'Connected' : 'Disconnected',
        emailService: {
            initialized: emailStatus.initialized,
            configured: emailStatus.configured,
            emailUser: emailStatus.emailUser
        }
    };
    
    res.json(status);
});

// Enhanced medical chat endpoint
app.post('/api/medical-chat', async (req, res) => {
    try {
        const { message, conversationHistory = [], systemPrompt, patientData } = req.body;

        if (!model) {
            return res.json({
                error: 'FastAPI model service not available',
                fallback: getFallbackResponse(message, patientData)
            });
        }

        if (modelLoading) {
            return res.json({
                error: 'Model is still loading. Please try again in a few minutes.',
                fallback: getFallbackResponse(message, patientData),
                modelStatus: 'loading'
            });
        }

        if (!modelReady) {
            return res.json({
                error: 'Model not ready. Please check FastAPI service.',
                fallback: getFallbackResponse(message, patientData),
                modelStatus: 'not_ready'
            });
        }

        // Build context-aware prompt
        const contextPrompt = buildMedicalChatPrompt(
            message, 
            conversationHistory, 
            systemPrompt, 
            patientData
        );

        console.log('🤖 FastAPI Model Chat Request:', {
            message: message.substring(0, 100) + '...',
            historyLength: conversationHistory.length,
            patientFields: Object.keys(patientData).filter(key => patientData[key])
        });

        const result = await model.generateContent(contextPrompt);
        const response = result.response;
        const text = response.text();

        console.log('✅ FastAPI Model Response Generated');

        res.json({
            text: text,
            timestamp: new Date().toISOString(),
            modelType: 'railway-fastapi',
            modelStatus: 'ready'
        });

    } catch (error) {
        console.error('❌ Error in medical chat:', error);
        
        // Provide specific error messages
        let errorMessage = error.message;
        let modelStatus = 'error';
        
        if (error.message.includes('loading')) {
            errorMessage = 'Model is still loading. Please try again in a few minutes.';
            modelStatus = 'loading';
        } else if (error.message.includes('ECONNREFUSED')) {
            errorMessage = 'FastAPI service is not reachable. Please check deployment.';
            modelStatus = 'disconnected';
        }
        
        res.json({
            error: errorMessage,
            fallback: getFallbackResponse(req.body.message, req.body.patientData),
            modelStatus: modelStatus
        });
    }
});

// New endpoint for patient information extraction
app.post('/api/extract-patient-info', async (req, res) => {
    try {
        const { conversationText, extractionPrompt, currentData } = req.body;

        if (!model) {
            return res.json({
                error: 'FastAPI model not available for extraction'
            });
        }

        if (modelLoading) {
            return res.json({
                error: 'Model is still loading. Please try again in a few minutes.',
                extractedInfo: null
            });
        }

        if (!modelReady) {
            return res.json({
                error: 'Model not ready. Please check FastAPI service.',
                extractedInfo: null
            });
        }

        // Build structured extraction prompt
        const fullExtractionPrompt = buildExtractionPrompt(
            conversationText, 
            extractionPrompt, 
            currentData
        );

        console.log('🧠 FastAPI Extraction Request:', {
            conversationLength: conversationText.length,
            currentFields: Object.keys(currentData).filter(key => currentData[key])
        });

        const result = await model.generateContent(fullExtractionPrompt);
        const response = result.response;
        const text = response.text();

        // Parse JSON response from FastAPI
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
            console.warn('⚠️ Failed to parse JSON from FastAPI response:', text.substring(0, 200));
            
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

// Test FastAPI connection endpoint
app.get('/api/test-fastapi', async (req, res) => {
    try {
        const healthStatus = await checkFastAPIHealth();
        
        if (healthStatus) {
            res.json({
                success: true,
                message: 'FastAPI service is reachable',
                status: healthStatus
            });
        } else {
            res.status(500).json({
                success: false,
                message: 'FastAPI service is not reachable',
                url: FASTAPI_CONFIG.baseUrl
            });
        }
        
    } catch (error) {
        res.status(500).json({
            success: false,
            message: 'Error testing FastAPI connection',
            error: error.message,
            url: FASTAPI_CONFIG.baseUrl
        });
    }
});

// Manual model loading trigger endpoint
app.post('/api/trigger-model-loading', async (req, res) => {
    try {
        console.log('🔄 Manually triggering model loading...');
        
        const response = await fetch(`${FASTAPI_CONFIG.baseUrl}/load-model`, {
            method: 'POST',
            timeout: 10000
        });
        
        if (response.ok) {
            const result = await response.json();
            res.json({
                success: true,
                message: 'Model loading triggered successfully',
                result: result
            });
        } else {
            const errorText = await response.text();
            res.status(500).json({
                success: false,
                message: 'Failed to trigger model loading',
                error: errorText
            });
        }
        
    } catch (error) {
        res.status(500).json({
            success: false,
            message: 'Error triggering model loading',
            error: error.message
        });
    }
});

// Enhanced specialization matching endpoint
app.post('/api/match-specialization', async (req, res) => {
    try {
        const { patientIssues, availableSpecializations, knowledgeBaseString } = req.body;

        if (!model) {
            return res.json({
                error: 'FastAPI model not available for specialization matching'
            });
        }

        if (modelLoading) {
            return res.json({
                error: 'Model is still loading. Please try again in a few minutes.'
            });
        }

        if (!modelReady) {
            return res.json({
                error: 'Model not ready. Please check FastAPI service.'
            });
        }

        const specializationPrompt = `You are a medical expert helping to match patient symptoms with the most appropriate medical specialization AND find the next available appointment slot.

PATIENT SYMPTOMS/ISSUES:
${patientIssues}

AVAILABLE SPECIALIZATIONS:
${availableSpecializations.map((spec, index) => `${index + 1}. ${spec}`).join('\n')}

COMPLETE DOCTOR KNOWLEDGE BASE:
${knowledgeBaseString}

MEDICAL SPECIALIZATION GUIDELINES:
- General Physician: Common illnesses, fever, cold, flu, headache, stomach pain, body aches, routine checkups, diabetes, hypertension, general health issues
- Cardiologist: Heart specialist who treats chest pain, heart problems, high blood pressure, heart palpitations, cardiovascular disease, heart attack symptoms
- Urologist: Urinary system specialist who treats kidney stones, bladder problems, urinary infections, prostate issues, kidney disease, urinary incontinence
- Gynaecologist: Women's health specialist who treats menstrual problems, pregnancy care, reproductive health, ovarian cysts, pelvic pain
- Ophthalmologist: Eye specialist who treats vision problems, eye pain, cataracts, glaucoma, eye infections, retinal problems, vision loss

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

        const result = await model.generateContent(specializationPrompt);
        const response = result.response;
        const text = response.text();

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
function buildMedicalChatPrompt(message, conversationHistory, systemPrompt, patientData) {
    const extractedFields = Object.keys(patientData).filter(key => 
        patientData[key] && patientData[key] !== "" && patientData[key] !== "N/A" && patientData[key] !== 0
    );

    const missingFields = ['name', 'age', 'gender', 'contact', 'symptoms', 'diagnosis']
        .filter(field => !extractedFields.includes(field));

    const patientContext = extractedFields.length > 0 ? 
        `\n\nCURRENT PATIENT INFORMATION COLLECTED:
${extractedFields.map(field => `- ${field}: ${patientData[field]}`).join('\n')}

STILL MISSING: ${missingFields.join(', ')}

IMPORTANT: If any required information is missing, you MUST ask for it specifically. Do NOT assume or guess missing information.` : 
        '\n\nNO PATIENT INFORMATION COLLECTED YET. You need: name, age, gender, contact, symptoms and diagnosis';

    let conversationContext = '';
    if (conversationHistory.length > 0) {
        conversationContext = '\n\nCONVERSATION HISTORY:\n' + 
            conversationHistory.slice(-6).map(msg => 
                `${msg.role}: ${msg.content}`
            ).join('\n');
    }

    return `${systemPrompt}${patientContext}${conversationContext}

CURRENT USER MESSAGE: ${message}

RESPONSE GUIDELINES:
- You already have: ${extractedFields.join(', ')}
- NEVER ask for information you already have
- If you have name, age, gender, contact, and symptoms: ask medical follow-up questions only
- If missing basic info: ${missingFields.join(', ')}, ask for missing info
- Ask maximum 3 medical follow-up questions about symptoms, then complete assessment
- After sufficient follow-up questions about symptoms, say "I have sufficient information to book your appointment now."`;
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
  "contact": "email or phone or null",
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

    // Ask for missing information one at a time
    if (missingFields.includes('name')) {
        return "I still need your full name to book the appointment. Could you please tell me your name?";
    }
    
    if (missingFields.includes('age')) {
        return "I need to know your age. How old are you?";
    }
    
    if (missingFields.includes('gender')) {
        return "Could you please tell me your gender (Male/Female/Other)?";
    }
    
    if (missingFields.includes('contact')) {
        return "I need your contact information to book the appointment. Could you please provide your email address or phone number?";
    }
    
    if (missingFields.includes('symptoms')) {
        return "I need to understand what health issues you're experiencing. Could you please describe your symptoms or the reason for your visit?";
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
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start server
app.listen(PORT, () => {
    console.log('🏥 Enhanced Medical Appointment Server Started');
    console.log(`🌐 Server running on http://localhost:${PORT}`);
    console.log(`🚂 FastAPI Service: ${FASTAPI_CONFIG.baseUrl}`);
    console.log(`🤖 AI Model: ${modelReady ? 'Ready' : modelLoading ? 'Loading' : 'Checking'}`);
    console.log(`📧 Email Service: ${emailService.getStatus().configured ? 'Configured' : 'Not Configured'}`);
    console.log(`📊 Features: Patient Info Extraction, Specialization Matching, Conversation AI, Email Notifications`);
    
    if (!process.env.FASTAPI_URL) {
        console.log('\n⚙️  FASTAPI SETUP INSTRUCTIONS:');
        console.log('1. Get your Railway FastAPI URL from Railway dashboard');
        console.log('2. Add FASTAPI_URL=https://your-railway-fastapi-url.railway.app to .env file');
        console.log('3. Restart the server');
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
