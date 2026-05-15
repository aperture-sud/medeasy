// emailService.js - Email notification service for doctor appointment notifications
// Sends professional email notifications to doctors when appointments are booked

const nodemailer = require('nodemailer');
require('dotenv').config();

class EmailService {
    constructor() {
        this.transporter = null;
        this.initializeTransporter();
    }

    // Initialize email transporter with Gmail SMTP
    initializeTransporter() {
        try {
            this.transporter = nodemailer.createTransport({
                service: 'gmail',
                host: 'smtp.gmail.com',
                port: 587,
                secure: false, // true for 465, false for other ports
                auth: {
                    user: process.env.EMAIL_USER, // Your email address
                    pass: process.env.EMAIL_APP_PASSWORD // Your app-specific password
                },
                tls: {
                    rejectUnauthorized: false
                }
            });

            console.log('✅ Email service initialized successfully');
        } catch (error) {
            console.error('❌ Error initializing email service:', error);
            this.transporter = null;
        }
    }

    // Verify email configuration
    async verifyEmailConfig() {
        if (!this.transporter) {
            throw new Error('Email transporter not initialized');
        }

        try {
            await this.transporter.verify();
            console.log('✅ Email configuration verified');
            return true;
        } catch (error) {
            console.error('❌ Email configuration verification failed:', error);
            throw new Error('Email configuration invalid');
        }
    }

    // Generate professional email HTML template
    generateEmailTemplate(appointmentData) {
        const { patient, doctor, appointment } = appointmentData;
        
        return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>New Appointment Notification</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            background-color: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0;
            font-size: 24px;
        }
        .appointment-details {
            background-color: #f8f9fa;
            border-left: 4px solid #4facfe;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .patient-info {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .info-row:last-child {
            border-bottom: none;
        }
        .label {
            font-weight: bold;
            color: #555;
            flex: 1;
        }
        .value {
            flex: 2;
            color: #333;
        }
        .symptoms {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }
        .footer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 12px 25px;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px 0;
            font-weight: bold;
        }
        .urgent {
            color: #dc3545;
            font-weight: bold;
        }
        .priority-high {
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }
        @media (max-width: 600px) {
            body {
                padding: 10px;
            }
            .container {
                padding: 20px;
            }
            .info-row {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 New Appointment Notification</h1>
            <p>You have a new patient appointment scheduled</p>
        </div>
        
        <div class="appointment-details">
            <h3>📅 Appointment Details</h3>
            <div class="info-row">
                <span class="label">Date:</span>
                <span class="value">${this.formatDate(appointment.date)}</span>
            </div>
            <div class="info-row">
                <span class="label">Time:</span>
                <span class="value">${appointment.time}</span>
            </div>
            <div class="info-row">
                <span class="label">Duration:</span>
                <span class="value">${appointment.duration || '15 minutes'}</span>
            </div>
            <div class="info-row">
                <span class="label">Appointment ID:</span>
                <span class="value">#APT${Date.now().toString().slice(-6)}</span>
            </div>
        </div>

        <div class="patient-info">
            <h3>👤 Patient Information</h3>
            <div class="info-row">
                <span class="label">Full Name:</span>
                <span class="value">${patient.name}</span>
            </div>
            <div class="info-row">
                <span class="label">Age:</span>
                <span class="value">${patient.age} years old</span>
            </div>
            <div class="info-row">
                <span class="label">Gender:</span>
                <span class="value">${patient.gender}</span>
            </div>
            <div class="info-row">
                <span class="label">Contact:</span>
                <span class="value">${patient.contact}</span>
            </div>
        </div>
        <div class="symptoms">
            <h3>🩺 Patient Symptoms/Issues</h3>
            <p><strong>Chief Complaint:</strong></p>
            <p>${patient.issues}</p>
            
            <p><strong>Diagnosis:</strong></p>
            <p>${patient.diagnosis || 'Assessment pending'}</p>
        </div>

        <div style="text-align: center; margin: 30px 0;">
            <p><strong>Please prepare for this appointment accordingly.</strong></p>
            <p>If you need to reschedule or have any questions, please contact the hospital administration.</p>
        </div>

        <div class="footer">
            <p><strong>Hospital Management System</strong></p>
            <p>📞 Contact: +91-XXX-XXX-XXXX | 📧 Email: admin@hospital.com</p>
            <p>🏥 Address: Hospital Address, City, State</p>
            <hr style="margin: 20px 0;">
            <p style="font-size: 12px; color: #999;">
                This is an automated notification. Please do not reply to this email directly.
                <br>Generated on ${new Date().toLocaleString()}
            </p>
        </div>
    </div>
</body>
</html>`;
    }

    // Generate plain text version for email clients that don't support HTML
    generatePlainTextTemplate(appointmentData) {
        const { patient, doctor, appointment } = appointmentData;
        
        return `
NEW APPOINTMENT NOTIFICATION
============================

Dear Dr. ${doctor.name},

You have a new patient appointment scheduled in your ${doctor.specialization} practice.

APPOINTMENT DETAILS:
-------------------
Date: ${this.formatDate(appointment.date)}
Time: ${appointment.time}
Duration: ${appointment.duration || '15 minutes'}
Appointment ID: #APT${Date.now().toString().slice(-6)}

PATIENT INFORMATION:
-------------------
Name: ${patient.name}
Age: ${patient.age} years old
Gender: ${patient.gender}
Contact: ${patient.contact}

SYMPTOMS/ISSUES:
---------------
${patient.issues}

DIAGNOSIS:
----------
${patient.diagnosis || 'Assessment pending'}

Please prepare for this appointment accordingly. If you need to reschedule or have any questions, please contact the hospital administration.

Best regards,
Hospital Management System

Contact: +91-XXX-XXX-XXXX
Email: admin@hospital.com
Address: Hospital Address, City, State

---
This is an automated notification generated on ${new Date().toLocaleString()}
        `.trim();
    }

    // Format date for better readability
    formatDate(dateString) {
        const date = new Date(dateString);
        const options = { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        };
        return date.toLocaleDateString('en-US', options);
    }

    // Validate appointment data
    validateAppointmentData(appointmentData) {
        const { patient, doctor, appointment } = appointmentData;

        // Required fields validation
        const requiredFields = {
            'patient.name': patient?.name,
            'patient.age': patient?.age,
            'patient.gender': patient?.gender,
            'patient.contact': patient?.contact,
            'patient.issues': patient?.issues,
            'doctor.name': doctor?.name,
            'doctor.email': doctor?.email,
            'appointment.date': appointment?.date,
            'appointment.time': appointment?.time
        };

        const missingFields = Object.entries(requiredFields)
            .filter(([key, value]) => !value)
            .map(([key]) => key);

        if (missingFields.length > 0) {
            throw new Error(`Missing required fields: ${missingFields.join(', ')}`);
        }

        // Email validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(doctor.email)) {
            throw new Error('Invalid doctor email address');
        }

        return true;
    }

    // Main function to send appointment notification email
    async sendAppointmentNotification(appointmentData) {
        try {
            console.log('📧 Preparing to send appointment notification email...');

            // Validate input data
            this.validateAppointmentData(appointmentData);

            // Check if email service is available
            if (!this.transporter) {
                throw new Error('Email service not initialized');
            }

            // Verify email configuration
            await this.verifyEmailConfig();

            const { patient, doctor, appointment } = appointmentData;

            // Generate email content
            const htmlContent = this.generateEmailTemplate(appointmentData);
            const textContent = this.generatePlainTextTemplate(appointmentData);

            // Prepare email options
            const mailOptions = {
                from: {
                    name: 'Hospital Management System',
                    address: process.env.EMAIL_USER
                },
                to: doctor.email,
                subject: `🏥 New Appointment: ${patient.name} - ${this.formatDate(appointment.date)} at ${appointment.time}`,
                html: htmlContent,
                text: textContent,
                priority: 'normal',
                headers: {
                    'X-Mailer': 'Hospital Appointment System',
                    'X-Priority': '3',
                    'X-MSMail-Priority': 'Normal'
                }
            };

            // Send email
            console.log(`📤 Sending email to Dr. ${doctor.name} at ${doctor.email}...`);
            const info = await this.transporter.sendMail(mailOptions);

            console.log('✅ Email sent successfully!');
            console.log(`📧 Message ID: ${info.messageId}`);
            console.log(`🎯 Recipient: ${doctor.email}`);
            console.log(`👤 Patient: ${patient.name}`);
            console.log(`📅 Appointment: ${appointment.date} at ${appointment.time}`);

            return {
                success: true,
                messageId: info.messageId,
                recipient: doctor.email,
                sentAt: new Date().toISOString(),
                patient: patient.name,
                doctor: doctor.name
            };

        } catch (error) {
            console.error('❌ Error sending appointment notification email:', error);
            
            return {
                success: false,
                error: error.message,
                recipient: appointmentData?.doctor?.email,
                sentAt: new Date().toISOString(),
                patient: appointmentData?.patient?.name,
                doctor: appointmentData?.doctor?.name
            };
        }
    }

    // Send test email to verify configuration
    async sendTestEmail(testEmail = null) {
        try {
            const recipient = testEmail || process.env.EMAIL_USER;
            
            const testData = {
                patient: {
                    name: "John Doe",
                    age: 30,
                    gender: "Male",
                    contact: "john.doe@email.com",
                    issues: "Regular health checkup and consultation"
                },
                doctor: {
                    name: "Test Doctor",
                    email: recipient,
                    specialization: "General Physician"
                },
                appointment: {
                    date: new Date().toISOString().split('T')[0],
                    time: "10:00 AM",
                    duration: "15 minutes"
                }
            };

            console.log('🧪 Sending test email...');
            const result = await this.sendAppointmentNotification(testData);
            
            if (result.success) {
                console.log('✅ Test email sent successfully!');
            } else {
                console.log('❌ Test email failed:', result.error);
            }
            
            return result;
            
        } catch (error) {
            console.error('❌ Test email error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    // Get email service status
    getStatus() {
        return {
            initialized: !!this.transporter,
            configured: !!(process.env.EMAIL_USER && process.env.EMAIL_APP_PASSWORD),
            emailUser: process.env.EMAIL_USER,
            timestamp: new Date().toISOString()
        };
    }
}

// Export the EmailService class
module.exports = EmailService;

// Example usage:
/*
const EmailService = require('./emailService');

const emailService = new EmailService();

const appointmentData = {
    patient: {
        name: "John Smith",
        age: 25,
        gender: "Male",
        contact: "john@email.com",
        issues: "Headache and fever for 2 days"
    },
    doctor: {
        name: "Dr. Sarah Johnson",
        email: "doctor@hospital.com",
        specialization: "General Physician"
    },
    appointment: {
        date: "2025-06-20",
        time: "10:00 AM",
        duration: "15 minutes"
    }
};

// Send notification
emailService.sendAppointmentNotification(appointmentData)
    .then(result => {
        if (result.success) {
            console.log('Email sent successfully!');
        } else {
            console.log('Email failed:', result.error);
        }
    });
*/