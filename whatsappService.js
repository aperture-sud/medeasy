// whatsappService.js - Twilio WhatsApp helper
const twilio = require('twilio');

class WhatsAppService {
  constructor() {
    this.accountSid = process.env.TWILIO_ACCOUNT_SID;
    this.authToken = process.env.TWILIO_AUTH_TOKEN;
    this.from = process.env.TWILIO_WHATSAPP_FROM; // must be like 'whatsapp:+1415xxxxxxx'
    this.defaultCountry = process.env.DEFAULT_COUNTRY_CODE || '+91';
    if (this.accountSid && this.authToken && this.from) {
      this.client = twilio(this.accountSid, this.authToken);
      console.log('✅ WhatsAppService (Twilio) initialized');
    } else {
      this.client = null;
      console.warn('⚠️ WhatsAppService not configured (TWILIO_ env vars missing). WhatsApp messages will be disabled.');
    }
  }

  // Very simple phone formatting helper.
  // If contact starts with +, assume it's E.164 and use it.
  // Otherwise, keep last 10 digits and prefix with DEFAULT_COUNTRY_CODE.
  formatToWhatsApp(contact) {
    if (!contact) return null;
    let s = contact.toString().trim();
    if (s.startsWith('+')) {
      return `whatsapp:${s}`;
    }
    // remove non-digits
    const digits = s.replace(/\D/g, '');
    if (digits.length >= 10) {
      const last10 = digits.slice(-10);
      return `whatsapp:${this.defaultCountry}${last10}`;
    }
    return null;
  }

  async sendMessageTo(contact, body) {
    if (!this.client) throw new Error('Twilio client not configured');
    const to = this.formatToWhatsApp(contact);
    if (!to) throw new Error('Invalid contact phone number for WhatsApp: ' + contact);
    return this.client.messages.create({
      from: this.from,
      to,
      body
    });
  }

  async sendAppointmentConfirmation(appointment) {
    const patientName = (appointment.patient && appointment.patient.name) ? appointment.patient.name : '';
    const contact = appointment.patient && appointment.patient.contact;
    const body = `📅 *Appointment Confirmed*
Doctor: ${appointment.doctorName}
Patient: ${patientName}
Date: ${appointment.appointmentDate}
Time: ${appointment.appointmentTime}

If you need to cancel, reply to this message or contact the clinic.`;
    try {
      const resp = await this.sendMessageTo(contact, body);
      console.log('✅ WhatsApp confirmation sent, sid=', resp.sid);
      return { success: true, sid: resp.sid };
    } catch (e) {
      console.warn('❌ Failed to send WhatsApp confirmation:', e.message || e);
      return { success: false, error: e.message || String(e) };
    }
  }

  async sendAppointmentReminder(appointment) {
    const patientName = (appointment.patient && appointment.patient.name) ? appointment.patient.name : '';
    const contact = appointment.patient && appointment.patient.contact;
    const body = `⏰ *Appointment Reminder*
Hello ${patientName},
This is a reminder for your appointment with ${appointment.doctorName}.
Date: ${appointment.appointmentDate}
Time: ${appointment.appointmentTime}

Please arrive ~10 minutes early.`;
    try {
      const resp = await this.sendMessageTo(contact, body);
      console.log('✅ WhatsApp reminder sent, sid=', resp.sid);
      return { success: true, sid: resp.sid };
    } catch (e) {
      console.warn('❌ Failed to send WhatsApp reminder:', e.message || e);
      return { success: false, error: e.message || String(e) };
    }
  }
}

module.exports = WhatsAppService;
