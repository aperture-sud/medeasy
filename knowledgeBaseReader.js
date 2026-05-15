// knowledgeBaseReader.js - Read doctor information from Excel/CSV file (header-aware)
const XLSX = require('xlsx');
const fs = require('fs');
const path = require('path');

class KnowledgeBaseReader {
    constructor(filePath = './doctors.xlsx') {
        this.filePath = filePath;
        this.doctors = [];
    }

    // Helper to find header key by candidate substrings
    findKey(objKeys, candidates) {
        const lk = objKeys.map(k => k.toLowerCase());
        for (const c of candidates) {
            const cc = c.toLowerCase();
            const idx = lk.findIndex(k => k.includes(cc));
            if (idx !== -1) return objKeys[idx];
        }
        return null;
    }

    // Generate a consistent pseudo-rating (3.5–5.0) from doctor name when no rating in data
    _pseudoRating(name) {
        if (!name) return 4.0;
        let h = 0;
        for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) & 0xffff;
        return Math.round((3.5 + (h % 16) / 10) * 10) / 10; // 3.5–5.0
    }

    // Normalize name for matching
    normalizeName(s) {
        if (!s) return '';
        return s.toString().replace(/\bDr\.?\b/ig, '').replace(/[^\w\s]/g, '').toLowerCase().trim();
    }

    // Read doctors data from Excel file (header-aware)
    async readFromExcel() {
        try {
            console.log('📚 Reading knowledge base from Excel file:', this.filePath);

            if (!fs.existsSync(this.filePath)) {
                throw new Error(`Knowledge base file not found: ${this.filePath}`);
            }

            const workbook = XLSX.readFile(this.filePath);
            const sheetName = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[sheetName];

            // Convert to array of objects (headers from first row)
            const rows = XLSX.utils.sheet_to_json(worksheet, { defval: '' });

            if (!Array.isArray(rows) || rows.length === 0) {
                throw new Error('Knowledge base file appears to be empty or header row missing');
            }

            const headerKeys = Object.keys(rows[0]);

            // Determine keys by common names
            const nameKey = this.findKey(headerKeys, ['name', 'doctor name', 'full name']);
            const emailKey = this.findKey(headerKeys, ['email', 'e-mail']);
            const specializationKey = this.findKey(headerKeys, ['special', 'specialization', 'field']);
            const availabilityKey = this.findKey(headerKeys, ['avail', 'available', 'availability', 'slot']);
            let latestKey = this.findKey(headerKeys, ['latest booked', 'latestbooked', 'latest_booked', 'latestbookedslot', 'latest']);
            const ratingKey = this.findKey(headerKeys, ['rating', 'stars', 'score', 'rate']);

            // If latest key not present, later code will treat missing as 'NIL'
            this.doctors = [];

            for (const row of rows) {
                const nameVal = nameKey ? (row[nameKey] || '') : '';
                const emailVal = emailKey ? (row[emailKey] || '') : '';
                const specializationVal = specializationKey ? (row[specializationKey] || '') : '';
                const availabilityVal = availabilityKey ? (row[availabilityKey] || '') : '';
                const latestVal = latestKey ? (row[latestKey] || 'NIL') : (row['LatestBookedSlot'] || row['Latest Booked Slot'] || 'NIL');
                const ratingVal = ratingKey ? parseFloat(row[ratingKey]) : null;

                const doctor = {
                    name: nameVal ? nameVal.toString().trim() : '',
                    email: emailVal ? emailVal.toString().trim() : '',
                    specialization: specializationVal ? specializationVal.toString().trim() : '',
                    availability: availabilityVal ? availabilityVal.toString().trim() : '',
                    latestBookedSlot: latestVal ? latestVal.toString().trim() : 'NIL',
                    rating: (ratingVal && !isNaN(ratingVal) && ratingVal >= 1 && ratingVal <= 5)
                        ? Math.round(ratingVal * 10) / 10
                        : this._pseudoRating(nameVal)
                };

                // Add only if basic info present (name + specialization recommended)
                if (doctor.name && doctor.specialization) {
                    this.doctors.push(doctor);
                    console.log(`✅ Loaded doctor: ${doctor.name} (${doctor.specialization})`);
                } else {
                    // still may be valid if name present; we skip rows without name
                    if (doctor.name) {
                        this.doctors.push(doctor);
                        console.log(`⚠️ Loaded doctor with partial info: ${doctor.name}`);
                    }
                }
            }

            console.log(`📊 Total doctors loaded: ${this.doctors.length}`);
            return this.doctors;

        } catch (error) {
            console.error('❌ Error reading knowledge base from Excel:', error);
            throw error;
        }
    }

    // Read doctors data from CSV file (header-aware)
    async readFromCSV() {
        try {
            console.log('📚 Reading knowledge base from CSV file:', this.filePath);

            if (!fs.existsSync(this.filePath)) {
                throw new Error(`Knowledge base file not found: ${this.filePath}`);
            }

            const csvData = fs.readFileSync(this.filePath, 'utf8');
            const lines = csvData.split(/\r?\n/).filter(l => l.trim().length > 0);

            if (lines.length < 2) {
                throw new Error('Knowledge base file appears to be empty or invalid');
            }

            // parse header
            const header = this.parseCSVLine(lines[0]);
            const headerKeys = header;

            // parse rows to objects
            const rows = [];
            for (let i = 1; i < lines.length; i++) {
                const cols = this.parseCSVLine(lines[i]);
                const obj = {};
                for (let j = 0; j < headerKeys.length; j++) {
                    obj[headerKeys[j]] = cols[j] || '';
                }
                rows.push(obj);
            }

            const nameKey = this.findKey(headerKeys, ['name', 'doctor name', 'full name']);
            const emailKey = this.findKey(headerKeys, ['email', 'e-mail']);
            const specializationKey = this.findKey(headerKeys, ['special', 'specialization', 'field']);
            const availabilityKey = this.findKey(headerKeys, ['avail', 'available', 'availability', 'slot']);
            const latestKey = this.findKey(headerKeys, ['latest booked', 'latestbooked', 'latest_booked', 'latestbookedslot', 'latest']);
            const ratingKey = this.findKey(headerKeys, ['rating', 'stars', 'score', 'rate']);

            this.doctors = [];

            for (const row of rows) {
                const nameVal = nameKey ? (row[nameKey] || '') : '';
                const emailVal = emailKey ? (row[emailKey] || '') : '';
                const specializationVal = specializationKey ? (row[specializationKey] || '') : '';
                const availabilityVal = availabilityKey ? (row[availabilityKey] || '') : '';
                const latestVal = latestKey ? (row[latestKey] || 'NIL') : (row['LatestBookedSlot'] || row['Latest Booked Slot'] || 'NIL');
                const ratingVal = ratingKey ? parseFloat(row[ratingKey]) : null;

                const doctor = {
                    name: nameVal ? nameVal.toString().trim() : '',
                    email: emailVal ? emailVal.toString().trim() : '',
                    specialization: specializationVal ? specializationVal.toString().trim() : '',
                    availability: availabilityVal ? availabilityVal.toString().trim() : '',
                    latestBookedSlot: latestVal ? latestVal.toString().trim() : 'NIL',
                    rating: (ratingVal && !isNaN(ratingVal) && ratingVal >= 1 && ratingVal <= 5)
                        ? Math.round(ratingVal * 10) / 10
                        : this._pseudoRating(nameVal)
                };

                if (doctor.name && doctor.specialization) {
                    this.doctors.push(doctor);
                    console.log(`✅ Loaded doctor: ${doctor.name} (${doctor.specialization})`);
                } else {
                    if (doctor.name) {
                        this.doctors.push(doctor);
                        console.log(`⚠️ Loaded doctor with partial info: ${doctor.name}`);
                    }
                }
            }

            console.log(`📊 Total doctors loaded: ${this.doctors.length}`);
            return this.doctors;

        } catch (error) {
            console.error('❌ Error reading knowledge base from CSV:', error);
            throw error;
        }
    }

    // Parse CSV line handling commas inside quotes
    parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            if (char === '"') {
                if (inQuotes && line[i+1] === '"') { // escaped quote
                    current += '"';
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (char === ',' && !inQuotes) {
                result.push(current.trim());
                current = '';
            } else {
                current += char;
            }
        }
        result.push(current.trim());
        return result;
    }

    // Auto-detect file type and read accordingly
    async readKnowledgeBase() {
        const extension = path.extname(this.filePath).toLowerCase();

        if (extension === '.xlsx' || extension === '.xls') {
            return await this.readFromExcel();
        } else if (extension === '.csv') {
            return await this.readFromCSV();
        } else {
            throw new Error(`Unsupported file format: ${extension}. Use .xlsx, .xls, or .csv`);
        }
    }

    // Convert doctors array to knowledge base string format for LLM
    generateKnowledgeBaseString() {
        if (!this.doctors || this.doctors.length === 0) {
            throw new Error('No doctors loaded. Please read knowledge base first.');
        }

        const knowledgeBaseLines = this.doctors.map(doctor => {
            return `Doctor name : ${doctor.name}, Doctor specialization: ${doctor.specialization}, Doctor email: ${doctor.email}, Availability: ${doctor.availability}, Latest Booked Slot: ${doctor.latestBookedSlot || 'NIL'}`;
        });

        return knowledgeBaseLines.join('\n');
    }

    // Get doctors array
    getDoctors() {
        return this.doctors;
    }

    // Get doctor by name (case-insensitive)
    getDoctorByName(doctorName) {
        if (!doctorName) return null;
        const norm = this.normalizeName(doctorName);
        return this.doctors.find(doctor => this.normalizeName(doctor.name).includes(norm) || norm.includes(this.normalizeName(doctor.name)));
    }

    // Get doctors by specialization
    getDoctorsBySpecialization(specialization) {
        if (!specialization) return [];
        return this.doctors.filter(doctor =>
            doctor.specialization && doctor.specialization.toLowerCase().includes(specialization.toLowerCase())
        );
    }

    // Get available specializations
    getAvailableSpecializations() {
        return [...new Set(this.doctors.map(doctor => doctor.specialization))];
    }
}

module.exports = KnowledgeBaseReader;
