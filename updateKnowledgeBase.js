// updateKnowledgeBase.js - Update the knowledge base file with latest booked appointments (robust header-based)
const XLSX = require('xlsx');
const fs = require('fs');
const path = require('path');

class KnowledgeBaseUpdater {
    constructor(filePath = './doctors.xlsx') {
        this.filePath = filePath;
    }

    // Helper: normalize doctor names for matching (remove "Dr.", punctuation, lowercase, trim)
    normalizeName(s) {
        if (!s) return '';
        return s.toString().replace(/\bDr\.?\b/ig, '').replace(/[^\w\s]/g, '').toLowerCase().trim();
    }

    // Find header key by matching any of the substrings (case-insensitive)
    findKey(objKeys, candidates) {
        const lowerKeys = objKeys.map(k => k.toLowerCase());
        for (const c of candidates) {
            const lc = c.toLowerCase();
            const idx = lowerKeys.findIndex(k => k.includes(lc));
            if (idx !== -1) return objKeys[idx];
        }
        return null;
    }

    // Update latest booked slot for a specific doctor in Excel file
    async updateExcelFile(doctorName, appointmentDate, appointmentTime) {
        try {
            console.log(`📝 Updating Excel file for Dr. ${doctorName}...`);

            if (!fs.existsSync(this.filePath)) {
                throw new Error(`Knowledge base file not found: ${this.filePath}`);
            }

            // Read workbook and first sheet
            const workbook = XLSX.readFile(this.filePath);
            const sheetName = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[sheetName];

            // Convert sheet to array of objects using header row as keys
            const data = XLSX.utils.sheet_to_json(worksheet, { defval: '' });

            if (!Array.isArray(data) || data.length === 0) {
                throw new Error('Knowledge base appears empty or header row missing');
            }

            // Determine header keys in the object (keys from first row)
            const objKeys = Object.keys(data[0]);

            // Identify which key is "name", "latest booked slot", etc.
            const nameKey = this.findKey(objKeys, ['name', 'doctor name', 'full name']);
            const emailKey = this.findKey(objKeys, ['email', 'e-mail']);
            const specializationKey = this.findKey(objKeys, ['special', 'specialization', 'field']);
            const availabilityKey = this.findKey(objKeys, ['avail', 'available', 'availability', 'slot']);
            let latestKey = this.findKey(objKeys, ['latest booked', 'latestbooked', 'latest_booked', 'latestBookedSlot', 'latest']);

            // If LatestBookedSlot not present, add a new header key to objects
            if (!latestKey) {
                latestKey = 'LatestBookedSlot';
                for (let r of data) {
                    r[latestKey] = r[latestKey] || 'NIL';
                }
            }

            // Normalize target name for matching
            const targetNorm = this.normalizeName(doctorName);

            // Find and update the row (case-insensitive, normalized)
            let doctorFound = false;
            const newBookedSlot = `${appointmentDate} ${appointmentTime}`;

            for (let i = 0; i < data.length; i++) {
                const row = data[i];
                const nameVal = nameKey ? (row[nameKey] || '') : '';
                const normRowName = this.normalizeName(nameVal);
                if (!normRowName) continue;

                // Exact normalized match or name inclusion
                if (normRowName === targetNorm || normRowName.includes(targetNorm) || targetNorm.includes(normRowName)) {
                    row[latestKey] = newBookedSlot;
                    doctorFound = true;
                    console.log(`✅ Updated ${nameVal}'s ${latestKey} to: ${newBookedSlot}`);
                    break;
                }
            }

            if (!doctorFound) {
                // Try matching by email or partial match fallback (search any field)
                for (let i = 0; i < data.length && !doctorFound; i++) {
                    const row = data[i];
                    const combined = Object.values(row).join(' ').toLowerCase();
                    if (combined.includes(doctorName.toLowerCase())) {
                        row[latestKey] = newBookedSlot;
                        doctorFound = true;
                        console.log(`✅ Fallback updated row ${i} to ${newBookedSlot}`);
                        break;
                    }
                }
            }

            if (!doctorFound) {
                throw new Error(`Doctor "${doctorName}" not found in knowledge base`);
            }

            // Write back to sheet (json -> sheet)
            const newWorksheet = XLSX.utils.json_to_sheet(data, { skipHeader: false });
            workbook.Sheets[sheetName] = newWorksheet;

            // Create backup first
            try {
                const backupPath = this.filePath.replace(path.extname(this.filePath), `_backup_${Date.now()}${path.extname(this.filePath)}`);
                fs.copyFileSync(this.filePath, backupPath);
                console.log(`💾 Backup created: ${backupPath}`);
            } catch (e) {
                console.warn('⚠️ Could not create backup (continuing):', e.message || e);
            }

            // Write file
            XLSX.writeFile(workbook, this.filePath);
            console.log(`💾 Knowledge base Excel file updated successfully: ${this.filePath}`);

            return {
                success: true,
                doctorName: doctorName,
                updatedSlot: newBookedSlot,
                filePath: this.filePath
            };

        } catch (error) {
            console.error('❌ Error updating Excel knowledge base:', error);
            return {
                success: false,
                error: error.message || String(error),
                doctorName: doctorName
            };
        }
    }

    // Update latest booked slot for a specific doctor in CSV file (header-aware)
    async updateCSVFile(doctorName, appointmentDate, appointmentTime) {
        try {
            console.log(`📝 Updating CSV file for Dr. ${doctorName}...`);

            if (!fs.existsSync(this.filePath)) {
                throw new Error(`Knowledge base file not found: ${this.filePath}`);
            }

            const csvData = fs.readFileSync(this.filePath, 'utf8');
            const lines = csvData.split(/\r?\n/);

            if (lines.length < 2) throw new Error('CSV knowledge base seems empty or missing header');

            // Parse header and build array of objects
            const header = this.parseCSVLine(lines[0]);
            const rows = [];
            for (let i = 1; i < lines.length; i++) {
                if (!lines[i].trim()) continue;
                const cols = this.parseCSVLine(lines[i]);
                const obj = {};
                for (let j = 0; j < header.length; j++) {
                    obj[header[j]] = cols[j] || '';
                }
                rows.push(obj);
            }

            // Determine which header is latest booked key
            const headerKeys = Object.keys(rows[0] || {});
            let latestKey = this.findKey(headerKeys, ['latest booked', 'latestbooked', 'latest_booked', 'latestBookedSlot', 'latest']);
            if (!latestKey) {
                latestKey = 'LatestBookedSlot';
                // add to header and rows
                header.push(latestKey);
                for (let r of rows) r[latestKey] = r[latestKey] || 'NIL';
            }

            const targetNorm = this.normalizeName(doctorName);
            let doctorFound = false;
            const newBookedSlot = `${appointmentDate} ${appointmentTime}`;

            for (let r of rows) {
                const nameKey = this.findKey(Object.keys(r), ['name', 'doctor name', 'full name']);
                const nameVal = nameKey ? (r[nameKey] || '') : '';
                const norm = this.normalizeName(nameVal);
                if (!norm) continue;
                if (norm === targetNorm || norm.includes(targetNorm) || targetNorm.includes(norm)) {
                    r[latestKey] = newBookedSlot;
                    doctorFound = true;
                    break;
                }
            }

            if (!doctorFound) {
                // fallback: any field contains doctorName
                for (let r of rows) {
                    const combined = Object.values(r).join(' ').toLowerCase();
                    if (combined.includes(doctorName.toLowerCase())) {
                        r[latestKey] = newBookedSlot;
                        doctorFound = true;
                        break;
                    }
                }
            }

            if (!doctorFound) {
                throw new Error(`Doctor "${doctorName}" not found in CSV knowledge base`);
            }

            // Reconstruct CSV
            const newLines = [];
            newLines.push(header.join(','));
            for (let r of rows) {
                const rowArr = header.map(h => {
                    const v = (r[h] || '').toString();
                    return v.includes(',') ? `"${v.replace(/"/g, '""')}"` : v;
                });
                newLines.push(rowArr.join(','));
            }

            // Backup then write
            try {
                const backupPath = this.filePath.replace(path.extname(this.filePath), `_backup_${Date.now()}${path.extname(this.filePath)}`);
                fs.copyFileSync(this.filePath, backupPath);
                console.log(`💾 CSV Backup created: ${backupPath}`);
            } catch (e) {
                console.warn('⚠️ Could not create CSV backup (continuing):', e.message || e);
            }

            fs.writeFileSync(this.filePath, newLines.join('\n'), 'utf8');
            console.log(`💾 Knowledge base CSV file updated successfully: ${this.filePath}`);

            return { success: true, doctorName: doctorName, updatedSlot: newBookedSlot, filePath: this.filePath };
        } catch (error) {
            console.error('❌ Error updating CSV knowledge base:', error);
            return { success: false, error: error.message || String(error), doctorName: doctorName };
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

    // Auto-detect file type and update accordingly
    async updateKnowledgeBase(doctorName, appointmentDate, appointmentTime) {
        try {
            console.log(`🔄 Updating knowledge base for appointment booking...`);
            console.log(`👨‍⚕️ Doctor: ${doctorName}`);
            console.log(`📅 Date: ${appointmentDate}`);
            console.log(`⏰ Time: ${appointmentTime}`);

            const extension = path.extname(this.filePath).toLowerCase();

            let result;
            if (extension === '.xlsx' || extension === '.xls') {
                result = await this.updateExcelFile(doctorName, appointmentDate, appointmentTime);
            } else if (extension === '.csv') {
                result = await this.updateCSVFile(doctorName, appointmentDate, appointmentTime);
            } else {
                throw new Error(`Unsupported file format: ${extension}. Use .xlsx, .xls, or .csv`);
            }

            if (result.success) {
                console.log(`✅ Knowledge base update completed successfully`);
                console.log(`📊 File: ${result.filePath}`);
                console.log(`🩺 Doctor: ${result.doctorName}`);
                console.log(`📋 Latest Slot: ${result.updatedSlot}`);
            }

            return result;

        } catch (error) {
            console.error('❌ Error updating knowledge base:', error);
            return {
                success: false,
                error: error.message || String(error),
                doctorName: doctorName
            };
        }
    }

    // Update the Rating and ReviewCount columns for a doctor in the Excel file.
    async updateDoctorRating(doctorName, averageRating, reviewCount) {
        try {
            if (!fs.existsSync(this.filePath)) return { success: false, error: 'File not found' };
            const workbook = XLSX.readFile(this.filePath);
            const sheetName = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[sheetName];
            const data = XLSX.utils.sheet_to_json(worksheet, { defval: '' });
            if (!data.length) return { success: false, error: 'Empty sheet' };

            const objKeys = Object.keys(data[0]);
            const nameKey = this.findKey(objKeys, ['name', 'doctor name', 'full name']);
            let ratingKey = this.findKey(objKeys, ['rating', 'stars', 'score', 'rate']);
            let reviewKey = this.findKey(objKeys, ['review', 'reviews', 'count', 'total']);
            if (!ratingKey) {
                ratingKey = 'Rating';
                for (const r of data) r[ratingKey] = r[ratingKey] || '';
            }
            if (!reviewKey) {
                reviewKey = 'ReviewCount';
                for (const r of data) r[reviewKey] = r[reviewKey] || '';
            }

            const targetNorm = this.normalizeName(doctorName);
            let found = false;
            for (const row of data) {
                const nameVal = nameKey ? (row[nameKey] || '') : '';
                const norm = this.normalizeName(nameVal);
                if (!norm) continue;
                if (norm === targetNorm || norm.includes(targetNorm) || targetNorm.includes(norm)) {
                    row[ratingKey] = averageRating;
                    row[reviewKey] = reviewCount;
                    found = true;
                    break;
                }
            }
            if (!found) return { success: false, error: `Doctor "${doctorName}" not found` };

            workbook.Sheets[sheetName] = XLSX.utils.json_to_sheet(data);
            XLSX.writeFile(workbook, this.filePath);
            console.log(`⭐ Updated ${doctorName} rating: ${averageRating} (${reviewCount} reviews)`);
            return { success: true };
        } catch (e) {
            console.error('❌ updateDoctorRating error:', e);
            return { success: false, error: e.message };
        }
    }

    // Backup the knowledge base file before updating (kept for compatibility)
    async createBackup() {
        try {
            const backupPath = this.filePath.replace(
                path.extname(this.filePath),
                `_backup_${Date.now()}${path.extname(this.filePath)}`
            );

            fs.copyFileSync(this.filePath, backupPath);
            console.log(`💾 Backup created: ${backupPath}`);
            return backupPath;
        } catch (error) {
            console.warn('⚠️ Could not create backup:', error.message || error);
            return null;
        }
    }
}

module.exports = KnowledgeBaseUpdater;
