// setup-doctor-auth.js
// Run ONCE to seed doctor credentials from doctors.xlsx into doctor_auth.db
// Usage: node setup-doctor-auth.js
// Default password for all doctors: doctor@123
// Doctors should change their password on first login.

const { createDoctor, listDoctors } = require('./doctorAuth');
const XLSX = require('xlsx');
const path = require('path');

const XLSX_PATH = path.join(__dirname, 'doctors.xlsx');
const DEFAULT_PASSWORD = 'doctor@123';

console.log('🔐 Medeasy Doctor Auth Setup');
console.log('='.repeat(40));

let workbook;
try {
    workbook = XLSX.readFile(XLSX_PATH);
} catch (err) {
    console.error('❌ Cannot read doctors.xlsx:', err.message);
    process.exit(1);
}

const sheet = workbook.Sheets[workbook.SheetNames[0]];
const doctors = XLSX.utils.sheet_to_json(sheet);

if (!doctors.length) {
    console.error('❌ No doctors found in doctors.xlsx');
    process.exit(1);
}

console.log(`Found ${doctors.length} doctors in doctors.xlsx\n`);

let created = 0, skipped = 0;

for (const doctor of doctors) {
    // Normalize field names (handle different capitalizations)
    const id    = doctor.ID    || doctor.id    || doctor.DoctorID;
    const email = doctor.Email || doctor.email || doctor.EMAIL;
    const name  = doctor.Name  || doctor.name  || doctor.NAME;

    if (!id || !email || !name) {
        console.log(`⚠️  Skipping incomplete row:`, JSON.stringify(doctor));
        skipped++;
        continue;
    }

    try {
        createDoctor(id, email.trim(), DEFAULT_PASSWORD, name.trim());
        console.log(`✅  ${name.padEnd(30)} (${id})  →  ${email}`);
        created++;
    } catch (err) {
        console.error(`❌  Failed for ${name}: ${err.message}`);
        skipped++;
    }
}

console.log('\n' + '='.repeat(40));
console.log(`✅  Created : ${created}`);
console.log(`⚠️  Skipped : ${skipped}`);
console.log('\n🔑 Default password for all doctors: ' + DEFAULT_PASSWORD);
console.log('⚠️  Doctors should change their password after first login.');

const all = listDoctors();
console.log(`\n📋 Total credentials in DB: ${all.length}`);
