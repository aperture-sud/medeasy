// doctorAuth.js - SQLite-backed authentication for doctor portal
// Manages doctor credentials (hashed passwords) separately from doctors.xlsx

const Database = require('better-sqlite3');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const path = require('path');

const DB_PATH = path.join(__dirname, 'doctor_auth.db');
const JWT_SECRET = process.env.JWT_SECRET || 'medeasy-doctor-secret-CHANGE-IN-PRODUCTION';
const JWT_EXPIRES_IN = '8h';

let db;

function getDb() {
    if (!db) {
        db = new Database(DB_PATH);
        db.exec(`
            CREATE TABLE IF NOT EXISTS doctor_credentials (
                id          TEXT PRIMARY KEY,
                email       TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name        TEXT NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login  DATETIME,
                must_change_password INTEGER DEFAULT 1,
                specialization TEXT DEFAULT '',
                clinic      TEXT DEFAULT '',
                contact_phone TEXT DEFAULT ''
            )
        `);
        // Add profile columns if upgrading from earlier schema
        for (const col of [
            "ALTER TABLE doctor_credentials ADD COLUMN specialization TEXT DEFAULT ''",
            "ALTER TABLE doctor_credentials ADD COLUMN clinic TEXT DEFAULT ''",
            "ALTER TABLE doctor_credentials ADD COLUMN contact_phone TEXT DEFAULT ''"
        ]) {
            try { db.exec(col); } catch (_) { /* column already exists */ }
        }
    }
    return db;
}

// Create or replace a doctor's credentials
function createDoctor(id, email, password, name) {
    const hash = bcrypt.hashSync(password, 10);
    const stmt = getDb().prepare(
        `INSERT OR REPLACE INTO doctor_credentials
         (id, email, password_hash, name, must_change_password)
         VALUES (?, ?, ?, ?, 1)`
    );
    return stmt.run(id, email, hash, name);
}

// Verify email + password; returns doctor object or null
function verifyDoctor(email, password) {
    const row = getDb().prepare(
        'SELECT * FROM doctor_credentials WHERE email = ?'
    ).get(email);

    if (!row) return null;
    if (!bcrypt.compareSync(password, row.password_hash)) return null;

    getDb().prepare(
        'UPDATE doctor_credentials SET last_login = CURRENT_TIMESTAMP WHERE id = ?'
    ).run(row.id);

    return {
        id: row.id,
        email: row.email,
        name: row.name,
        mustChangePassword: !!row.must_change_password
    };
}

// Change a doctor's password (clears must_change_password flag)
function changePassword(doctorId, newPassword) {
    const hash = bcrypt.hashSync(newPassword, 10);
    getDb().prepare(
        'UPDATE doctor_credentials SET password_hash = ?, must_change_password = 0 WHERE id = ?'
    ).run(hash, doctorId);
}

// Sign a JWT for the given doctor object
function generateToken(doctor) {
    return jwt.sign(
        { id: doctor.id, email: doctor.email, name: doctor.name },
        JWT_SECRET,
        { expiresIn: JWT_EXPIRES_IN }
    );
}

// Verify and decode a JWT; returns payload or null
function verifyToken(token) {
    try {
        return jwt.verify(token, JWT_SECRET);
    } catch (_) {
        return null;
    }
}

// Fetch a doctor's profile (no password hash)
function getDoctorById(id) {
    return getDb().prepare(
        'SELECT id, email, name, specialization, clinic, contact_phone, last_login, must_change_password FROM doctor_credentials WHERE id = ?'
    ).get(id);
}

// Update editable profile fields
function updateProfile(doctorId, { name, specialization, clinic, contact_phone }) {
    getDb().prepare(
        'UPDATE doctor_credentials SET name = ?, specialization = ?, clinic = ?, contact_phone = ? WHERE id = ?'
    ).run(name, specialization, clinic, contact_phone, doctorId);
}

// List all registered doctors (for admin use)
function listDoctors() {
    return getDb().prepare(
        'SELECT id, email, name, last_login, must_change_password FROM doctor_credentials ORDER BY name'
    ).all();
}

module.exports = {
    getDb,
    createDoctor,
    verifyDoctor,
    changePassword,
    generateToken,
    verifyToken,
    getDoctorById,
    updateProfile,
    listDoctors
};
