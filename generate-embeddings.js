#!/usr/bin/env node
/**
 * generate-embeddings.js
 * One-time script: reads all *.txt transcript files in prompts/<Specialty>/,
 * chunks them into dialogue windows, calls Ollama /api/embeddings, and saves
 * the result to embeddings.json.
 *
 * Usage:
 *   ollama pull nomic-embed-text   # one-time
 *   node generate-embeddings.js
 *
 * Output: embeddings.json  (array of { specialty, file, chunkIndex, text, embedding })
 */

const fs = require('fs');
const path = require('path');
const axios = require('axios');

const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const EMBED_MODEL = 'nomic-embed-text';
const PROMPTS_DIR = path.resolve('./prompts');
const OUTPUT_FILE = path.resolve('./embeddings.json');

// ── Tuning knobs ──────────────────────────────────────────────────────────────
const CHUNK_LINES = 6;    // doctor-question lines per chunk (tighter, more focused)
const CHUNK_STEP  = 3;    // sliding window step (50% overlap for better recall)
const BATCH_SIZE  = 8;    // concurrent embedding requests (respect Ollama RAM)
// ─────────────────────────────────────────────────────────────────────────────

async function embedText(text) {
    const res = await axios.post(
        `${OLLAMA_URL}/api/embeddings`,
        { model: EMBED_MODEL, prompt: text },
        { timeout: 30000 }
    );
    return res.data.embedding; // float[]
}

function chunkTranscript(content) {
    // Only keep doctor lines that are actual questions — richer embedding signal
    const lines = content
        .split('\n')
        .map(l => l.trim())
        .filter(l => l.startsWith('D:') && l.includes('?'));

    const chunks = [];
    for (let i = 0; i < lines.length; i += CHUNK_STEP) {
        const slice = lines.slice(i, i + CHUNK_LINES);
        if (slice.length < 2) break; // skip tiny tail fragments
        chunks.push(slice.join('\n'));
    }
    return chunks;
}

async function processBatch(items) {
    return Promise.all(items.map(async item => {
        try {
            item.embedding = await embedText(item.text);
        } catch (e) {
            console.error(`  ✗ embedding failed for ${item.file} chunk ${item.chunkIndex}: ${e.message}`);
            item.embedding = null;
        }
        return item;
    }));
}

async function main() {
    // ── Verify Ollama + model ─────────────────────────────────────────────────
    try {
        const tags = await axios.get(`${OLLAMA_URL}/api/tags`, { timeout: 5000 });
        const models = (tags.data.models || []).map(m => m.name);
        if (!models.some(m => m.startsWith('nomic-embed-text'))) {
            console.error(`❌ Model not found. Run: ollama pull ${EMBED_MODEL}`);
            process.exit(1);
        }
        console.log(`✅ Ollama running — using ${EMBED_MODEL}`);
    } catch {
        console.error('❌ Ollama not reachable. Run: ollama serve');
        process.exit(1);
    }

    // ── Collect all chunks ────────────────────────────────────────────────────
    const specialties = fs.readdirSync(PROMPTS_DIR)
        .filter(d => fs.statSync(path.join(PROMPTS_DIR, d)).isDirectory());

    const allItems = [];
    for (const specialty of specialties) {
        const dir = path.join(PROMPTS_DIR, specialty);
        const files = fs.readdirSync(dir).filter(f => f.endsWith('.txt')).sort();
        console.log(`📂 ${specialty}: ${files.length} file(s)`);
        for (const file of files) {
            const content = fs.readFileSync(path.join(dir, file), 'utf8');
            const chunks = chunkTranscript(content);
            chunks.forEach((text, chunkIndex) => {
                allItems.push({ specialty, file, chunkIndex, text, embedding: null });
            });
        }
    }
    console.log(`\n📊 Total chunks to embed: ${allItems.length}`);

    // ── Embed in batches ──────────────────────────────────────────────────────
    const results = [];
    for (let i = 0; i < allItems.length; i += BATCH_SIZE) {
        const batch = allItems.slice(i, i + BATCH_SIZE);
        process.stdout.write(`  Embedding ${i + 1}–${Math.min(i + BATCH_SIZE, allItems.length)} / ${allItems.length}...`);
        const done = await processBatch(batch);
        results.push(...done.filter(r => r.embedding !== null));
        process.stdout.write(' ✓\n');
    }

    // ── Save ──────────────────────────────────────────────────────────────────
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2));
    console.log(`\n✅ Saved ${results.length} embeddings → ${OUTPUT_FILE}`);
    console.log(`   (${allItems.length - results.length} chunks skipped due to errors)`);
}

main().catch(e => { console.error('Fatal:', e.message); process.exit(1); });
