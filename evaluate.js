#!/usr/bin/env node
/**
 * MedEasy — Specialty Classification Evaluation
 * Compares Gemini vs BART-MNLI pipeline accuracy.
 *
 * Usage:
 *   node evaluate.js                  # run both pipelines
 *   node evaluate.js --gemini-only    # skip BART
 *   node evaluate.js --bart-only      # skip Gemini
 *
 * Requires: server.js running on localhost:3000 and (for BART) main.py on localhost:8000
 */

const http = require('http');

const TEST_FILE  = './test_cases.json';
const SERVER_URL = process.env.SERVER_URL || 'http://localhost:3000';
const BART_URL   = process.env.BART_URL   || 'http://localhost:8000';

const args = process.argv.slice(2);
const RUN_GEMINI = !args.includes('--bart-only');
const RUN_BART   = !args.includes('--gemini-only');

function post(url, body) {
    return new Promise((resolve, reject) => {
        const parsed = new URL(url);
        const data = JSON.stringify(body);
        const opts = {
            hostname: parsed.hostname,
            port: parsed.port,
            path: parsed.pathname,
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
        };
        const req = http.request(opts, res => {
            let raw = '';
            res.on('data', c => raw += c);
            res.on('end', () => {
                try { resolve(JSON.parse(raw)); }
                catch { reject(new Error(`Bad JSON from ${url}: ${raw.substring(0, 200)}`)); }
            });
        });
        req.on('error', reject);
        req.setTimeout(30000, () => { req.destroy(); reject(new Error('Timeout')); });
        req.write(data);
        req.end();
    });
}

function get(url) {
    return new Promise((resolve, reject) => {
        const parsed = new URL(url);
        http.get({ hostname: parsed.hostname, port: parsed.port, path: parsed.pathname }, res => {
            let raw = '';
            res.on('data', c => raw += c);
            res.on('end', () => { try { resolve(JSON.parse(raw)); } catch { reject(new Error(`Bad JSON: ${raw.substring(0,200)}`)); } });
        }).on('error', reject);
    });
}

function normalizeSpec(s) {
    return (s || '').toLowerCase().replace(/[^a-z]/g, '');
}

function specsMatch(predicted, correct) {
    const p = normalizeSpec(predicted);
    const c = normalizeSpec(correct);
    if (p === c) return true;
    // Handle common aliases
    const aliases = {
        'generalpractitioner': 'generalphysician',
        'gp':                  'generalphysician',
        'gynecologist':        'gynaecologist',
        'gynecology':          'gynaecology',
        'dermatology':         'dermatologist',
        'cardiology':          'cardiologist',
        'urology':             'urologist',
        'ophthalmology':       'ophthalmologist',
    };
    const pn = aliases[p] || p;
    const cn = aliases[c] || c;
    return pn === cn || pn.includes(cn) || cn.includes(pn);
}

async function classifyGemini(symptoms, knowledgeBaseString, availableSpecializations) {
    const t0 = Date.now();
    try {
        const res = await post(`${SERVER_URL}/api/match-specialization`, {
            patientIssues: symptoms,
            availableSpecializations,
            knowledgeBaseString,
            preferredTime: null,
        });
        return {
            predicted: res.specialization || res.error || 'error',
            confidence: res.confidence || 0,
            latencyMs: Date.now() - t0,
            error: res.error,
        };
    } catch (e) {
        return { predicted: 'error', confidence: 0, latencyMs: Date.now() - t0, error: e.message };
    }
}

async function classifyBART(symptoms, availableSpecializations) {
    const t0 = Date.now();
    try {
        const hypotheses = availableSpecializations.map(s => `This patient requires ${s} care`);
        const res = await post(`${BART_URL}/classify`, {
            premise: symptoms,
            hypotheses,
            multi_label: false,
        });
        // Map top hypothesis back to specialty name
        const topHyp = res.top_label || '';
        const matched = availableSpecializations.find(s => topHyp.includes(s)) || availableSpecializations[0];
        return {
            predicted: matched,
            confidence: res.top_score || 0,
            latencyMs: Date.now() - t0,
            allScores: (res.labels || []).map((l, i) => ({
                label: availableSpecializations.find(s => l.includes(s)) || l,
                score: (res.scores || [])[i] || 0,
            })),
        };
    } catch (e) {
        return { predicted: 'error', confidence: 0, latencyMs: Date.now() - t0, error: e.message };
    }
}

function pad(s, len) {
    return String(s).padEnd(len).substring(0, len);
}

function bar(value, total, width = 20) {
    const filled = Math.round((value / total) * width);
    return '█'.repeat(filled) + '░'.repeat(width - filled);
}

function printRow(cols) {
    const COL = [30, 22, 22, 10];
    console.log('| ' + cols.map((c, i) => pad(c, COL[i] || 20)).join(' | ') + ' |');
}

async function main() {
    const testCases = require(TEST_FILE);
    console.log(`\n🏥 MedEasy Specialty Classification Evaluation`);
    console.log(`📊 Test cases: ${testCases.length}`);
    console.log(`🌐 Server: ${SERVER_URL}  |  BART: ${BART_URL}\n`);

    // Fetch knowledge base string once
    let knowledgeBaseString = '';
    let availableSpecializations = [];
    try {
        const kbRes = await get(`${SERVER_URL}/api/knowledge-base`);
        knowledgeBaseString = kbRes.knowledgeBaseString || '';
        availableSpecializations = kbRes.availableSpecializations || [
            'General Physician', 'Cardiologist', 'Dermatologist',
            'Urologist', 'Gynaecologist', 'Ophthalmologist'
        ];
    } catch (e) {
        availableSpecializations = [
            'General Physician', 'Cardiologist', 'Dermatologist',
            'Urologist', 'Gynaecologist', 'Ophthalmologist'
        ];
        console.warn('⚠️  Could not fetch knowledge base; using default specializations');
    }

    const geminiResults = [];
    const bartResults   = [];

    // Run evaluations sequentially to avoid hammering the server
    for (let i = 0; i < testCases.length; i++) {
        const tc = testCases[i];
        process.stdout.write(`\r  Evaluating case ${i + 1}/${testCases.length}...`);

        if (RUN_GEMINI) {
            const r = await classifyGemini(tc.symptoms, knowledgeBaseString, availableSpecializations);
            geminiResults.push({ ...tc, ...r, correct: specsMatch(r.predicted, tc.correct) });
            // Small delay to avoid rate limiting
            await new Promise(r => setTimeout(r, 500));
        }
        if (RUN_BART) {
            const r = await classifyBART(tc.symptoms, availableSpecializations);
            bartResults.push({ ...tc, ...r, correct: specsMatch(r.predicted, tc.correct) });
        }
    }
    process.stdout.write('\r' + ' '.repeat(50) + '\r');

    // Print results table
    const COL = [45, 22, 22, 10];
    const SEP = '+' + COL.map(c => '-'.repeat(c + 2)).join('+') + '+';

    console.log('\n' + SEP);
    printRow(['Symptoms (truncated)', 'Gemini', 'BART-MNLI', 'Correct?']);
    console.log(SEP);

    for (let i = 0; i < testCases.length; i++) {
        const tc = testCases[i];
        const gRes = geminiResults[i];
        const bRes = bartResults[i];
        const gLabel = gRes ? (gRes.error ? '❌ err' : `${gRes.predicted} (${(gRes.confidence * 100).toFixed(0)}%)`) : '-';
        const bLabel = bRes ? (bRes.error ? '❌ err' : `${bRes.predicted} (${(bRes.confidence * 100).toFixed(0)}%)`) : '-';
        const gOk = gRes && gRes.correct ? '✅' : (gRes ? '❌' : '-');
        const bOk = bRes && bRes.correct ? '✅' : (bRes ? '❌' : '-');
        const correctCell = `${RUN_GEMINI ? 'G:' + gOk : ''}${RUN_BART ? ' B:' + bOk : ''}`;
        printRow([tc.symptoms.substring(0, 43), gLabel.substring(0, 20), bLabel.substring(0, 20), correctCell]);
    }
    console.log(SEP);

    // Summary
    console.log('\n📈 Summary\n');
    function summarize(label, results) {
        if (!results.length) return;
        const correct = results.filter(r => r.correct).length;
        const accuracy = correct / results.length;
        const avgConf  = results.reduce((s, r) => s + (r.confidence || 0), 0) / results.length;
        const avgLatMs = results.reduce((s, r) => s + (r.latencyMs || 0), 0) / results.length;
        const errors   = results.filter(r => r.error).length;
        console.log(`  ${label}`);
        console.log(`  Accuracy  : ${bar(correct, results.length)} ${correct}/${results.length} (${(accuracy * 100).toFixed(1)}%)`);
        console.log(`  Avg Conf  : ${bar(avgConf, 1)} ${(avgConf * 100).toFixed(1)}%`);
        console.log(`  Avg Latency: ${avgLatMs.toFixed(0)} ms`);
        if (errors) console.log(`  Errors    : ${errors}`);
        console.log();

        // Per-specialty breakdown
        const specs = [...new Set(results.map(r => r.correct_display || r.correct))];
        const bySpec = {};
        for (const r of results) {
            const key = r.correct;
            if (!bySpec[key]) bySpec[key] = { total: 0, ok: 0 };
            bySpec[key].total++;
            if (r.correct) bySpec[key].ok++;
        }
        console.log(`  Per-specialty:`);
        for (const [spec, { total, ok }] of Object.entries(bySpec)) {
            console.log(`    ${pad(spec, 22)} ${bar(ok, total, 10)} ${ok}/${total}`);
        }
        console.log();
    }

    let geminiReport = null, bartReport = null;
    if (RUN_GEMINI) { summarize('Gemini (LLM prompt)', geminiResults); geminiReport = classificationReport('Gemini', geminiResults); }
    if (RUN_BART)   { summarize('BART-MNLI (NLI classifier)', bartResults); bartReport = classificationReport('BART-MNLI', bartResults); }

    if (RUN_GEMINI && RUN_BART && geminiResults.length && bartResults.length) {
        const gAcc = geminiResults.filter(r => r.correct).length / geminiResults.length;
        const bAcc = bartResults.filter(r => r.correct).length / bartResults.length;
        const winner = gAcc > bAcc ? 'Gemini' : bAcc > gAcc ? 'BART-MNLI' : 'Tie';
        console.log(`  🏆 Better accuracy: ${winner}`);
        console.log(`     Gemini: ${(gAcc * 100).toFixed(1)}%  |  BART-MNLI: ${(bAcc * 100).toFixed(1)}%\n`);
    }

    // Write a machine-readable report alongside the console output
    const fs = require('fs');
    const outPath = process.env.REPORT_FILE || './evaluation-report.json';
    const report = {
        generatedAt: new Date().toISOString(),
        testCaseCount: testCases.length,
        gemini: RUN_GEMINI ? { results: geminiResults, report: geminiReport } : null,
        bart: RUN_BART ? { results: bartResults, report: bartReport } : null,
    };
    try {
        fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
        console.log(`  💾 Full report written to ${outPath}\n`);
    } catch (e) {
        console.warn(`  ⚠️  Could not write report file: ${e.message}\n`);
    }
}

// ── Precision / Recall / F1 (macro + weighted), per class, from a confusion matrix ──
function classificationReport(label, results) {
    if (!results.length) return null;

    const displayFor = {};
    const norm = s => normalizeSpec(s);

    const classes = new Set();
    for (const r of results) {
        classes.add(norm(r.correct));
        displayFor[norm(r.correct)] = r.correct;
        if (r.predicted && r.predicted !== 'error') {
            classes.add(norm(r.predicted));
            displayFor[norm(r.predicted)] = displayFor[norm(r.predicted)] || r.predicted;
        }
    }

    const stats = {};
    for (const c of classes) stats[c] = { tp: 0, fp: 0, fn: 0, support: 0 };

    for (const r of results) {
        const truth = norm(r.correct);
        const pred  = r.predicted && r.predicted !== 'error' ? norm(r.predicted) : null;
        stats[truth].support++;

        if (pred === null) {
            stats[truth].fn++;
            continue;
        }
        if (specsMatch(r.predicted, r.correct)) {
            stats[truth].tp++;
        } else {
            stats[truth].fn++;
            if (!stats[pred]) stats[pred] = { tp: 0, fp: 0, fn: 0, support: 0 };
            stats[pred].fp++;
        }
    }

    const perClass = [];
    let macroP = 0, macroR = 0, macroF1 = 0, weightedP = 0, weightedR = 0, weightedF1 = 0;
    const totalSupport = results.length;
    const classList = Object.keys(stats).sort();

    for (const c of classList) {
        const { tp, fp, fn, support } = stats[c];
        const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
        const recall    = (tp + fn) > 0 ? tp / (tp + fn) : 0;
        const f1        = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;
        perClass.push({ class: displayFor[c] || c, precision, recall, f1, support, tp, fp, fn });
        macroP += precision; macroR += recall; macroF1 += f1;
        weightedP += precision * support; weightedR += recall * support; weightedF1 += f1 * support;
    }
    const n = classList.length || 1;

    const report = {
        label,
        perClass,
        macro: { precision: macroP / n, recall: macroR / n, f1: macroF1 / n },
        weighted: {
            precision: totalSupport ? weightedP / totalSupport : 0,
            recall:    totalSupport ? weightedR / totalSupport : 0,
            f1:        totalSupport ? weightedF1 / totalSupport : 0,
        },
        accuracy: results.filter(r => r.correct).length / results.length,
    };

    console.log(`  ${label} — Precision / Recall / F1 (per class)\n`);
    printRow(['Specialty', 'Precision', 'Recall', 'F1']);
    for (const row of perClass) {
        printRow([
            row.class,
            `${(row.precision * 100).toFixed(1)}% (${row.tp}/${row.tp + row.fp})`,
            `${(row.recall * 100).toFixed(1)}% (${row.tp}/${row.tp + row.fn})`,
            `${(row.f1 * 100).toFixed(1)}%`,
        ]);
    }
    console.log(`\n  Macro avg    -> P: ${(report.macro.precision * 100).toFixed(1)}%  R: ${(report.macro.recall * 100).toFixed(1)}%  F1: ${(report.macro.f1 * 100).toFixed(1)}%`);
    console.log(`  Weighted avg -> P: ${(report.weighted.precision * 100).toFixed(1)}%  R: ${(report.weighted.recall * 100).toFixed(1)}%  F1: ${(report.weighted.f1 * 100).toFixed(1)}%`);
    console.log(`  Accuracy     -> ${(report.accuracy * 100).toFixed(1)}%\n`);

    return report;
}

main().catch(e => { console.error('Fatal:', e.message); process.exit(1); });
