// doctorAuthMiddleware.js - JWT verification middleware for /api/doctor/* routes

const { verifyToken } = require('./doctorAuth');

function doctorAuthMiddleware(req, res, next) {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.startsWith('Bearer ')
        ? authHeader.slice(7)
        : null;

    if (!token) {
        return res.status(401).json({ error: 'Authentication required', code: 'NO_TOKEN' });
    }

    const decoded = verifyToken(token);
    if (!decoded) {
        return res.status(401).json({ error: 'Token is invalid or expired', code: 'INVALID_TOKEN' });
    }

    req.doctor = decoded; // { id, email, name }
    next();
}

module.exports = doctorAuthMiddleware;
