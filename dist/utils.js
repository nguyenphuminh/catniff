"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.erf = erf;
exports.erfc = erfc;
exports.erfinv = erfinv;
exports.randUniform = randUniform;
exports.randNormal = randNormal;
exports.randInt = randInt;
// Error function using Abramowitz and Stegun approximation
function erf(x) {
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    // Formula 7.1.26
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
}
// Complementary error function
function erfc(x) {
    return 1 - erf(x);
}
// Inverse error function using Winitzki approximation
function erfinv(x) {
    if (Math.abs(x) >= 1)
        throw new Error("Input must be in range (-1, 1)");
    const a = 0.147;
    const ln = Math.log(1 - x * x);
    const part1 = 2 / (Math.PI * a) + ln / 2;
    const part2 = ln / a;
    const sign = x >= 0 ? 1 : -1;
    return sign * Math.sqrt(-part1 + Math.sqrt(part1 * part1 - part2));
}
function randUniform(low = 0, high = 1) {
    return Math.random() * (high - low) + low;
}
function randNormal(mean = 0, stdDev = 1) {
    const u = 1 - Math.random();
    const v = 1 - Math.random();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * stdDev + mean;
}
function randInt(low, high) {
    return Math.floor(Math.random() * (high - low) + low);
}
