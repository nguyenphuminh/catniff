"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TypedArray = exports.dtypeHiearchy = void 0;
// Compatibility check
const float16Supported = typeof Float16Array !== "undefined";
if (!float16Supported) {
    console.log("Warning: Your JS runtime does not support Float16Array, falling back to Float32Array...");
}
exports.dtypeHiearchy = {
    "float64": 8,
    "float32": 7,
    "float16": 6,
    "int32": 5,
    "int16": 4,
    "int8": 3,
    "uint32": 2,
    "uint16": 1,
    "uint8": 0
};
exports.TypedArray = {
    "float64": Float64Array,
    "float32": Float32Array,
    "float16": float16Supported ? Float16Array : Float32Array,
    "int32": Int32Array,
    "int16": Int16Array,
    "int8": Int8Array,
    "uint32": Uint32Array,
    "uint16": Uint16Array,
    "uint8": Uint8Array
};
