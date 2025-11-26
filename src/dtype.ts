export type dtype = "float64" | "float32" | "float16" | "int32" | "int16" | "int8" | "uint32" | "uint16" | "uint8";

export const dtypeHiearchy: Record<dtype, number> = {
    "float64": 8,
    "float32": 7,
    "float16": 6,
    "int32": 5,
    "int16": 4,
    "int8": 3,
    "uint32": 2,
    "uint16": 1,
    "uint8": 0
}

export type MemoryBuffer = 
    Float64Array |
    Float32Array |
    Float16Array |
    Int32Array   |
    Int16Array   |
    Int8Array    |
    Uint32Array  |
    Uint16Array  |
    Uint8Array;

export type TypedArrayConstructor = 
    Float64ArrayConstructor |
    Float32ArrayConstructor |
    Float16ArrayConstructor |
    Int32ArrayConstructor   |
    Int16ArrayConstructor   |
    Int8ArrayConstructor    |
    Uint32ArrayConstructor  |
    Uint16ArrayConstructor  |
    Uint8ArrayConstructor;

export const TypedArray: Record<dtype, TypedArrayConstructor> = {
    "float64": Float64Array,
    "float32": Float32Array,
    "float16": Float16Array,
    "int32": Int32Array,
    "int16": Int16Array,
    "int8": Int8Array,
    "uint32": Uint32Array,
    "uint16": Uint16Array,
    "uint8": Uint8Array
}
