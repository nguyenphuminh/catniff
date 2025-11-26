export type dtype = "float64" | "float32" | "float16" | "int32" | "int16" | "int8" | "uint32" | "uint16" | "uint8";
export declare const dtypeHiearchy: Record<dtype, number>;
export type MemoryBuffer = Float64Array | Float32Array | Float16Array | Int32Array | Int16Array | Int8Array | Uint32Array | Uint16Array | Uint8Array;
export type TypedArrayConstructor = Float64ArrayConstructor | Float32ArrayConstructor | Float16ArrayConstructor | Int32ArrayConstructor | Int16ArrayConstructor | Int8ArrayConstructor | Uint32ArrayConstructor | Uint16ArrayConstructor | Uint8ArrayConstructor;
export declare const TypedArray: Record<dtype, TypedArrayConstructor>;
