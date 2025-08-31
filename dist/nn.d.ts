import { Tensor, TensorValue } from "./core";
declare class Linear {
    weight: Tensor;
    bias?: Tensor;
    constructor(inFeatures: number, outFeatures: number, bias?: boolean, device?: string);
    forward(input: Tensor | TensorValue): Tensor;
}
declare class RNNCell {
    weightIH: Tensor;
    weightHH: Tensor;
    biasIH?: Tensor;
    biasHH?: Tensor;
    constructor(inputSize: number, hiddenSize: number, bias?: boolean, device?: string);
    forward(input: Tensor | TensorValue, hidden: Tensor | TensorValue): Tensor;
}
declare class GRUCell {
    weightIR: Tensor;
    weightIZ: Tensor;
    weightIN: Tensor;
    weightHR: Tensor;
    weightHZ: Tensor;
    weightHN: Tensor;
    biasIR?: Tensor;
    biasIZ?: Tensor;
    biasIN?: Tensor;
    biasHR?: Tensor;
    biasHZ?: Tensor;
    biasHN?: Tensor;
    constructor(inputSize: number, hiddenSize: number, bias?: boolean, device?: string);
    forward(input: Tensor | TensorValue, hidden: Tensor | TensorValue): Tensor;
}
declare class LSTMCell {
    weightII: Tensor;
    weightIF: Tensor;
    weightIG: Tensor;
    weightIO: Tensor;
    weightHI: Tensor;
    weightHF: Tensor;
    weightHG: Tensor;
    weightHO: Tensor;
    biasII?: Tensor;
    biasIF?: Tensor;
    biasIG?: Tensor;
    biasIO?: Tensor;
    biasHI?: Tensor;
    biasHF?: Tensor;
    biasHG?: Tensor;
    biasHO?: Tensor;
    constructor(inputSize: number, hiddenSize: number, bias?: boolean, device?: string);
    forward(input: Tensor | TensorValue, hidden: Tensor | TensorValue, cell: Tensor | TensorValue): [Tensor, Tensor];
}
declare class LayerNorm {
    weight?: Tensor;
    bias?: Tensor;
    eps: number;
    normalizedShape: number[];
    constructor(normalizedShape: number | number[], eps?: number, elementwiseAffine?: boolean, bias?: boolean, device?: string);
    forward(input: Tensor): Tensor;
}
export interface StateDict {
    [key: string]: any;
}
export declare const nn: {
    Linear: typeof Linear;
    RNNCell: typeof RNNCell;
    GRUCell: typeof GRUCell;
    LSTMCell: typeof LSTMCell;
    LayerNorm: typeof LayerNorm;
    state: {
        getParameters(model: any, visited?: WeakSet<object>): Tensor[];
        moveParameters(model: any, device: string): void;
        getStateDict(model: any, prefix?: string, visited?: WeakSet<object>): StateDict;
        loadStateDict(model: any, stateDict: StateDict, prefix?: string, visited?: WeakSet<object>): void;
    };
};
export {};
