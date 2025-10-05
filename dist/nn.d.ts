import { Tensor, TensorValue } from "./core";
export declare class Linear {
    weight: Tensor;
    bias?: Tensor;
    constructor(inFeatures: number, outFeatures: number, bias?: boolean, device?: string);
    forward(input: Tensor | TensorValue): Tensor;
}
export declare class RNNCell {
    weightIH: Tensor;
    weightHH: Tensor;
    biasIH?: Tensor;
    biasHH?: Tensor;
    constructor(inputSize: number, hiddenSize: number, bias?: boolean, device?: string);
    forward(input: Tensor | TensorValue, hidden: Tensor | TensorValue): Tensor;
}
export declare class GRUCell {
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
export declare class LSTMCell {
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
export declare class LayerNorm {
    weight?: Tensor;
    bias?: Tensor;
    eps: number;
    normalizedShape: number[];
    constructor(normalizedShape: number | number[], eps?: number, elementwiseAffine?: boolean, bias?: boolean, device?: string);
    forward(input: Tensor): Tensor;
}
export declare class RMSNorm {
    weight?: Tensor;
    eps: number;
    normalizedShape: number[];
    constructor(normalizedShape: number | number[], eps?: number, elementwiseAffine?: boolean, device?: string);
    forward(input: Tensor): Tensor;
}
export declare class Embedding {
    weight: Tensor;
    constructor(numEmbeddings: number, embeddingDim: number, device: string);
    forward(input: Tensor | TensorValue): Tensor;
}
export declare class MultiheadAttention {
    qProjection: Linear;
    kProjection: Linear;
    vProjection: Linear;
    oProjection: Linear;
    embedDim: number;
    numHeads: number;
    headDim: number;
    dropout: number;
    constructor(embedDim: number, numHeads: number, dropout?: number, bias?: boolean, device?: string);
    forward(query: Tensor, key: Tensor, value: Tensor, needWeights?: boolean, attnMask?: Tensor, averageAttnWeights?: boolean): [Tensor, Tensor | undefined];
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
    RMSNorm: typeof RMSNorm;
    Embedding: typeof Embedding;
    MultiheadAttention: typeof MultiheadAttention;
    state: {
        getParameters(model: any, visited?: WeakSet<object>): Tensor[];
        moveParameters(model: any, device: string): void;
        getStateDict(model: any, prefix?: string, visited?: WeakSet<object>): StateDict;
        loadStateDict(model: any, stateDict: StateDict, prefix?: string, visited?: WeakSet<object>): void;
    };
};
