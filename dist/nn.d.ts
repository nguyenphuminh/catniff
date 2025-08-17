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
export declare const nn: {
    Linear: typeof Linear;
    RNNCell: typeof RNNCell;
    state: {
        getParameters(model: any, visited?: WeakSet<object>): Tensor[];
    };
};
export {};
