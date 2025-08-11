import { Tensor, TensorValue } from "./core";
declare class Linear {
    weight: Tensor;
    bias?: Tensor;
    constructor(inFeatures: number, outFeatures: number, bias?: boolean, customInit?: (shape: number[]) => Tensor);
    forward(input: Tensor | TensorValue): Tensor;
}
export declare const nn: {
    Linear: typeof Linear;
    state: {
        getParameters(model: any, visited?: WeakSet<object>): Tensor[];
    };
};
export {};
