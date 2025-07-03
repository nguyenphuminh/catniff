export type Tensor = number | Tensor[];
export declare class TensorMath {
    static create(num: number, shape: number[]): Tensor;
    static getShape(tA: Tensor): number[];
    static add(tA: Tensor, tB: Tensor): Tensor;
    static sub(tA: Tensor, tB: Tensor): Tensor;
    static mul(tA: Tensor, tB: Tensor): Tensor;
    static pow(tA: Tensor, tB: Tensor): Tensor;
    static div(tA: Tensor, tB: Tensor): Tensor;
    static gt(tA: Tensor, tB: Tensor): Tensor;
    static lt(tA: Tensor, tB: Tensor): Tensor;
    static ge(tA: Tensor, tB: Tensor): Tensor;
    static le(tA: Tensor, tB: Tensor): Tensor;
    static neg(tA: Tensor): Tensor;
    static exp(tA: Tensor): Tensor;
    static log(tA: Tensor): Tensor;
    static relu(tA: Tensor): Tensor;
    static sigmoid(tA: Tensor): Tensor;
    static tanh(tA: Tensor): Tensor;
}
