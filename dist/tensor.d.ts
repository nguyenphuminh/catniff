export type Tensor = number | Tensor[];
export declare class TensorMath {
    static create(num: number, shape: number[]): Tensor;
    static getShape(tA: Tensor): number[];
    static padShape(tA: Tensor, tB: Tensor): [Tensor[], Tensor[]];
    static add(tA: Tensor, tB: Tensor): Tensor;
    static sub(tA: Tensor, tB: Tensor): Tensor;
    static mul(tA: Tensor, tB: Tensor): Tensor;
    static pow(tA: Tensor, tB: Tensor): Tensor;
    static div(tA: Tensor, tB: Tensor): Tensor;
    static gt(tA: Tensor, tB: Tensor): Tensor;
    static lt(tA: Tensor, tB: Tensor): Tensor;
    static ge(tA: Tensor, tB: Tensor): Tensor;
    static le(tA: Tensor, tB: Tensor): Tensor;
    static eq(tA: Tensor, tB: Tensor): Tensor;
    static neg(tA: Tensor): Tensor;
    static exp(tA: Tensor): Tensor;
    static log(tA: Tensor): Tensor;
    static relu(tA: Tensor): Tensor;
    static sigmoid(tA: Tensor): Tensor;
    static tanh(tA: Tensor): Tensor;
    static squeezeAxis(tA: Tensor, axis: number): Tensor;
    static squeeze(tA: Tensor, dims?: number[] | number): Tensor;
    static sumAxis(tA: Tensor, axis: number): Tensor;
    static sum(tA: Tensor, dims?: number[] | number, keepDims?: boolean): Tensor;
    static t(tA: Tensor): Tensor;
    static mm(tA: Tensor, tB: Tensor): Tensor;
}
