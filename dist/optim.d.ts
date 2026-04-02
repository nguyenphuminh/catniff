import { Tensor } from "./core";
export interface BaseOptimizerOptions {
    lr?: number;
}
export declare abstract class BaseOptimizer {
    params: Tensor[];
    constructor(params: Tensor[], options?: BaseOptimizerOptions);
    zeroGrad(): void;
}
export interface SGDOptions {
    lr?: number;
    momentum?: number;
    dampening?: number;
    weightDecay?: number;
    nesterov?: boolean;
}
export interface OptimizerWithLR extends BaseOptimizer {
    lr: number;
}
export declare class SGD extends BaseOptimizer {
    lr: number;
    momentumBuffers: Map<Tensor, Tensor>;
    momentum: number;
    dampening: number;
    weightDecay: number;
    nesterov: boolean;
    constructor(params: Tensor[], options?: SGDOptions);
    step(): void;
}
export interface AdamOptions {
    lr?: number;
    betas?: [number, number];
    eps?: number;
    weightDecay?: number;
}
export declare class Adam extends BaseOptimizer {
    lr: number;
    momentumBuffers: Map<Tensor, Tensor>;
    velocityBuffers: Map<Tensor, Tensor>;
    stepCount: number;
    betas: [number, number];
    eps: number;
    weightDecay: number;
    constructor(params: Tensor[], options?: AdamOptions);
    step(): void;
}
export interface AdamWOptions {
    lr?: number;
    betas?: [number, number];
    eps?: number;
    weightDecay?: number;
}
export declare class AdamW extends BaseOptimizer {
    lr: number;
    momentumBuffers: Map<Tensor, Tensor>;
    velocityBuffers: Map<Tensor, Tensor>;
    stepCount: number;
    betas: [number, number];
    eps: number;
    weightDecay: number;
    constructor(params: Tensor[], options?: AdamWOptions);
    step(): void;
}
export declare const Optim: {
    BaseOptimizer: typeof BaseOptimizer;
    SGD: typeof SGD;
    Adam: typeof Adam;
    AdamW: typeof AdamW;
};
