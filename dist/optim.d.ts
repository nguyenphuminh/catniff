import { Tensor } from "./core";
declare abstract class BaseOptimizer {
    params: Tensor[];
    constructor(params: Tensor[]);
    zeroGrad(): void;
}
export interface SGDOptions {
    lr?: number;
    momentum?: number;
    dampening?: number;
    weightDecay?: number;
    nesterov?: boolean;
}
declare class SGD extends BaseOptimizer {
    momentumBuffers: Map<Tensor, Tensor>;
    lr: number;
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
declare class Adam extends BaseOptimizer {
    momentumBuffers: Map<Tensor, Tensor>;
    velocityBuffers: Map<Tensor, Tensor>;
    stepCount: number;
    lr: number;
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
declare class AdamW extends BaseOptimizer {
    momentumBuffers: Map<Tensor, Tensor>;
    velocityBuffers: Map<Tensor, Tensor>;
    stepCount: number;
    lr: number;
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
export {};
