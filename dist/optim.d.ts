import { Tensor } from "./core";
export interface BaseParamGroup {
    params: Tensor[];
    [key: string]: any;
}
export declare abstract class BaseOptimizer {
    paramGroups: BaseParamGroup[];
    constructor(params: Tensor[] | BaseParamGroup[]);
    zeroGrad(del?: boolean): void;
}
export interface OptimizerWithLR extends BaseOptimizer {
    lr: number;
}
export interface SGDOptions {
    lr?: number;
    momentum?: number;
    dampening?: number;
    weightDecay?: number;
    nesterov?: boolean;
}
export interface SGDParamGroup extends SGDOptions {
    params: Tensor[];
}
export declare class SGD extends BaseOptimizer {
    paramGroups: SGDParamGroup[];
    lr: number;
    momentum: number;
    dampening: number;
    weightDecay: number;
    nesterov: boolean;
    momentumBuffers: Map<Tensor, Tensor>;
    constructor(params: Tensor[] | SGDParamGroup[], options?: SGDOptions);
    step(): void;
}
export interface AdamOptions {
    lr?: number;
    betas?: [number, number];
    eps?: number;
    weightDecay?: number;
}
export interface AdamParamGroup extends AdamOptions {
    params: Tensor[];
}
export declare class Adam extends BaseOptimizer {
    paramGroups: AdamParamGroup[];
    lr: number;
    betas: [number, number];
    eps: number;
    weightDecay: number;
    momentumBuffers: Map<Tensor, Tensor>;
    velocityBuffers: Map<Tensor, Tensor>;
    stepCounts: Map<Tensor, number>;
    constructor(params: Tensor[] | AdamParamGroup[], options?: AdamOptions);
    step(): void;
}
export interface AdamWOptions {
    lr?: number;
    betas?: [number, number];
    eps?: number;
    weightDecay?: number;
}
export interface AdamWParamGroup extends AdamWOptions {
    params: Tensor[];
}
export declare class AdamW extends BaseOptimizer {
    paramGroups: AdamWParamGroup[];
    lr: number;
    betas: [number, number];
    eps: number;
    weightDecay: number;
    momentumBuffers: Map<Tensor, Tensor>;
    velocityBuffers: Map<Tensor, Tensor>;
    stepCounts: Map<Tensor, number>;
    constructor(params: Tensor[] | AdamWParamGroup[], options?: AdamWOptions);
    step(): void;
}
export declare const Optim: {
    BaseOptimizer: typeof BaseOptimizer;
    SGD: typeof SGD;
    Adam: typeof Adam;
    AdamW: typeof AdamW;
};
