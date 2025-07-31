import { Tensor } from "./core";
export interface SGDOptions {
    lr?: number;
    momentum?: number;
    dampening?: number;
    weightDecay?: number;
    nesterov?: boolean;
}
declare class SGD {
    params: Tensor[];
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
declare class Adam {
    params: Tensor[];
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
export declare class Optim {
    static SGD: typeof SGD;
    static Adam: typeof Adam;
}
export {};
