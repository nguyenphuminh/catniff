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
export declare class Optim {
    static SGD: typeof SGD;
}
export {};
