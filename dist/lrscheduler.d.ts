import { BaseOptimizer } from "./optim";
export declare class StepLR {
    optimizer: BaseOptimizer;
    stepSize: number;
    gamma: number;
    lastEpoch: number;
    baseLR: number;
    constructor(optimizer: BaseOptimizer, stepSize: number, gamma?: number, lastEpoch?: number);
    step(epoch?: number): void;
}
export declare const LRScheduler: {
    StepLR: typeof StepLR;
};
