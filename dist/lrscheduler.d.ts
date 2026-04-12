import { OptimizerWithLR } from "./optim";
export declare class StepLR {
    optimizer: OptimizerWithLR;
    stepSize: number;
    gamma: number;
    lastEpoch: number;
    baseLR: number;
    baseGroupLRs: number[];
    constructor(optimizer: OptimizerWithLR, stepSize: number, gamma?: number, lastEpoch?: number);
    step(): void;
}
export declare class LinearLR {
    optimizer: OptimizerWithLR;
    startFactor: number;
    endFactor: number;
    totalIters: number;
    lastEpoch: number;
    baseLR: number;
    baseGroupLRs: number[];
    constructor(optimizer: OptimizerWithLR, startFactor?: number, endFactor?: number, totalIters?: number, lastEpoch?: number);
    step(): void;
}
export declare class CosineAnnealingLR {
    optimizer: OptimizerWithLR;
    TMax: number;
    etaMin: number;
    lastEpoch: number;
    baseLR: number;
    baseGroupLRs: number[];
    constructor(optimizer: OptimizerWithLR, TMax: number, etaMin?: number, lastEpoch?: number);
    step(): void;
}
export interface Scheduler {
    step: Function;
}
export declare class SequentialLR {
    optimizer: OptimizerWithLR;
    schedulers: Scheduler[];
    milestones: number[];
    lastEpoch: number;
    constructor(optimizer: OptimizerWithLR, schedulers: Scheduler[], milestones: number[], lastEpoch?: number);
    step(): void;
}
export declare const LRScheduler: {
    StepLR: typeof StepLR;
    LinearLR: typeof LinearLR;
    CosineAnnealingLR: typeof CosineAnnealingLR;
    SequentialLR: typeof SequentialLR;
};
