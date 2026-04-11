import { OptimizerWithLR } from "./optim";

export class StepLR {
    public optimizer: OptimizerWithLR;
    public stepSize: number;
    public gamma: number;
    public lastEpoch: number;
    public baseLR: number;
    public baseGroupLRs: number[];

    constructor(optimizer: OptimizerWithLR, stepSize: number, gamma = 0.1, lastEpoch = -1) {
        this.optimizer = optimizer;
        this.stepSize = stepSize;
        this.gamma = gamma;
        this.lastEpoch = lastEpoch;
        this.baseLR = optimizer.lr;
        this.baseGroupLRs = this.optimizer.paramGroups.map(paramGroup => paramGroup.lr ?? this.optimizer.lr);
    }

    step() {
        this.lastEpoch++;

        // Update LR of each group
        for (let index = 0; index < this.baseGroupLRs.length; index++) {
            this.optimizer.paramGroups[index].lr = this.baseGroupLRs[index] * this.gamma ** Math.floor(this.lastEpoch / this.stepSize);
        }

        // Update default LR
        this.optimizer.lr = this.baseLR * this.gamma ** Math.floor(this.lastEpoch / this.stepSize);
    }
}

export class LinearLR {
    public optimizer: OptimizerWithLR;
    public startFactor: number;
    public endFactor: number;
    public totalIters: number;
    public lastEpoch: number;
    public baseLR: number;
    public baseGroupLRs: number[];

    constructor(optimizer: OptimizerWithLR, startFactor = 0.3333333333333333, endFactor = 1, totalIters = 5, lastEpoch = -1) {
        this.optimizer = optimizer;
        this.startFactor = startFactor;
        this.endFactor = endFactor;
        this.totalIters = totalIters;
        this.lastEpoch = lastEpoch;
        this.baseLR = optimizer.lr;
        this.baseGroupLRs = this.optimizer.paramGroups.map(paramGroup => paramGroup.lr ?? this.optimizer.lr);
    }

    step() {
        this.lastEpoch++;

        // Clamp under total allowed iterations
        const t = Math.min(this.lastEpoch, this.totalIters);
        // Precalculate factor
        const factor = this.startFactor + (t / this.totalIters) * (this.endFactor - this.startFactor);

        // Update LR of each group
        for (let index = 0; index < this.baseGroupLRs.length; index++) {
            this.optimizer.paramGroups[index].lr = this.baseGroupLRs[index] * factor;
        }

        // Update default LR
        this.optimizer.lr = this.baseLR * factor;
    }
}

export class CosineAnnealingLR {
    public optimizer: OptimizerWithLR;
    public TMax: number;
    public etaMin: number;
    public lastEpoch: number;
    public baseLR: number;
    public baseGroupLRs: number[];

    constructor(optimizer: OptimizerWithLR, TMax: number, etaMin = 0, lastEpoch = -1) {
        this.optimizer = optimizer;
        this.TMax = TMax;
        this.etaMin = etaMin;
        this.lastEpoch = lastEpoch;
        this.baseLR = optimizer.lr;
        this.baseGroupLRs = this.optimizer.paramGroups.map(paramGroup => paramGroup.lr ?? this.optimizer.lr);
    }

    step() {
        this.lastEpoch++;

        const cosine = (1 + Math.cos((this.lastEpoch * Math.PI) / this.TMax)) / 2;

        for (let index = 0; index < this.baseGroupLRs.length; index++) {
            this.optimizer.paramGroups[index].lr = this.etaMin + (this.baseGroupLRs[index] - this.etaMin) * cosine;
        }

        this.optimizer.lr = this.etaMin + (this.baseLR - this.etaMin) * cosine;
    }
}

export const LRScheduler = {
    StepLR,
    LinearLR,
    CosineAnnealingLR
}
